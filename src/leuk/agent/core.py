"""Core agent: agentic loop with tool dispatch."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator

from leuk.agent.context import compact
from leuk.config import PermissionAction, Settings
from leuk.persistence.base import HotStore
from leuk.persistence.sqlite import SQLiteStore
from leuk.providers.base import LLMProvider
from leuk.safety import SafetyGuard
from leuk.tools.base import ToolRegistry
from leuk.types import (
    Message,
    Role,
    Session,
    SessionStatus,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger(__name__)


def dangling_user_input(messages: list[Message]) -> str | None:
    """Return the last user turn that never got an assistant reply, else None.

    A crash (or SIGKILL/OOM) mid-turn leaves the user message persisted with no
    assistant text after it. The REPL uses this on session load to offer a
    one-keystroke ``/retry`` of the unfinished turn (refactor-plan §5.6).
    """
    last_user: str | None = None
    replied = False
    for m in messages:
        if m.role is Role.USER:
            content = (m.content or "").strip()
            if content and not content.startswith(("[SYSTEM]", "[STEERING]")):
                last_user = content
                replied = False
        elif m.role is Role.ASSISTANT and (m.content or "").strip():
            replied = True
    return None if replied else last_user

# ── Context-overflow detection ────────────────────────────────────

_MAX_OVERFLOW_RETRIES = 3

# Server-reported limits, per provider phrasing.
_CTX_LIMIT_RES = [
    # llama.cpp llama-server: "request (21873 tokens) exceeds the available
    # context size (17920 tokens), try increasing it"
    re.compile(r"exceeds the available context size \((\d+) tokens?\)"),
    # OpenAI-compatible: "This model's maximum context length is 8192 tokens"
    re.compile(r"maximum context length is (\d+)"),
    # Anthropic: "prompt is too long: 213071 tokens > 200000 maximum"
    re.compile(r"prompt is too long: \d+ tokens > (\d+) maximum"),
]
_CTX_OVERFLOW_HINTS = (
    "context size",
    "context length",
    "context_length_exceeded",
    "prompt is too long",
    "context limit",
)


def context_overflow_limit(exc: Exception) -> int | None:
    """The server-reported context limit when *exc* is a context-overflow error.

    Returns the limit in tokens, ``0`` when the error is clearly an overflow
    but states no number, or ``None`` when it isn't an overflow error at all.
    Used to compact harder and retry instead of failing the turn.
    """
    text = str(exc)
    for rx in _CTX_LIMIT_RES:
        m = rx.search(text)
        if m:
            return int(m.group(1))
    low = text.lower()
    if any(h in low for h in _CTX_OVERFLOW_HINTS):
        return 0
    return None


# ── Rate-limit detection ──────────────────────────────────────────

_MAX_RETRIES = 3
_MAX_BACKOFF = 120  # seconds
_RETRY_DELAY_RE = re.compile(r"retry\s+(?:in|after)\s+([\d.]+)\s*s", re.IGNORECASE)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a rate-limit (HTTP 429) error."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return True
    cls_name = type(exc).__name__
    if "RateLimit" in cls_name:
        return True
    # Google wraps 429 as ClientError with RESOURCE_EXHAUSTED
    msg = str(exc)[:300]
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        return True
    return False


def _extract_retry_delay(exc: Exception) -> float | None:
    """Try to parse a retry delay from the error message."""
    msg = str(exc)
    m = _RETRY_DELAY_RE.search(msg)
    if m:
        try:
            return min(float(m.group(1)), _MAX_BACKOFF)
        except ValueError:
            pass
    return None


class Agent:
    """The core agent that runs the generate-execute loop.

    Lifecycle:
        agent = Agent(...)
        await agent.init()
        async for msg in agent.run("Hello"):
            print(msg)
        await agent.shutdown()
    """

    def __init__(
        self,
        *,
        settings: Settings,
        provider: LLMProvider,
        tool_registry: ToolRegistry,
        sqlite: SQLiteStore,
        hot_store: HotStore,
        session: Session | None = None,
        safety_guard: SafetyGuard | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.tools = tool_registry
        self.sqlite = sqlite
        self.hot_store = hot_store
        self.session = session or Session(system_prompt=settings.agent.system_prompt)
        self.safety_guard = safety_guard
        self._stop_requested = False
        self._messages: list[Message] = []
        # Media (images/audio) to attach to the next user message, set by the
        # REPL via /file before the turn is pushed.
        self.pending_attachments: list[Any] | None = None
        # Learned from the server's own context-overflow errors: an upper bound
        # on the usable window. Persists for the agent's lifetime so later
        # turns pre-compact instead of re-hitting the limit.
        self._window_clamp: int | None = None
        self._overflow_retries = 0  # per-turn; reset in run()/run_stream()
        # Steering / persistence-guard state (reset per turn; see steering.py).
        self._continuations = 0  # self-reflection nudges spent this turn
        self._round = 0  # current tool-round index, read by _prepare_context
        self._last_round_had_error = False  # previous round produced a tool error
        self._used_tools_this_turn = False  # any tool ran since the user's input
        self._round_signatures: list[str] = []  # per-round tool-call signatures
        self._loop_interventions = 0  # circle-breaker nudges spent this turn

    def _tighten_window(self, reported_limit: int) -> None:
        """Shrink the effective window after a context-overflow error.

        Uses the server-reported limit when stated, applying a growing safety
        margin on successive retries — the char-based token estimator can
        undercount relative to the server's real tokenizer.
        """
        from leuk.agent.context import estimate_total_tokens

        base = reported_limit or self._window_clamp or estimate_total_tokens(self._messages)
        factor = max(0.5, 0.9 - 0.15 * (self._overflow_retries - 1))
        self._window_clamp = max(2048, int(base * factor))

    async def init(self) -> None:
        """Initialise persistence and load existing session state."""
        await self.sqlite.init()

        existing = await self.sqlite.get_session(self.session.id)
        if existing:
            self.session = existing
            self._messages = await self.sqlite.get_messages(self.session.id)
            logger.info("Resumed session %s with %d messages", self.session.id, len(self._messages))
        else:
            await self.sqlite.create_session(self.session)
            logger.info("Created new session %s", self.session.id)

        # Prepend system prompt if not already present. For weak/local models,
        # steering augments it with persistence/recovery discipline (a no-op for
        # strong models). Done here — not on the stored prompt — so the persisted
        # session prompt and sub-agent role prompts stay clean, and it keys on the
        # effective provider.
        if not self._messages or self._messages[0].role != Role.SYSTEM:
            from leuk.agent.steering import compose_system_prompt

            content = compose_system_prompt(
                self.session.system_prompt,
                self.settings.steering,
                self.settings.llm.provider,
            )
            sys_msg = Message(role=Role.SYSTEM, content=content)
            self._messages.insert(0, sys_msg)

        # Wire the history tool to this session's full stored conversation, so
        # the model can navigate everything compaction summarized away.
        from leuk.tools.history import HistoryTool

        history_tool = self.tools.get("history")
        if isinstance(history_tool, HistoryTool):
            session_id = self.session.id
            sqlite = self.sqlite

            async def _full_history() -> list[Message]:
                return await sqlite.get_messages(session_id)

            history_tool.set_source(_full_history)

        await self.hot_store.set_active_session(self.session.id)

    async def run(self, user_input: str) -> AsyncIterator[Message]:
        """Process a user message through the agent loop.

        Yields each assistant/tool message as it's produced.
        """
        # Add user message
        user_msg = Message(
            role=Role.USER, content=user_input, attachments=self.pending_attachments or None
        )
        self.pending_attachments = None
        self._messages.append(user_msg)
        await self._persist_message(user_msg)

        max_rounds = self.settings.agent.max_tool_rounds
        self._overflow_retries = 0  # fresh budget for this turn
        self._reset_steering_state()

        for _round in range(max_rounds):
            self._round = _round
            # Apply context management before calling LLM
            context = await self._prepare_context()
            assistant_msg = await self._generate_with_retry(context)
            self._maybe_salvage_tool_calls(assistant_msg)
            self._messages.append(assistant_msg)
            await self._persist_message(assistant_msg)
            yield assistant_msg

            # No tool calls: the model thinks it's done. Steering may disagree
            # (refusal / premature stop / truncation) and nudge it to continue,
            # bounded by max_continuations. The nudge is internal — not yielded.
            if not assistant_msg.tool_calls:
                nudge = await self._reflect_and_maybe_continue(assistant_msg)
                if nudge is None:
                    break
                self._messages.append(nudge)
                await self._persist_message(nudge)
                continue

            # Execute all tool calls
            round_had_error = False
            for tc in assistant_msg.tool_calls:
                result = await self._execute_tool(tc)
                round_had_error = round_had_error or result.is_error
                tool_msg = Message(role=Role.TOOL, tool_result=result)
                self._messages.append(tool_msg)
                await self._persist_message(tool_msg)
                yield tool_msg
            self._used_tools_this_turn = True
            self._last_round_had_error = round_had_error

            # Stop the loop if a tool call was denied by the user.
            if self._stop_requested:
                self._stop_requested = False
                break

            # Anti-circling: break a no-progress loop early instead of spinning
            # to max_tool_rounds. On escalation, consolidate with a tools-off reply.
            circle = await self._maybe_break_circle(assistant_msg)
            if circle == "escalate":
                logger.warning("Circling detected — forcing a consolidation reply")
                context = await self._prepare_context()
                final = await self._generate_with_retry(context, tools=None)
                self._messages.append(final)
                await self._persist_message(final)
                yield final
                break
        else:
            # Exceeded max rounds -- force a text reply
            logger.warning("Hit max tool rounds (%d), forcing text reply", max_rounds)
            forced = Message(
                role=Role.USER,
                content="[SYSTEM] You have exceeded the maximum number of tool-use rounds. Please provide a final text response.",
            )
            self._messages.append(forced)
            context = await self._prepare_context()
            final = await self._generate_with_retry(context, tools=None)
            self._messages.append(final)
            await self._persist_message(final)
            yield final

        # Cache context in hot store
        await self._cache_context()

    async def run_stream(self, user_input: str) -> AsyncIterator[StreamEvent | Message]:
        """Process a user message with streaming.

        Yields StreamEvent for real-time token delivery, and Message objects
        for tool results (same as run()). The assistant message is also
        yielded as MESSAGE_COMPLETE inside StreamEvent.

        The generator is cancellation-safe: if the consuming task is
        cancelled (e.g. by :meth:`AgentSession.interrupt`), the current
        provider stream is broken out of and partial text already collected
        is persisted as an incomplete assistant message so context is not
        lost.
        """
        user_msg = Message(
            role=Role.USER, content=user_input, attachments=self.pending_attachments or None
        )
        self.pending_attachments = None
        self._messages.append(user_msg)
        await self._persist_message(user_msg)

        max_rounds = self.settings.agent.max_tool_rounds
        text_parts: list[str] = []  # accumulate text deltas for interrupt persistence
        self._overflow_retries = 0  # fresh budget for this turn
        self._reset_steering_state()

        try:
            for _round in range(max_rounds):
                self._round = _round
                context = await self._prepare_context()
                assistant_msg: Message | None = None
                text_parts.clear()

                async for event in self._stream_with_retry(context):
                    yield event
                    if event.type == StreamEventType.TEXT_DELTA:
                        text_parts.append(event.content)
                    elif event.type == StreamEventType.MESSAGE_COMPLETE:
                        assistant_msg = event.message

                if assistant_msg is None:
                    break

                self._maybe_salvage_tool_calls(assistant_msg)
                self._messages.append(assistant_msg)
                await self._persist_message(assistant_msg)
                text_parts.clear()  # persisted successfully

                # No tool calls: steering may nudge a premature stop to continue
                # (bounded). The nudge is internal — never yielded to the renderer.
                if not assistant_msg.tool_calls:
                    nudge = await self._reflect_and_maybe_continue(assistant_msg)
                    if nudge is None:
                        break
                    self._messages.append(nudge)
                    await self._persist_message(nudge)
                    continue

                round_had_error = False
                for tc in assistant_msg.tool_calls:
                    result = await self._execute_tool(tc)
                    round_had_error = round_had_error or result.is_error
                    tool_msg = Message(role=Role.TOOL, tool_result=result)
                    self._messages.append(tool_msg)
                    await self._persist_message(tool_msg)
                    yield tool_msg
                self._used_tools_this_turn = True
                self._last_round_had_error = round_had_error

                # Stop the loop if a tool call was denied by the user.
                if self._stop_requested:
                    self._stop_requested = False
                    break

                # Anti-circling: break a no-progress loop early; on escalation,
                # consolidate with a tools-off reply (streamed).
                circle = await self._maybe_break_circle(assistant_msg)
                if circle == "escalate":
                    logger.warning("Circling detected — forcing a consolidation reply (stream)")
                    context = await self._prepare_context()
                    async for event in self._stream_with_retry(context, tools=None):
                        yield event
                        if event.type == StreamEventType.TEXT_DELTA:
                            text_parts.append(event.content)
                        elif event.type == StreamEventType.MESSAGE_COMPLETE and event.message:
                            self._messages.append(event.message)
                            await self._persist_message(event.message)
                            text_parts.clear()
                    break
            else:
                logger.warning("Hit max tool rounds (%d) in stream mode", max_rounds)
                forced = Message(
                    role=Role.USER,
                    content="[SYSTEM] You have exceeded the maximum number of tool-use rounds. Please provide a final text response.",
                )
                self._messages.append(forced)
                context = await self._prepare_context()
                async for event in self._stream_with_retry(context, tools=None):
                    yield event
                    if event.type == StreamEventType.TEXT_DELTA:
                        text_parts.append(event.content)
                    elif event.type == StreamEventType.MESSAGE_COMPLETE and event.message:
                        self._messages.append(event.message)
                        await self._persist_message(event.message)
                        text_parts.clear()

        except (asyncio.CancelledError, GeneratorExit):
            # Interrupted mid-stream: persist whatever text we collected
            if text_parts:
                partial = Message(
                    role=Role.ASSISTANT,
                    content="".join(text_parts) + "\n\n[interrupted]",
                )
                self._messages.append(partial)
                await self._persist_message(partial)
                text_parts.clear()
            # Heal any orphaned tool_use messages (tool_calls without a
            # matching tool_result) left behind by the interrupted round —
            # otherwise the next turn would see them and the provider would
            # reject the request.
            await self._heal_orphaned_tool_calls()
            raise
        except Exception:
            # Non-cancellation errors (timeout, provider 5xx, etc.) bubble
            # up to the AgentSession, which surfaces them to the user.
            # Heal orphans first so the next /retry or new message doesn't
            # find a broken tool_use/tool_result chain.
            await self._heal_orphaned_tool_calls()
            raise

        await self._cache_context()

    async def _heal_orphaned_tool_calls(self) -> None:
        """Append placeholder tool_result messages for any orphaned tool_calls.

        Called from ``run_stream``'s exception handlers so a half-completed
        tool round (interrupt, network error, …) doesn't leave the message
        history in a state the provider will reject on the next call.

        Mutates ``self._messages`` in place and persists the placeholders.
        """
        # Collect IDs of tool_calls that already have results.
        result_ids: set[str] = set()
        for msg in self._messages:
            if msg.tool_result:
                result_ids.add(msg.tool_result.tool_call_id)

        # Find any tool_call without a matching result.
        orphans: list[ToolCall] = []
        for msg in self._messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.id not in result_ids:
                        orphans.append(tc)

        if not orphans:
            return

        logger.debug("Healing %d orphaned tool_call(s) after interrupted turn", len(orphans))
        for tc in orphans:
            placeholder = Message(
                role=Role.TOOL,
                tool_result=ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content="[Tool execution interrupted before completion]",
                    is_error=True,
                ),
            )
            self._messages.append(placeholder)
            try:
                await self._persist_message(placeholder)
            except Exception:  # noqa: BLE001 — best-effort persistence
                logger.exception("Failed to persist healing placeholder for %s", tc.id)

    async def _prepare_context(self) -> list[Message]:
        """Apply context window management to the current message history.

        Returns a (possibly truncated/summarized) copy suitable for the LLM.
        The internal _messages list is NOT modified -- only the copy sent to the
        LLM is affected.
        """
        cfg = self.settings.agent

        # Query the model's own metadata once (cached) — used for BOTH the
        # compaction budget and the vision check. Never name-guessed.
        info = None
        model_info = getattr(self.provider, "model_info", None)
        if callable(model_info):
            try:
                info = await model_info()
            except Exception:  # noqa: BLE001 — capability query is best-effort
                info = None

        # The compaction budget is derived from the model's *own* context window
        # (queried, or the LEUK_LLM_CONTEXT_WINDOW param), reserving room for the
        # reply — not a hardcoded value. ``max_context_tokens`` is only an
        # explicit override; a fixed fallback is used solely when the window is
        # genuinely undeterminable and unconfigured.
        from leuk.agent.context import compaction_budget

        window = (info.context_window if info else None) or self.settings.llm.context_window
        # A clamp learned from the server's own overflow errors wins over a
        # larger (or unknown) queried window.
        if self._window_clamp:
            window = min(window, self._window_clamp) if window else self._window_clamp
        budget = compaction_budget(
            window,
            override=cfg.max_context_tokens,
            reserve=self.settings.llm.max_tokens,
        )
        if self._window_clamp:
            # The learned limit also caps an explicit max_context_tokens
            # override — the server has the final word on what fits.
            budget = min(
                budget,
                compaction_budget(self._window_clamp, reserve=self.settings.llm.max_tokens),
            )

        archive_cfg = self.settings.archive
        archive_kwargs: dict[str, str] = {}
        if archive_cfg.enabled:
            archive_kwargs["session_id"] = self.session.id
            archive_kwargs["archive_dir"] = archive_cfg.directory

        context = await compact(
            self._messages,
            self.provider,
            max_tokens=budget,
            max_result_tokens=cfg.max_tool_result_tokens,
            **archive_kwargs,
        )

        # Images/video are sent to the model natively (provider image blocks),
        # never as base64 text. Only when the query reports vision is
        # **definitely** absent do we strip the media (leaving a note) — so the
        # model never gets a base64 blob to "read" as text. Unknown → send
        # natively and let the API tell us (no name-based guessing).
        if info is not None and info.supports_vision is False:
            from leuk.media import strip_media

            context = strip_media(
                context,
                note=f"the active model '{self.settings.llm.model}' has no "
                "vision support; switch to a vision-capable model to analyse it",
            )

        # Steering: re-inject a short reminder for weak/local models that lose
        # the system prompt over long contexts. Into the context COPY only — never
        # persisted or compacted. Cadence: every N rounds, or right after an error.
        from leuk.agent.steering import STEERING_REMINDER, steering_active

        scfg = self.settings.steering
        if steering_active(scfg, self.settings.llm.provider):
            interval = scfg.reminder_interval
            due = bool(interval) and self._round > 0 and self._round % interval == 0
            if due or self._last_round_had_error:
                context = [*context, Message(role=Role.USER, content=STEERING_REMINDER)]
        return context

    # ── Steering / persistence guard ──────────────────────────────

    def _reset_steering_state(self) -> None:
        """Reset per-turn steering counters at the start of each user turn."""
        self._continuations = 0
        self._round = 0
        self._last_round_had_error = False
        self._used_tools_this_turn = False
        self._round_signatures = []
        self._loop_interventions = 0

    def _last_tool_result(self) -> ToolResult | None:
        """The most recent tool result in the conversation, if any."""
        for m in reversed(self._messages):
            if m.role is Role.TOOL and m.tool_result is not None:
                return m.tool_result
        return None

    def _maybe_recovery_hint(self, content: str) -> str:
        """Append a steering recovery hint to an errored tool result when active."""
        from leuk.agent.steering import RECOVERY_HINT, steering_active

        cfg = self.settings.steering
        if cfg.enrich_tool_errors and steering_active(cfg, self.settings.llm.provider):
            return content + RECOVERY_HINT
        return content

    def _maybe_salvage_tool_calls(self, msg: Message) -> None:
        """Recover tool calls a weak model emitted as plain text into real calls.

        Some local models write the call into the message content (e.g.
        ``<tool_call><function=…><parameter=…>``) instead of using the native
        function-calling API, so the round would otherwise end with nothing run.
        Mutates *msg* in place (before persistence). Gated by steering + validated
        against registered tool names, so it never misreads ordinary prose.
        """
        if msg.tool_calls or not msg.content:
            return
        from leuk.agent.steering import parse_text_tool_calls, steering_active

        cfg = self.settings.steering
        if not cfg.salvage_text_tool_calls or not steering_active(cfg, self.settings.llm.provider):
            return
        salvaged = parse_text_tool_calls(msg.content, {s.name for s in self.tools.specs()})
        if salvaged:
            msg.tool_calls = salvaged
            logger.info("Salvaged %d text-encoded tool call(s) from model output", len(salvaged))

    async def _reflect_and_maybe_continue(self, assistant_msg: Message) -> Message | None:
        """Decide whether a no-tool-call stop is genuine or a premature give-up.

        Returns a steering nudge ``Message`` to inject (then ``continue`` the
        loop), or ``None`` to accept the stop. Bounded by
        ``steering.max_continuations`` so it can never loop forever, and a no-op
        when steering is inactive (today's behaviour for strong models).
        """
        from leuk.agent.steering import (
            REFLECTION_PROMPT,
            continue_nudge,
            parse_reflection,
            steering_active,
            truncation_nudge,
        )

        cfg = self.settings.steering
        if not steering_active(cfg, self.settings.llm.provider):
            return None
        # Never nudge against a user's tool denial.
        last = self._last_tool_result()
        if last is not None and last.content.startswith("[BLOCKED]"):
            return None
        # Hard inner cap — independent of (and bounded by) max_tool_rounds.
        if self._continuations >= cfg.max_continuations:
            return None
        # Truncation fast-path: a cut-off reply just needs to continue; no
        # self-reflection call spent.
        if cfg.nudge_on_truncation and assistant_msg.metadata.get("finish_reason") == "length":
            self._continuations += 1
            return Message(role=Role.USER, content=truncation_nudge())
        # Skip the reflection call for casual chat (no tool ran this turn);
        # first-turn refusals are handled by the system-prompt steering instead.
        if cfg.reflect_only_after_tool_use and not self._used_tools_this_turn:
            return None
        # Ask the model itself whether the task is actually complete.
        try:
            reflect_ctx = await self._prepare_context()
            reflect_ctx = [*reflect_ctx, Message(role=Role.USER, content=REFLECTION_PROMPT)]
            verdict = await self.provider.generate(
                reflect_ctx,
                tools=None,
                max_tokens=cfg.reflection_max_tokens,
                temperature=0.0,
            )
        except Exception:  # noqa: BLE001 — reflection is best-effort
            logger.exception("Steering reflection failed; accepting the stop")
            return None
        should_continue, hint = parse_reflection(verdict.content)
        if not should_continue:
            return None
        self._continuations += 1
        return Message(role=Role.USER, content=continue_nudge(hint))

    async def _maybe_break_circle(self, assistant_msg: Message) -> str | None:
        """Detect a no-progress tool loop in a lengthy session and intervene.

        Records this round's tool-call signature, then checks for circling. On
        detection it appends a redirect nudge and returns ``"nudge"`` (let the
        loop continue) until ``max_interventions`` is reached, after which it
        appends a consolidation directive and returns ``"escalate"`` (the caller
        should force a tools-off reply and stop). Returns ``None`` otherwise.
        A no-op when steering or loop detection is off.
        """
        from leuk.agent.steering import (
            circle_consolidation_nudge,
            circle_redirect_nudge,
            detect_circling,
            steering_active,
            tool_call_signature,
        )

        cfg = self.settings.steering
        if not cfg.loop_detection or not steering_active(cfg, self.settings.llm.provider):
            return None
        self._round_signatures.append(tool_call_signature(assistant_msg.tool_calls or []))
        if not detect_circling(self._round_signatures, min_rounds=cfg.loop_min_rounds):
            return None
        if self._loop_interventions < cfg.loop_max_interventions:
            self._loop_interventions += 1
            nudge = Message(role=Role.USER, content=circle_redirect_nudge())
            self._messages.append(nudge)
            await self._persist_message(nudge)
            return "nudge"
        consolidate = Message(role=Role.USER, content=circle_consolidation_nudge())
        self._messages.append(consolidate)
        await self._persist_message(consolidate)
        return "escalate"

    # ── Rate-limit retry helpers ──────────────────────────────────

    async def _generate_with_retry(
        self,
        context: list[Message],
        tools: list[Any] | None = ...,  # type: ignore[assignment]
    ) -> Message:
        """Call ``provider.generate()`` with backoff on rate limits and
        compact-and-retry on context overflow (no data is lost — compaction
        archives and summarizes; the full history stays in SQLite)."""
        if tools is ...:
            tools = self.tools.specs() if len(self.tools) > 0 else None
        attempt = 0
        while True:
            try:
                return await self.provider.generate(context, tools=tools)
            except Exception as exc:
                limit = context_overflow_limit(exc)
                if limit is not None and self._overflow_retries < _MAX_OVERFLOW_RETRIES:
                    self._overflow_retries += 1
                    self._tighten_window(limit)
                    logger.warning(
                        "Context overflow (limit %s) — compacting to ~%d tokens and retrying",
                        limit or "unstated", self._window_clamp,
                    )
                    context = await self._prepare_context()
                    continue
                if not _is_rate_limit_error(exc) or attempt >= _MAX_RETRIES:
                    raise
                attempt += 1
                delay = _extract_retry_delay(exc) or min(2**attempt, _MAX_BACKOFF)
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %.0fs",
                    attempt, _MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)

    async def _stream_with_retry(
        self,
        context: list[Message],
        tools: list[Any] | None = ...,  # type: ignore[assignment]
    ) -> AsyncIterator[StreamEvent]:
        """Call ``provider.stream()`` with backoff on rate limits and
        compact-and-retry on context overflow.

        Both error kinds occur at the start of the stream (the server rejects
        the request before any token), so the whole call is retried — guarded
        by ``yielded`` so a mid-stream failure is never silently replayed. A
        ``RATE_LIMITED`` event is yielded so the renderer shows what happened.
        """
        if tools is ...:
            tools = self.tools.specs() if len(self.tools) > 0 else None
        attempt = 0
        while True:
            yielded = False
            try:
                async for event in self.provider.stream(context, tools=tools):
                    yielded = True
                    yield event
                return  # stream completed successfully
            except Exception as exc:
                limit = context_overflow_limit(exc)
                if (
                    limit is not None
                    and not yielded
                    and self._overflow_retries < _MAX_OVERFLOW_RETRIES
                ):
                    self._overflow_retries += 1
                    self._tighten_window(limit)
                    logger.warning(
                        "Context overflow (limit %s) — compacting to ~%d tokens and retrying",
                        limit or "unstated", self._window_clamp,
                    )
                    yield StreamEvent(
                        type=StreamEventType.RATE_LIMITED,
                        content=(
                            "Context overflow — compacting to fit the model's "
                            f"window (~{self._window_clamp:,} tokens) and retrying… "
                            "(nothing is lost: the full history stays available "
                            "via the history tool)"
                        ),
                    )
                    context = await self._prepare_context()
                    continue
                if not _is_rate_limit_error(exc) or yielded or attempt >= _MAX_RETRIES:
                    raise
                attempt += 1
                delay = _extract_retry_delay(exc) or min(2**attempt, _MAX_BACKOFF)
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %.0fs",
                    attempt, _MAX_RETRIES, delay,
                )
                yield StreamEvent(
                    type=StreamEventType.RATE_LIMITED,
                    content=f"Rate limited, retrying in {delay:.0f}s... "
                    f"(attempt {attempt}/{_MAX_RETRIES})",
                )
                await asyncio.sleep(delay)

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Dispatch a tool call to the appropriate handler."""
        # Forward provider metadata (e.g. Google thought_signature) so it
        # can be replayed in the tool result message.
        meta = tool_call.metadata

        # Safety gate
        arguments = tool_call.arguments
        if self.safety_guard is not None:
            verdict = await self.safety_guard.gate(tool_call)
            if verdict.verdict == PermissionAction.DENY:
                # Signal the agent loop to stop after this round.
                self._stop_requested = True
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=f"[BLOCKED] {verdict.reason}",
                    is_error=True,
                    metadata=meta,
                )
            # The user may have edited the arguments at the approval prompt
            # (Tab→amend); run the tool with their version and record it on the
            # call so the conversation reflects what actually ran.
            if verdict.amended_args is not None:
                arguments = verdict.amended_args
                tool_call.arguments = arguments

        tool = self.tools.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=self._maybe_recovery_hint(f"[ERROR] Unknown tool: {tool_call.name}"),
                is_error=True,
                metadata=meta,
            )

        try:
            result = await tool.execute(arguments)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=result,
                metadata=meta,
            )
        except Exception as exc:
            logger.exception("Tool %s failed", tool_call.name)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=self._maybe_recovery_hint(f"[ERROR] Tool execution failed: {exc}"),
                is_error=True,
                metadata=meta,
            )

    async def _persist_message(self, msg: Message) -> None:
        """Save a message to durable storage."""
        await self.sqlite.append_message(self.session.id, msg)

    async def _cache_context(self) -> None:
        """Snapshot current messages to the hot-state cache."""
        # Simple serialisation -- just the last N messages
        data: list[dict[str, Any]] = []
        for m in self._messages[-100:]:  # Keep last 100 in hot cache
            entry: dict[str, Any] = {
                "role": m.role.value,
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
            }
            if m.tool_calls:
                entry["tool_calls"] = [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls
                ]
            if m.tool_result:
                entry["tool_result"] = {
                    "tool_call_id": m.tool_result.tool_call_id,
                    "name": m.tool_result.name,
                    "content": m.tool_result.content,
                    "is_error": m.tool_result.is_error,
                }
            data.append(entry)
        await self.hot_store.set_context(self.session.id, json.dumps(data))

    async def shutdown(self) -> None:
        """Persist final state and release resources."""
        self.session.status = SessionStatus.PAUSED
        await self.sqlite.update_session(self.session)
        await self._cache_context()

        # Clean up browser if one was registered
        from leuk.tools.browser import BrowserTool
        browser_tool = self.tools.get("browser")
        if isinstance(browser_tool, BrowserTool):
            await browser_tool.close()

        # Shut down sandbox container if one is running
        from leuk.tools.shell import ShellTool
        shell_tool = self.tools.get("shell")
        if isinstance(shell_tool, ShellTool):
            await shell_tool.shutdown_sandbox()

        logger.info("Session %s paused", self.session.id)
