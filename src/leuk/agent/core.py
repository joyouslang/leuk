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

        # Prepend system prompt if not already present
        if not self._messages or self._messages[0].role != Role.SYSTEM:
            sys_msg = Message(role=Role.SYSTEM, content=self.session.system_prompt)
            self._messages.insert(0, sys_msg)

        await self.hot_store.set_active_session(self.session.id)

    async def run(self, user_input: str) -> AsyncIterator[Message]:
        """Process a user message through the agent loop.

        Yields each assistant/tool message as it's produced.
        """
        # Add user message
        user_msg = Message(role=Role.USER, content=user_input)
        self._messages.append(user_msg)
        await self._persist_message(user_msg)

        max_rounds = self.settings.agent.max_tool_rounds

        for _round in range(max_rounds):
            # Apply context management before calling LLM
            context = await self._prepare_context()
            assistant_msg = await self._generate_with_retry(context)
            self._messages.append(assistant_msg)
            await self._persist_message(assistant_msg)
            yield assistant_msg

            # If no tool calls, we're done
            if not assistant_msg.tool_calls:
                break

            # Execute all tool calls
            for tc in assistant_msg.tool_calls:
                result = await self._execute_tool(tc)
                tool_msg = Message(role=Role.TOOL, tool_result=result)
                self._messages.append(tool_msg)
                await self._persist_message(tool_msg)
                yield tool_msg

            # Stop the loop if a tool call was denied by the user.
            if self._stop_requested:
                self._stop_requested = False
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
        user_msg = Message(role=Role.USER, content=user_input)
        self._messages.append(user_msg)
        await self._persist_message(user_msg)

        max_rounds = self.settings.agent.max_tool_rounds
        text_parts: list[str] = []  # accumulate text deltas for interrupt persistence

        try:
            for _round in range(max_rounds):
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

                self._messages.append(assistant_msg)
                await self._persist_message(assistant_msg)
                text_parts.clear()  # persisted successfully

                if not assistant_msg.tool_calls:
                    break

                for tc in assistant_msg.tool_calls:
                    result = await self._execute_tool(tc)
                    tool_msg = Message(role=Role.TOOL, tool_result=result)
                    self._messages.append(tool_msg)
                    await self._persist_message(tool_msg)
                    yield tool_msg

                # Stop the loop if a tool call was denied by the user.
                if self._stop_requested:
                    self._stop_requested = False
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
            raise

        await self._cache_context()

    async def _prepare_context(self) -> list[Message]:
        """Apply context window management to the current message history.

        Returns a (possibly truncated/summarized) copy suitable for the LLM.
        The internal _messages list is NOT modified -- only the copy sent to the
        LLM is affected.
        """
        cfg = self.settings.agent

        archive_cfg = self.settings.archive
        archive_kwargs: dict[str, str] = {}
        if archive_cfg.enabled:
            archive_kwargs["session_id"] = self.session.id
            archive_kwargs["archive_dir"] = archive_cfg.directory

        return await compact(
            self._messages,
            self.provider,
            max_tokens=cfg.max_context_tokens,
            max_result_tokens=cfg.max_tool_result_tokens,
            **archive_kwargs,
        )

    # ── Rate-limit retry helpers ──────────────────────────────────

    async def _generate_with_retry(
        self,
        context: list[Message],
        tools: list[Any] | None = ...,  # type: ignore[assignment]
    ) -> Message:
        """Call ``provider.generate()`` with exponential backoff on rate limits."""
        if tools is ...:
            tools = self.tools.specs() if len(self.tools) > 0 else None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return await self.provider.generate(context, tools=tools)
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt == _MAX_RETRIES:
                    raise
                delay = _extract_retry_delay(exc) or min(2 ** (attempt + 1), _MAX_BACKOFF)
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %.0fs",
                    attempt + 1, _MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("Unreachable")  # pragma: no cover

    async def _stream_with_retry(
        self,
        context: list[Message],
        tools: list[Any] | None = ...,  # type: ignore[assignment]
    ) -> AsyncIterator[StreamEvent]:
        """Call ``provider.stream()`` with exponential backoff on rate limits.

        The 429 error occurs at the start of the stream (before any events
        are yielded), so we retry the entire ``provider.stream()`` call.
        A ``RATE_LIMITED`` event is yielded so the renderer can display a
        user-visible message.
        """
        if tools is ...:
            tools = self.tools.specs() if len(self.tools) > 0 else None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async for event in self.provider.stream(context, tools=tools):
                    yield event
                return  # stream completed successfully
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt == _MAX_RETRIES:
                    raise
                delay = _extract_retry_delay(exc) or min(2 ** (attempt + 1), _MAX_BACKOFF)
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %.0fs",
                    attempt + 1, _MAX_RETRIES, delay,
                )
                yield StreamEvent(
                    type=StreamEventType.RATE_LIMITED,
                    content=f"Rate limited, retrying in {delay:.0f}s... "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES})",
                )
                await asyncio.sleep(delay)

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Dispatch a tool call to the appropriate handler."""
        # Forward provider metadata (e.g. Google thought_signature) so it
        # can be replayed in the tool result message.
        meta = tool_call.metadata

        # Safety gate
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

        tool = self.tools.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"[ERROR] Unknown tool: {tool_call.name}",
                is_error=True,
                metadata=meta,
            )

        try:
            result = await tool.execute(tool_call.arguments)
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
                content=f"[ERROR] Tool execution failed: {exc}",
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
