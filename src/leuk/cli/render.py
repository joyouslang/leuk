"""Rich-based rendering for agent streaming output.

Provides compact, informative display of tool-call lifecycle and streamed
text.  Tool calls appear as single status lines with spinners while in-flight,
and collapse to ✓/✗ indicators with truncated results when complete.

The renderer supports two consumption modes:

1. **Iterator mode** (``render_stream``): consumes an ``AsyncIterator`` of
   events — used by the legacy synchronous REPL flow.
2. **Queue mode** (``render_queue``): consumes from an ``asyncio.Queue`` —
   used by the new concurrent REPL where input and rendering run as
   separate tasks.

Tool/sub-agent results always render **compact** here; the full output is
expandable in the interactive history browser (``Tab`` in the REPL).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum

from rich.box import ROUNDED
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

from leuk.cli import theme as _theme
from leuk.types import Message, Role, StreamEvent, StreamEventType, ToolCall, ToolResult


def _code_theme() -> str:
    """Active pygments code style (read dynamically so theme switches apply)."""
    return _theme.CODE_THEME


# ── Constants ──────────────────────────────────────────────────────

_TRUNCATE_LEN = 200  # chars shown in compact mode
_SPINNER_STYLE = "dots"


# ── Tool status tracking ──────────────────────────────────────────


class ToolState(StrEnum):
    """Lifecycle state of a tool call."""

    PENDING = "pending"  # TOOL_CALL_START received, waiting for END
    RUNNING = "running"  # TOOL_CALL_END received, executing
    SUCCESS = "success"  # Result received, no error
    FAILED = "failed"  # Result received, is_error=True


@dataclass(slots=True)
class ToolStatus:
    """Tracks one tool call through its lifecycle."""

    tool_call: ToolCall
    state: ToolState = ToolState.PENDING
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None
    result: ToolResult | None = None

    @property
    def elapsed(self) -> float:
        end = self.end_time or time.monotonic()
        return end - self.start_time

    @property
    def elapsed_str(self) -> str:
        e = self.elapsed
        if e < 1:
            return f"{e * 1000:.0f}ms"
        return f"{e:.1f}s"

    @property
    def args_str(self) -> str:
        """Compact argument representation."""
        args = self.tool_call.arguments
        if not args:
            return ""
        parts: list[str] = []
        for k, v in args.items():
            s = repr(v)
            if len(s) > 60:
                s = s[:57] + "..."
            parts.append(f"{k}={s}")
        return ", ".join(parts)


class ToolStatusTracker:
    """Tracks all tool calls within the current agent run."""

    def __init__(self) -> None:
        self._statuses: list[ToolStatus] = []
        self._by_id: dict[str, ToolStatus] = {}
        self._round: int = 0

    @property
    def round(self) -> int:
        return self._round

    def new_round(self) -> None:
        self._round += 1

    def start(self, tool_call: ToolCall) -> ToolStatus:
        ts = ToolStatus(tool_call=tool_call)
        self._statuses.append(ts)
        self._by_id[tool_call.id] = ts
        return ts

    def mark_running(self, tool_call_id: str) -> None:
        ts = self._by_id.get(tool_call_id)
        if ts:
            ts.state = ToolState.RUNNING

    def complete(self, result: ToolResult) -> None:
        ts = self._by_id.get(result.tool_call_id)
        if ts:
            ts.state = ToolState.FAILED if result.is_error else ToolState.SUCCESS
            ts.end_time = time.monotonic()
            ts.result = result

    @property
    def active(self) -> list[ToolStatus]:
        return [ts for ts in self._statuses if ts.state in (ToolState.PENDING, ToolState.RUNNING)]

    @property
    def all_statuses(self) -> list[ToolStatus]:
        return list(self._statuses)

    def clear(self) -> None:
        self._statuses.clear()
        self._by_id.clear()
        self._round = 0


# ── Rendering helpers ─────────────────────────────────────────────


def _truncate(text: str, max_len: int = _TRUNCATE_LEN) -> str:
    """Truncate text with an ellipsis marker noting how much was elided."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"… [+{len(text) - max_len} chars]"


def _tool_status_line(ts: ToolStatus, full: bool = False) -> Text:
    """Render a single tool status as a Rich Text line."""
    line = Text()

    match ts.state:
        case ToolState.PENDING:
            line.append("⟳ ", style="yellow")
            line.append(ts.tool_call.name, style="bold yellow")
            line.append(f"({ts.args_str})", style="dim")
        case ToolState.RUNNING:
            line.append("⟳ ", style="yellow")
            line.append(ts.tool_call.name, style="bold yellow")
            line.append(f"({ts.args_str})", style="dim")
            line.append(f"  {ts.elapsed_str}", style="dim")
        case ToolState.SUCCESS:
            line.append("✓ ", style="green")
            line.append(ts.tool_call.name, style="bold green")
            line.append(f"  {ts.elapsed_str}", style="dim")
            if ts.result:
                preview = ts.result.content if full else _truncate(ts.result.content)
                if preview.strip():
                    line.append("\n  ", style="")
                    line.append(preview, style="dim")
        case ToolState.FAILED:
            line.append("✗ ", style="red")
            line.append(ts.tool_call.name, style="bold red")
            line.append(f"  {ts.elapsed_str}", style="dim")
            if ts.result:
                preview = ts.result.content if full else _truncate(ts.result.content)
                if preview.strip():
                    line.append("\n  ", style="")
                    line.append(preview, style="red dim")

    return line


def render_tool_statuses(tracker: ToolStatusTracker, full: bool = False) -> Text:
    """Build a Rich renderable showing all tool statuses (compact, one line
    each). Used for the in-flight Live spinner phase."""
    output = Text()
    for i, ts in enumerate(tracker.all_statuses):
        if i > 0:
            output.append("\n")
        output.append_text(_tool_status_line(ts, full=full))
    return output


# ── Finalised tool blocks (gemini-cli style) ───────────────────────

_SUMMARY_KEYS = ("command", "path", "file_path", "url", "query", "pattern", "name")
_STATE_GLYPH = {
    ToolState.PENDING: ("○", "tool.pending"),
    ToolState.RUNNING: ("⟳", "tool.running"),
    ToolState.SUCCESS: ("✓", "tool.success"),
    ToolState.FAILED: ("✗", "tool.failed"),
}


def _tool_summary(tool_call: ToolCall) -> str:
    """A short, human description of a tool call for the header row."""
    args = tool_call.arguments or {}
    for key in _SUMMARY_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().splitlines()[0]
    # Fallback: compact arg list.
    parts: list[str] = []
    for k, v in args.items():
        s = repr(v)
        if len(s) > 40:
            s = s[:37] + "…"
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def _looks_like_diff(content: str) -> bool:
    head = content.lstrip()[:80]
    return head.startswith(("@@", "diff --git", "--- ", "+++ "))


def _result_body(tool_name: str, content: str, full: bool) -> RenderableType:
    """Render a tool result's body for the bordered block."""
    text = content if full else _truncate(content)
    if _looks_like_diff(content):
        return Syntax(
            content if full else text,
            "diff",
            theme=_code_theme(),
            word_wrap=True,
            background_color="default",
        )
    return Text(text, style="primary")


def render_tool_block(ts: ToolStatus, full: bool = False) -> RenderableType:
    """Render a completed tool call as a header row + bordered result body.

    Mirrors gemini-cli's signature look: ``[glyph] toolname  summary  time``
    with the result hanging beneath in a rounded box. Compact by default
    (*full* shows the entire result — used by the history browser).
    """
    glyph, glyph_style = _STATE_GLYPH.get(ts.state, ("•", "tool.desc"))

    header = Text()
    header.append(f"{glyph} ", style=glyph_style)
    name_style = "tool.failed" if ts.state is ToolState.FAILED else "tool.name"
    header.append(ts.tool_call.name, style=name_style)
    summary = _tool_summary(ts.tool_call)
    if summary:
        header.append("  ")
        header.append(summary, style="tool.desc")
    if ts.end_time is not None:
        header.append(f"   {ts.elapsed_str}", style="comment")

    content = ts.result.content if ts.result else ""
    if not content.strip():
        return header

    border = "tool.failed" if ts.state is ToolState.FAILED else "tool.border"
    body = Panel(
        _result_body(ts.tool_call.name, content, full),
        box=ROUNDED,
        border_style=border,
        padding=(0, 1),
        expand=False,
    )
    return Group(header, Padding(body, (0, 0, 0, 2)))


# ── Conversation history replay ────────────────────────────────────


def render_history(
    console: Console,
    messages: list[Message],
    *,
    full: bool = False,
    title: str = "session history",
) -> int:
    """Replay a stored conversation to *console* so it's visible/scrollable.

    User turns get a prompt-style marker, assistant turns render as Markdown,
    and tool results render as the same bordered blocks used live. System and
    internal ``[SYSTEM]`` housekeeping messages are skipped. Returns the number
    of messages rendered (0 if there was nothing to show).
    """
    # Pair tool_calls to their results so blocks show the original call args.
    calls_by_id: dict[str, ToolCall] = {}
    for m in messages:
        if m.tool_calls:
            for tc in m.tool_calls:
                calls_by_id[tc.id] = tc

    renderables: list[RenderableType] = []
    for m in messages:
        if m.role is Role.SYSTEM:
            continue
        if m.role is Role.USER:
            content = (m.content or "").strip()
            if not content or content.startswith("[SYSTEM]"):
                continue
            line = Text()
            line.append("❯ ", style="user.label")
            line.append(content, style="primary")
            renderables.append(line)
        elif m.role is Role.ASSISTANT:
            if m.content and m.content.strip():
                renderables.append(Markdown(m.content, code_theme=_code_theme()))
        elif m.role is Role.TOOL and m.tool_result:
            tc = calls_by_id.get(m.tool_result.tool_call_id) or ToolCall(
                id=m.tool_result.tool_call_id,
                name=m.tool_result.name,
                arguments={},
            )
            ts = ToolStatus(
                tool_call=tc,
                state=ToolState.FAILED if m.tool_result.is_error else ToolState.SUCCESS,
                result=m.tool_result,
            )
            ts.end_time = None  # no timing for replayed history
            renderables.append(render_tool_block(ts, full=full))

    if not renderables:
        return 0

    # Ensure named styles resolve even if the caller's console is unthemed.
    from leuk.cli.theme import LEUK_THEME

    console.push_theme(LEUK_THEME)
    try:
        console.rule(f"[comment]{title}[/comment]", style="rule.line")
        for r in renderables:
            console.print(r)
        console.rule(style="rule.line")
    finally:
        console.pop_theme()
    return len(renderables)


# ── StreamRenderer ────────────────────────────────────────────────


class StreamRenderer:
    """Renders agent streaming output to the terminal.

    Manages the transition between text streaming and tool-call display.
    Uses ``rich.Live`` only during tool-call phases for animated spinners,
    and raw ``print()`` for text token streaming (lowest latency).

    Tool/sub-agent results always render **compact** in the live scrollback; the
    full output is browsable/expandable in the history view (Tab in the REPL).

    Parameters
    ----------
    console:
        The Rich Console to use.
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        # Ensure our named styles (tool.*, accent.*, diff.*, …) always
        # resolve, even if the caller's console wasn't built with the theme.
        # push_theme stacks, so this is harmless on an already-themed console.
        from leuk.cli.theme import LEUK_THEME

        console.push_theme(LEUK_THEME)
        self.tracker = ToolStatusTracker()
        self._text_buffer: list[str] = []
        self._in_text_stream = False
        self._tts_speaker: object | None = None  # StreamingTTSSpeaker, if active
        self._live: Live | None = None
        self._text_live: Live | None = None  # markdown-streaming region
        self._thinking_live: Live | None = None
        self._current_round = 0
        # Render assistant text as Markdown (headings, lists, code, …).
        # Disable to fall back to plain streamed text (e.g. for tests).
        self.markdown = True

    def set_tts_speaker(self, speaker: object | None) -> None:
        """Attach or detach a :class:`~leuk.voice.tts.StreamingTTSSpeaker`."""
        self._tts_speaker = speaker

    def _start_thinking(self) -> None:
        """Show a braille 'Thinking…' spinner with a stop hint."""
        if self._thinking_live is not None:
            return
        label = Text()
        label.append("Thinking… ", style="italic dim")
        label.append("(Ctrl-C to stop)", style="comment")
        self._thinking_live = Live(
            Spinner("dots", text=label, style="accent.purple"),
            console=self.console,
            transient=True,
        )
        self._thinking_live.start()

    def _stop_thinking(self) -> None:
        """Stop the thinking spinner if active."""
        if self._thinking_live is not None:
            self._thinking_live.stop()
            self._thinking_live = None

    async def render_stream(self, stream: AsyncIterator[StreamEvent | Message]) -> None:
        """Consume an agent stream and render all output.

        This method handles the full lifecycle:
        text deltas → tool calls → tool results → next round.
        """
        self.tracker.clear()
        self._text_buffer.clear()
        self._in_text_stream = False
        self._current_round = 0

        async for event in stream:
            if isinstance(event, StreamEvent):
                self._handle_stream_event(event)
            elif isinstance(event, Message):
                self._handle_message(event)

        # Final cleanup
        self._stop_thinking()
        self._flush_text()
        self._stop_text_live()
        self._stop_live()

    def _handle_stream_event(self, event: StreamEvent) -> None:
        match event.type:
            case StreamEventType.TEXT_DELTA:
                self._stop_thinking()
                self._on_text_delta(event)
            case StreamEventType.TOOL_CALL_START:
                self._stop_thinking()
                self._on_tool_call_start(event)
            case StreamEventType.TOOL_CALL_DELTA:
                pass  # arguments accumulate in the provider
            case StreamEventType.TOOL_CALL_END:
                self._on_tool_call_end(event)
            case StreamEventType.MESSAGE_COMPLETE:
                self._on_message_complete(event)
            case StreamEventType.RATE_LIMITED:
                self._stop_thinking()
                self.console.print(f"[status.warn]⏳ {event.content}[/status.warn]")

    def _on_text_delta(self, event: StreamEvent) -> None:
        """Handle incremental text tokens.

        When :attr:`markdown` is on, tokens accumulate into a ``rich.Live``
        region that re-renders the buffer as Markdown so headings, lists,
        and code blocks format live. Otherwise tokens print raw (lowest
        latency; used by tests and non-TTY output).
        """
        if not self._in_text_stream:
            self._stop_live()
            self._in_text_stream = True
            # Start a fresh buffer for this text segment so its Markdown
            # render isn't polluted by an earlier segment in the same turn.
            self._text_buffer.clear()
            if self.markdown:
                self._start_text_live()
        self._text_buffer.append(event.content)

        if self.markdown and self._text_live is not None:
            self._text_live.update(self._render_assistant_md())
        else:
            print(event.content, end="", flush=True)

        # Feed token to streaming TTS for sentence-by-sentence speech.
        if self._tts_speaker is not None and hasattr(self._tts_speaker, "feed"):
            self._tts_speaker.feed(event.content)

    def _render_assistant_md(self) -> RenderableType:
        """Render the accumulated assistant text as Markdown."""
        return Markdown("".join(self._text_buffer), code_theme=_code_theme())

    def _start_text_live(self) -> None:
        """Begin a Live region that streams assistant Markdown.

        The region is **transient** and capped to the visible screen
        (``vertical_overflow="ellipsis"``). ``rich.Live`` can only redraw in
        place when its content fits the terminal; with ``"visible"`` a response
        taller than the screen is re-emitted in full on every refresh, flooding
        the scrollback with dozens of duplicated copies. Capping + transient
        keeps the live preview to one screen, and :meth:`_stop_text_live` erases
        it and prints the complete Markdown exactly once.
        """
        if self._text_live is None:
            self._text_live = Live(
                Markdown(""),
                console=self.console,
                refresh_per_second=6,
                vertical_overflow="ellipsis",
                transient=True,
            )
            self._text_live.start()

    def _stop_text_live(self) -> None:
        """Erase the streaming preview and emit the complete Markdown once."""
        if self._text_live is not None:
            # Stop the transient live first so its (screen-capped) preview is
            # cleared, then print the full rendered Markdown a single time — no
            # per-token duplication, and never truncated for tall responses.
            self._text_live.stop()
            self._text_live = None
            if "".join(self._text_buffer).strip():
                self.console.print(self._render_assistant_md())

    def _on_tool_call_start(self, event: StreamEvent) -> None:
        """A tool call is beginning — show spinner."""
        self._flush_text()
        if event.tool_call:
            # New round?
            if not self.tracker.active:
                self._current_round += 1
                self.tracker.new_round()
                if self._current_round > 1:
                    self.console.print(f"\n[dim]── round {self._current_round} ──[/dim]")
            self.tracker.start(event.tool_call)
            self._update_live()

    def _on_tool_call_end(self, event: StreamEvent) -> None:
        """Tool call arguments fully received — mark running."""
        if event.tool_call:
            self.tracker.mark_running(event.tool_call.id)
            # Update the status in the existing tracker entry with full args
            ts = self.tracker._by_id.get(event.tool_call.id)
            if ts:
                ts.tool_call = event.tool_call
            self._update_live()

    def _on_message_complete(self, event: StreamEvent) -> None:
        """Full assistant message assembled."""
        self._flush_text()

    def _handle_message(self, msg: Message) -> None:
        """Handle a raw Message (tool results from agent loop)."""
        if msg.role == Role.TOOL and msg.tool_result:
            self.tracker.complete(msg.tool_result)
            self._update_live()

            # If no more active tool calls, finalize the display
            if not self.tracker.active:
                self._finalize_round()

    def _flush_text(self) -> None:
        """End text streaming mode (the buffer is kept for the next segment
        to reset on demand, and for callers/tests to inspect)."""
        if self._in_text_stream:
            if self.markdown and self._text_live is not None:
                self._stop_text_live()
            else:
                print()  # newline after raw streamed text
            self._in_text_stream = False

    def _start_live(self) -> None:
        """Start a Rich Live context for animated tool display."""
        if self._live is None:
            self._live = Live(
                Text(""),
                console=self.console,
                refresh_per_second=8,
                transient=True,  # Clear spinner lines when done
            )
            self._live.start()

    def _stop_live(self) -> None:
        """Stop the Rich Live context."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def pause(self) -> None:
        """Pause animated rendering so an interactive prompt can take the TTY.

        ``rich.Live`` and ``prompt_toolkit`` both want exclusive control of
        stdin/stdout; if Live is running while a confirmation prompt asks
        for input, key echoes can be eaten and Enter often resolves to the
        empty string (→ default deny). Call this *before* showing an
        interactive prompt and :meth:`resume` after it returns.
        """
        self._stop_thinking()
        self._flush_text()
        self._stop_text_live()
        self._stop_live()

    def resume(self) -> None:
        """Counterpart to :meth:`pause` — currently a no-op marker.

        Live is restarted automatically on the next tool event; this method
        exists so callers can express intent symmetrically.
        """
        return None

    def _update_live(self) -> None:
        """Refresh the Live display with current tool statuses."""
        self._start_live()
        renderable = render_tool_statuses(self.tracker)
        if self._live is not None:
            self._live.update(renderable)

    def _finalize_round(self) -> None:
        """Print finalised tool blocks (non-transient) after all calls done."""
        self._stop_live()
        for ts in self.tracker.all_statuses:
            self.console.print(render_tool_block(ts))
        # Clear tracked statuses for next round
        self.tracker._statuses.clear()
        self.tracker._by_id.clear()

    # ── Queue-based rendering (new concurrent architecture) ───────

    async def render_queue(
        self,
        queue: asyncio.Queue[StreamEvent | Message | object],
        *,
        stop_sentinel: object | None = None,
    ) -> None:
        """Consume events from *queue* and render them.

        Runs until a ``TURN_COMPLETE`` event is received (one full agent
        turn), or until *stop_sentinel* appears in the queue.

        This is the queue-mode counterpart of :meth:`render_stream` and is
        designed to be run as a concurrent asyncio task alongside the input
        task.
        """
        from leuk.agent.session import _STOP_SENTINEL
        from leuk.types import StreamEventType as SET

        sentinel = stop_sentinel or _STOP_SENTINEL

        self.tracker.clear()
        self._text_buffer.clear()
        self._in_text_stream = False
        self._current_round = 0
        self._start_thinking()

        try:
            while True:
                event = await queue.get()

                if event is sentinel:
                    break

                if isinstance(event, StreamEvent):
                    if event.type == SET.TURN_COMPLETE:
                        break
                    if event.type == SET.STATE_CHANGE:
                        # Show thinking spinner when agent is back to THINKING
                        # (e.g. between tool rounds).
                        if event.content == "thinking":
                            self._start_thinking()
                        continue
                    if event.type == SET.ERROR:
                        self._stop_thinking()
                        self._flush_text()
                        self._stop_text_live()
                        self._stop_live()
                        self.console.print(
                            f"[status.error]Agent error:[/status.error] {event.content}"
                        )
                        self.console.print(
                            "[comment]Use [footer.key]/retry[/footer.key] to re-send the "
                            "last message, or type a new one.[/comment]"
                        )
                        continue
                    self._handle_stream_event(event)
                elif isinstance(event, Message):
                    self._handle_message(event)
        finally:
            # Always tear down live regions — even if the turn is cancelled
            # (Ctrl-C), so the "Thinking…" spinner never leaks past the prompt.
            self._stop_thinking()
            self._flush_text()
            self._stop_text_live()
            self._stop_live()
