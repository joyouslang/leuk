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

The ``/verbose`` toggle controls whether tool results are shown in full or
truncated to a compact summary.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from leuk.types import Message, Role, StreamEvent, StreamEventType, ToolCall, ToolResult


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
    """Truncate text with an ellipsis marker."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"… [{len(text)} chars]"


def _tool_status_line(ts: ToolStatus, verbose: bool = False) -> Text:
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
                preview = ts.result.content if verbose else _truncate(ts.result.content)
                if preview.strip():
                    line.append("\n  ", style="")
                    line.append(preview, style="dim")
        case ToolState.FAILED:
            line.append("✗ ", style="red")
            line.append(ts.tool_call.name, style="bold red")
            line.append(f"  {ts.elapsed_str}", style="dim")
            if ts.result:
                preview = ts.result.content if verbose else _truncate(ts.result.content)
                if preview.strip():
                    line.append("\n  ", style="")
                    line.append(preview, style="red dim")

    return line


def render_tool_statuses(tracker: ToolStatusTracker, verbose: bool = False) -> Text:
    """Build a Rich renderable showing all tool statuses."""
    output = Text()
    for i, ts in enumerate(tracker.all_statuses):
        if i > 0:
            output.append("\n")
        output.append_text(_tool_status_line(ts, verbose=verbose))
    return output


# ── StreamRenderer ────────────────────────────────────────────────


class StreamRenderer:
    """Renders agent streaming output to the terminal.

    Manages the transition between text streaming and tool-call display.
    Uses ``rich.Live`` only during tool-call phases for animated spinners,
    and raw ``print()`` for text token streaming (lowest latency).

    Parameters
    ----------
    console:
        The Rich Console to use.
    verbose:
        If True, show full tool results instead of truncated previews.
    """

    def __init__(self, console: Console, *, verbose: bool = False) -> None:
        self.console = console
        self.verbose = verbose
        self.tracker = ToolStatusTracker()
        self._text_buffer: list[str] = []
        self._in_text_stream = False
        self._tts_speaker: object | None = None  # StreamingTTSSpeaker, if active
        self._live: Live | None = None
        self._thinking_live: Live | None = None
        self._current_round = 0

    def set_tts_speaker(self, speaker: object | None) -> None:
        """Attach or detach a :class:`~leuk.voice.tts.StreamingTTSSpeaker`."""
        self._tts_speaker = speaker

    def _start_thinking(self) -> None:
        """Show a 'Thinking...' spinner."""
        if self._thinking_live is not None:
            return
        self._thinking_live = Live(
            Spinner("dots", text="Thinking…", style="dim"),
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
                self.console.print(f"[yellow dim]⏳ {event.content}[/yellow dim]")

    def _on_text_delta(self, event: StreamEvent) -> None:
        """Handle incremental text tokens — printed immediately."""
        if not self._in_text_stream:
            self._stop_live()
            self._in_text_stream = True
        self._text_buffer.append(event.content)
        print(event.content, end="", flush=True)
        # Feed token to streaming TTS for sentence-by-sentence speech.
        if self._tts_speaker is not None and hasattr(self._tts_speaker, "feed"):
            self._tts_speaker.feed(event.content)

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
        """End text streaming mode."""
        if self._in_text_stream:
            print()  # newline after streamed text
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

    def _update_live(self) -> None:
        """Refresh the Live display with current tool statuses."""
        self._start_live()
        renderable = render_tool_statuses(self.tracker, verbose=self.verbose)
        if self._live is not None:
            self._live.update(renderable)

    def _finalize_round(self) -> None:
        """Print final tool statuses (non-transient) after all calls complete."""
        self._stop_live()
        # Print the final static status lines
        output = render_tool_statuses(self.tracker, verbose=self.verbose)
        self.console.print(output)
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
                    self._flush_text()
                    self._stop_live()
                    self.console.print(f"[red]Agent error: {event.content}[/red]")
                    continue
                self._handle_stream_event(event)
            elif isinstance(event, Message):
                self._handle_message(event)

        # Final cleanup
        self._stop_thinking()
        self._flush_text()
        self._stop_live()
