"""Persistent-input TUI — the invalidate-driven renderer sink.

This is the rendering half of the full-screen REPL described in
``docs/repl-tui-design.md``. Where :class:`leuk.cli.render.StreamRenderer`
drives three ``rich.Live`` regions and ``console.print`` (which own the
terminal and so cannot coexist with a ``prompt_toolkit`` full-screen
``Application``), :class:`TuiRenderer` instead:

* appends **finalized** content to ``blocks`` — the shared scrollback model
  (:mod:`leuk.cli.blocks`), the same blocks the history browser uses; and
* exposes the single **in-flight** renderable (thinking spinner / streaming
  markdown / tool spinner) as an ANSI string in ``live_ansi``.

Every mutation calls the injected ``invalidate`` callback so the Application
repaints; a periodic :meth:`tick` advances the spinner frame. There is **no**
``rich.Live`` and no console ownership here — the Application owns the terminal
and renders ``blocks`` + ``live_ansi`` itself. ``rich`` remains the *content*
renderer (Markdown/panels/diffs → ANSI via :func:`leuk.cli.blocks.rich_to_ansi`).

The event-handling logic mirrors ``StreamRenderer`` exactly; only the output
sink differs, which keeps the two in lock-step and makes this unit-testable
without a TTY (pass a no-op ``invalidate`` and assert on ``blocks``/``live_ansi``).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from functools import partial

from rich.markdown import Markdown
from rich.text import Text

from leuk.cli.blocks import Block, render_static, render_tool, rich_to_ansi
from leuk.cli.render import (
    ToolStatusTracker,
    _code_theme,
    render_tool_block,
    render_tool_statuses,
)
from leuk.types import Message, Role, StreamEvent, StreamEventType

# Braille spinner frames (matches rich's "dots" feel, but we own the timing).
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class TuiRenderer:
    """Renderer sink for the persistent-input TUI (no ``rich.Live``).

    Parameters
    ----------
    invalidate:
        Called after any state change to request a repaint (wired to
        ``Application.invalidate``). Defaults to a no-op for tests.
    width_fn:
        Returns the current content width in columns (wired to the scrollback
        window width). Defaults to 80.
    markdown:
        Render assistant text as Markdown. Disable for plain-text tests.
    """

    def __init__(
        self,
        *,
        invalidate: Callable[[], None] | None = None,
        width_fn: Callable[[], int] | None = None,
        markdown: bool = True,
    ) -> None:
        self._invalidate = invalidate or (lambda: None)
        self._width_fn = width_fn or (lambda: 80)
        self.markdown = markdown

        self.blocks: list[Block] = []
        self.live_ansi: str | None = None  # in-flight region, or None when idle

        self.tracker = ToolStatusTracker()
        self._text_buffer: list[str] = []
        self._in_text_stream = False
        self._current_round = 0
        self._tts_speaker: object | None = None

        # Live-region mode: "", "thinking", "text", or "tools".
        self._mode = ""
        self._thinking_start = 0.0
        self._frame = 0

    # ── public knobs ──────────────────────────────────────────────────────

    def set_tts_speaker(self, speaker: object | None) -> None:
        """Attach/detach a streaming TTS speaker (sentence-by-sentence speech)."""
        self._tts_speaker = speaker

    def reset(self) -> None:
        """Clear per-turn streaming state (keeps the finalized ``blocks``)."""
        self.tracker.clear()
        self._text_buffer.clear()
        self._in_text_stream = False
        self._current_round = 0
        self._mode = ""
        self.live_ansi = None

    def width(self) -> int:
        return max(20, self._width_fn())

    # ── live-region composition ───────────────────────────────────────────

    def tick(self) -> None:
        """Advance the spinner frame; repaint if a spinner is showing."""
        if self._mode in ("thinking", "tools"):
            self._frame = (self._frame + 1) % len(_SPINNER_FRAMES)
            self._compose_live()
            self._invalidate()

    def _spinner(self) -> str:
        return _SPINNER_FRAMES[self._frame % len(_SPINNER_FRAMES)]

    def _compose_live(self) -> None:
        """Recompute ``live_ansi`` from the current mode + state."""
        if self._mode == "thinking":
            elapsed = max(0, int(time.monotonic() - self._thinking_start))
            label = Text()
            label.append(f"{self._spinner()} ", style="accent.purple")
            label.append("Thinking… ", style="italic dim")
            label.append(f"{elapsed}s ", style="comment")
            label.append("(Ctrl-C to stop)", style="comment")
            self.live_ansi = rich_to_ansi(label, self.width())
        elif self._mode == "text":
            self.live_ansi = rich_to_ansi(self._assistant_md(), self.width())
        elif self._mode == "tools":
            self.live_ansi = rich_to_ansi(render_tool_statuses(self.tracker), self.width())
        else:
            self.live_ansi = None

    def _assistant_md(self) -> Markdown:
        return Markdown("".join(self._text_buffer), code_theme=_code_theme())

    # ── thinking ──────────────────────────────────────────────────────────

    def start_thinking(self) -> None:
        if self._mode == "thinking":
            return
        self._mode = "thinking"
        self._thinking_start = time.monotonic()
        self._compose_live()
        self._invalidate()

    def _stop_thinking(self) -> None:
        if self._mode == "thinking":
            self._mode = ""
            self.live_ansi = None

    # ── finalized-block helpers ───────────────────────────────────────────

    def append_block(self, block: Block) -> None:
        """Append a finalized scrollback block and repaint."""
        self.blocks.append(block)
        self._invalidate()

    def append_static(self, renderable: object, *, expandable: bool = False) -> None:
        self.append_block(Block(expandable, partial(render_static, renderable)))

    # ── event handling (mirrors StreamRenderer) ───────────────────────────

    def handle_event(self, event: StreamEvent) -> None:
        match event.type:
            case StreamEventType.TEXT_DELTA:
                self._stop_thinking()
                self._on_text_delta(event)
            case StreamEventType.TOOL_CALL_START:
                self._stop_thinking()
                self._on_tool_call_start(event)
            case StreamEventType.TOOL_CALL_DELTA:
                pass  # args accumulate in the provider
            case StreamEventType.TOOL_CALL_END:
                self._on_tool_call_end(event)
            case StreamEventType.MESSAGE_COMPLETE:
                self._flush_text()
            case StreamEventType.RATE_LIMITED:
                self._stop_thinking()
                self.append_static(Text(f"⏳ {event.content}", style="status.warn"))

    def handle_message(self, msg: Message) -> None:
        """Handle a raw tool-result Message from the agent loop."""
        if msg.role is Role.TOOL and msg.tool_result:
            self.tracker.complete(msg.tool_result)
            if self.tracker.active:
                self._mode = "tools"
                self._compose_live()
                self._invalidate()
            else:
                self._finalize_round()

    def _on_text_delta(self, event: StreamEvent) -> None:
        if not self._in_text_stream:
            self._in_text_stream = True
            self._text_buffer.clear()
            self._mode = "text"
        self._text_buffer.append(event.content)
        if self.markdown:
            self._compose_live()
        self._invalidate()
        if self._tts_speaker is not None and hasattr(self._tts_speaker, "feed"):
            self._tts_speaker.feed(event.content)

    def _flush_text(self) -> None:
        """Freeze the streamed assistant text into a finalized block."""
        if not self._in_text_stream:
            return
        text = "".join(self._text_buffer)
        self._in_text_stream = False
        self._mode = ""
        self.live_ansi = None
        if text.strip():
            self.append_static(self._assistant_md())

    def _on_tool_call_start(self, event: StreamEvent) -> None:
        self._flush_text()
        if not event.tool_call:
            return
        if not self.tracker.active:
            self._current_round += 1
            self.tracker.new_round()
            if self._current_round > 1:
                self.append_static(Text(f"── round {self._current_round} ──", style="dim"))
        self.tracker.start(event.tool_call)
        self._mode = "tools"
        self._compose_live()
        self._invalidate()

    def _on_tool_call_end(self, event: StreamEvent) -> None:
        if not event.tool_call:
            return
        self.tracker.mark_running(event.tool_call.id)
        ts = self.tracker._by_id.get(event.tool_call.id)
        if ts:
            ts.tool_call = event.tool_call
        self._mode = "tools"
        self._compose_live()
        self._invalidate()

    def _finalize_round(self) -> None:
        """Freeze all tool calls in the round into finalized (expandable) blocks."""
        self._mode = ""
        self.live_ansi = None
        for ts in self.tracker.all_statuses:
            # Snapshot the status at completion so later rounds don't mutate it.
            frozen = ts
            self.blocks.append(Block(True, partial(render_tool, frozen)))
        self.tracker._statuses.clear()
        self.tracker._by_id.clear()
        self._invalidate()

    # ── queue consumption (counterpart of StreamRenderer.render_queue) ─────

    async def consume(
        self,
        queue: asyncio.Queue[StreamEvent | Message | object],
        *,
        stop_sentinel: object | None = None,
    ) -> None:
        """Consume one agent turn's events from *queue* into blocks/live region.

        Runs until ``TURN_COMPLETE`` or *stop_sentinel*. Mirrors
        ``StreamRenderer.render_queue`` but never touches the terminal directly.
        """
        from leuk.agent.session import _STOP_SENTINEL
        from leuk.types import StreamEventType as SET

        sentinel = stop_sentinel or _STOP_SENTINEL
        self.reset()
        self.start_thinking()

        try:
            while True:
                event = await queue.get()
                if event is sentinel:
                    break
                if isinstance(event, StreamEvent):
                    if event.type == SET.TURN_COMPLETE:
                        break
                    if event.type == SET.STATE_CHANGE:
                        if event.content == "thinking":
                            self.start_thinking()
                        continue
                    if event.type == SET.ERROR:
                        self._flush_text()
                        self._mode = ""
                        self.live_ansi = None
                        self.append_static(
                            Text(f"Agent error: {event.content}", style="status.error")
                        )
                        self.append_static(
                            Text(
                                "Use /retry to re-send the last message, or type a new one.",
                                style="comment",
                            )
                        )
                        continue
                    self.handle_event(event)
                elif isinstance(event, Message):
                    self.handle_message(event)
        finally:
            # Always tear down the live region so a spinner never leaks.
            self._flush_text()
            self._mode = ""
            self.live_ansi = None
            self._invalidate()


__all__ = ["TuiRenderer", "render_tool_block"]
