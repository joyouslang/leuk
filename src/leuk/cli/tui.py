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
import shutil
import time
from collections.abc import Callable
from functools import partial
from typing import Any

from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, to_formatted_text
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


# ── Scrollback flattening (pure, testable) ────────────────────────────────


def flatten_blocks(
    blocks: list[Block],
    *,
    live_ansi: str | None,
    selected: int,
    expanded: set[int],
    width: int,
) -> tuple[StyleAndTextTuples, list[int], int]:
    """Render *blocks* (+ optional ``live_ansi`` slice) into formatted text.

    Returns ``(fragments, block_line_offsets, total_lines)``. Each block gets a
    one-row gutter (``▌`` when selected). The live region, when present, is
    appended below the finalized blocks as a non-selectable trailing slice.

    Mirrors the proven flattening in ``history_browser`` so the scrollback pane
    behaves identically; kept pure (no prompt_toolkit Window) so the line math
    is unit-testable without a TTY.
    """
    out: StyleAndTextTuples = []
    block_lines: list[int] = []
    line_no = 0

    def _emit(ansi: str, gutter_style: str, gutter: str) -> None:
        nonlocal line_no
        fragments = to_formatted_text(ANSI(ansi))
        out.append((gutter_style, gutter))
        for style, text, *_ in fragments:
            parts = text.split("\n")
            for j, part in enumerate(parts):
                if j > 0:
                    out.append(("", "\n"))
                    out.append((gutter_style, gutter))
                    line_no += 1
                if part:
                    out.append((style, part))
        out.append(("", "\n"))
        line_no += 1

    for i, blk in enumerate(blocks):
        block_lines.append(line_no)
        selected_here = i == selected
        gutter_style = "class:sel" if selected_here else ""
        gutter = "▌ " if selected_here else "  "
        _emit(blk.render(i in expanded, width), gutter_style, gutter)

    if live_ansi:
        _emit(live_ansi, "", "  ")

    return out, block_lines, line_no


# ── Full-screen REPL Application ───────────────────────────────────────────


class ReplTUI:
    """Full-screen REPL: scrollback (blocks + live region) + persistent input.

    The input box is **always typable**, even while the agent streams — the
    renderer repaints the scrollback via ``invalidate()`` while the input keeps
    focus. ``Tab`` toggles focus between the scrollback (navigable/expandable,
    like the history browser) and the input.

    Parameters
    ----------
    renderer:
        The :class:`TuiRenderer` whose ``blocks``/``live_ansi`` this paints.
    on_submit:
        Async callback invoked with the submitted input text.
    footer_fn:
        Optional ``() -> str`` returning the footer status line.
    prompt:
        The input prompt label.
    """

    def __init__(
        self,
        renderer: TuiRenderer,
        *,
        on_submit: Callable[[str], object],
        footer_fn: Callable[[], str] | None = None,
        prompt: str = "leuk› ",
    ) -> None:
        self.renderer = renderer
        self._on_submit = on_submit
        self._footer_fn = footer_fn or (lambda: "")
        self._prompt = prompt

        self.selected = 0
        self.expanded: set[int] = set()
        self._cursor_line = 0
        self._total_lines = 0
        self._follow = True  # auto-scroll to bottom while new content streams
        self._block_lines: list[int] = []
        self._body_window: Any = None
        self.app: Any = None

    # ── width ─────────────────────────────────────────────────────────────

    def _width(self) -> int:
        return max(20, shutil.get_terminal_size((100, 30)).columns - 2)

    # ── scrollback control ────────────────────────────────────────────────

    def _get_text(self) -> StyleAndTextTuples:
        out, block_lines, total = flatten_blocks(
            self.renderer.blocks,
            live_ansi=self.renderer.live_ansi,
            selected=self.selected,
            expanded=self.expanded,
            width=self._width(),
        )
        self._block_lines = block_lines
        self._total_lines = total
        if self._follow:
            self._cursor_line = max(0, total - 1)
        return out

    def _cursor_position(self):  # noqa: ANN202 — prompt_toolkit Point
        from prompt_toolkit.data_structures import Point

        return Point(x=0, y=max(0, min(self._cursor_line, max(0, self._total_lines - 1))))

    def invalidate(self) -> None:
        """Request a repaint (wired into the renderer)."""
        if self.app is not None:
            self.app.invalidate()

    # ── navigation ────────────────────────────────────────────────────────

    def _scroll(self, delta: int) -> None:
        self._follow = False
        self._cursor_line = max(0, min(max(0, self._total_lines - 1), self._cursor_line + delta))

    def _move(self, delta: int) -> None:
        blocks = self.renderer.blocks
        if not blocks:
            return
        self._follow = False
        self.selected = max(0, min(len(blocks) - 1, self.selected + delta))
        if hasattr(self, "_block_lines") and self.selected < len(self._block_lines):
            self._cursor_line = self._block_lines[self.selected]

    def _toggle(self) -> None:
        blocks = self.renderer.blocks
        if not blocks or self.selected >= len(blocks):
            return
        block = blocks[self.selected]
        if block.on_activate is not None:
            try:
                block.on_activate()
            except Exception:  # noqa: BLE001 — opening is best-effort
                pass
            return
        if not block.expandable:
            return
        if self.selected in self.expanded:
            self.expanded.discard(self.selected)
        else:
            self.expanded.add(self.selected)

    # ── application ───────────────────────────────────────────────────────

    def build_app(self):  # noqa: ANN201 — prompt_toolkit Application
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import HSplit, Layout, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import TextArea

        kb = KeyBindings()

        input_area = TextArea(
            prompt=self._prompt,
            multiline=False,
            wrap_lines=True,
            height=1,
        )

        def _accept(buff) -> bool:  # noqa: ANN001
            text = buff.text
            buff.text = ""
            self._follow = True
            result = self._on_submit(text)
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result)
            return False  # keep the buffer (we already cleared it)

        input_area.buffer.accept_handler = _accept

        body = Window(
            content=FormattedTextControl(
                self._get_text,
                focusable=True,
                get_cursor_position=self._cursor_position,
            ),
            wrap_lines=False,
            always_hide_cursor=True,
        )
        self._body_window = body

        footer = Window(
            FormattedTextControl(lambda: [("class:help", " " + self._footer_fn() + " ")]),
            height=1,
        )

        @kb.add("tab")
        def _(event) -> None:  # noqa: ANN001
            layout = event.app.layout
            if layout.has_focus(input_area):
                layout.focus(body)
            else:
                layout.focus(input_area)

        @kb.add("up", filter=~_has_focus(input_area))
        @kb.add("k", filter=~_has_focus(input_area))
        def _(event) -> None:  # noqa: ANN001
            self._move(-1)

        @kb.add("down", filter=~_has_focus(input_area))
        @kb.add("j", filter=~_has_focus(input_area))
        def _(event) -> None:  # noqa: ANN001
            self._move(1)

        @kb.add("pageup", filter=~_has_focus(input_area))
        def _(event) -> None:  # noqa: ANN001
            self._scroll(-10)

        @kb.add("pagedown", filter=~_has_focus(input_area))
        def _(event) -> None:  # noqa: ANN001
            self._scroll(10)

        @kb.add("enter", filter=~_has_focus(input_area))
        @kb.add(" ", filter=~_has_focus(input_area))
        def _(event) -> None:  # noqa: ANN001
            self._toggle()

        @kb.add("escape", filter=~_has_focus(input_area))
        def _(event) -> None:  # noqa: ANN001
            event.app.layout.focus(input_area)

        style = Style.from_dict(
            {
                "sel": "bold #fabd2f",
                "help": "#928374",
            }
        )
        self.app = Application(
            layout=Layout(HSplit([body, input_area, footer]), focused_element=input_area),
            key_bindings=kb,
            style=style,
            full_screen=True,
            mouse_support=True,
        )
        # Wire the renderer's repaint to this application.
        self.renderer._invalidate = self.invalidate
        return self.app

    async def run(self) -> None:
        """Run the full-screen application until exit."""
        if self.app is None:
            self.build_app()
        await self.app.run_async()


def _has_focus(target):  # noqa: ANN001, ANN202 — prompt_toolkit Condition
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.application import get_app

    return Condition(lambda: get_app().layout.has_focus(target))


__all__ = ["ReplTUI", "TuiRenderer", "flatten_blocks", "render_tool_block"]
