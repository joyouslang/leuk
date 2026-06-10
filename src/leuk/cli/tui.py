"""Persistent-input TUI — the invalidate-driven renderer sink.

This is the rendering half of the full-screen REPL described in
``docs/repl-tui-design.md``. Where :class:`leuk.cli.render.StreamRenderer`
drives three ``rich.Live`` regions and ``console.print`` (which own the
terminal and so cannot coexist with a ``prompt_toolkit`` full-screen
``Application``), :class:`TuiRenderer` instead:

* appends **finalized** content to ``blocks`` — the shared scrollback model
  (:mod:`leuk.cli.blocks`); and
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
from typing import Any, NamedTuple

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
        media_mode: str = "metadata",
    ) -> None:
        self._invalidate = invalidate or (lambda: None)
        self._width_fn = width_fn or (lambda: 80)
        self.markdown = markdown
        self._media_mode = media_mode

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

        # Streamed reasoning (THINKING_DELTA): accumulated per turn, shown in
        # the live region (collapsed by default, Ctrl-T expands) and frozen
        # into an expandable scrollback block when the answer starts.
        self._thinking_buffer: list[str] = []
        self.thinking_expanded = False  # sticky across turns

    # ── public knobs ──────────────────────────────────────────────────────

    def set_tts_speaker(self, speaker: object | None) -> None:
        """Attach/detach a streaming TTS speaker (sentence-by-sentence speech)."""
        self._tts_speaker = speaker

    def reset(self) -> None:
        """Clear per-turn streaming state (keeps the finalized ``blocks``)."""
        self.tracker.clear()
        self._text_buffer.clear()
        self._thinking_buffer.clear()
        self._in_text_stream = False
        self._current_round = 0
        self._mode = ""
        self.live_ansi = None

    def width(self) -> int:
        return max(20, self._width_fn())

    # ── live-region composition ───────────────────────────────────────────

    def _advance_spinner(self) -> bool:
        """Advance the spinner frame + recompose live; True if a spinner shows."""
        if self._mode in ("thinking", "tools"):
            self._frame = (self._frame + 1) % len(_SPINNER_FRAMES)
            self._compose_live()
            return True
        return False

    def tick(self) -> None:
        """Advance the spinner frame and repaint (for external/timer callers)."""
        if self._advance_spinner():
            self._invalidate()

    def render_tick(self) -> None:
        """Advance the spinner without invalidating — called from a repaint."""
        self._advance_spinner()

    def _spinner(self) -> str:
        return _SPINNER_FRAMES[self._frame % len(_SPINNER_FRAMES)]

    def _compose_live(self) -> None:
        """Recompute ``live_ansi`` from the current mode + state."""
        if self._mode == "thinking":
            elapsed = max(0, int(time.monotonic() - self._thinking_start))
            n_chars = sum(len(s) for s in self._thinking_buffer)
            label = Text()
            label.append(f"{self._spinner()} ", style="accent.purple")
            label.append("Thinking… ", style="italic dim")
            label.append(f"{elapsed}s ", style="comment")
            if n_chars:
                label.append(f"· {n_chars} chars ", style="comment")
                hint = "(^T collapse · Ctrl-C stop)" if self.thinking_expanded else "(^T expand · Ctrl-C stop)"
            else:
                hint = "(Ctrl-C to stop)"
            label.append(hint, style="comment")
            if self.thinking_expanded and n_chars:
                # Live reasoning panel: show the streaming tail, bounded so the
                # live region never outgrows the screen.
                from rich.box import ROUNDED
                from rich.console import Group
                from rich.panel import Panel

                tail = "\n".join("".join(self._thinking_buffer).splitlines()[-15:])
                panel = Panel(
                    Text(tail, style="dim"),
                    box=ROUNDED,
                    border_style="comment",
                    padding=(0, 1),
                    expand=False,
                )
                self.live_ansi = rich_to_ansi(Group(label, panel), self.width())
            else:
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
        self._freeze_thinking()

    def _freeze_thinking(self) -> None:
        """Freeze the streamed reasoning into an expandable scrollback block."""
        text = "".join(self._thinking_buffer).strip()
        self._thinking_buffer.clear()
        if text:
            from leuk.cli.blocks import thinking_block

            self.blocks.append(thinking_block(text))
            self._invalidate()

    def toggle_thinking_expand(self) -> None:
        """Ctrl-T: expand/collapse the live reasoning panel (sticky)."""
        self.thinking_expanded = not self.thinking_expanded
        self._compose_live()
        self._invalidate()

    def _on_thinking_delta(self, event: StreamEvent) -> None:
        if self._mode != "thinking":
            self.start_thinking()
        self._thinking_buffer.append(event.content)
        self._compose_live()
        self._invalidate()

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
            case StreamEventType.THINKING_DELTA:
                self._on_thinking_delta(event)
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
        """Freeze the round's tool calls into blocks (with media as thumbnails)."""
        from leuk.cli.blocks import tool_result_blocks

        self._mode = ""
        self.live_ansi = None
        for ts in self.tracker.all_statuses:
            if ts.result is not None:
                # Strip base64 media into separate thumbnail/metadata blocks
                # instead of dumping the raw blob into the tool block.
                self.blocks.extend(
                    tool_result_blocks(ts.tool_call, ts.result, media_mode=self._media_mode)
                )
            else:
                self.blocks.append(Block(True, partial(render_tool, ts)))
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
            # Always tear down the live region so a spinner never leaks; a
            # dangling reasoning trace (interrupted mid-think) is frozen too.
            self._stop_thinking()
            self._flush_text()
            self._mode = ""
            self.live_ansi = None
            self._invalidate()


# ── Scrollback flattening (pure, testable) ────────────────────────────────

_GUTTER = "  "  # left margin on every content line; also stripped when copying

# A selection range in content coordinates: ((y0, x0), (y1, x1)).
Selection = tuple[tuple[int, int], tuple[int, int]]


class Flattened(NamedTuple):
    """Result of :func:`flatten_blocks`."""

    fragments: StyleAndTextTuples
    block_lines: list[int]   # first content-line index of each block
    plain_lines: list[str]   # plain text of each content line (gutter included)


def _norm_selection(sel: Selection) -> Selection:
    """Return *sel* ordered so the start point precedes the end point."""
    a, b = sel
    return (a, b) if a <= b else (b, a)


def _highlight_line(segs: list[tuple[str, str]], lo: int, hi: int) -> list[tuple[str, str]]:
    """Apply the selection style to columns ``[lo, hi)`` of one line's segments."""
    if hi <= lo:
        return segs
    out: list[tuple[str, str]] = []
    col = 0
    for style, text in segs:
        start, end = col, col + len(text)
        col = end
        if end <= lo or start >= hi:
            out.append((style, text))
            continue
        a = max(lo, start) - start
        b = min(hi, end) - start
        if a > 0:
            out.append((style, text[:a]))
        out.append((f"{style} class:selection".strip(), text[a:b]))
        if b < len(text):
            out.append((style, text[b:]))
    return out


def _ansi_to_line_segs(ansi: str) -> list[list[tuple[str, str]]]:
    """Split an ANSI string into per-content-line ``(style, text)`` segments,
    each prefixed with the left gutter. This is the expensive step (ANSI parse)
    that callers cache per block."""
    lines: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = [("", _GUTTER)]
    for style, text, *_ in to_formatted_text(ANSI(ansi)):
        parts = text.split("\n")
        for j, part in enumerate(parts):
            if j > 0:
                lines.append(cur)
                cur = [("", _GUTTER)]
            if part:
                cur.append((style, part))
    lines.append(cur)
    return lines


def _selection_bounds(i: int, sel: Selection, line_len: int) -> tuple[int, int]:
    """The ``[lo, hi)`` column range to highlight on content line *i*."""
    (y0, x0), (y1, x1) = sel
    if y0 == y1:
        return x0, x1
    if i == y0:
        return x0, line_len
    if i == y1:
        return 0, x1
    return 0, line_len


def emit_lines(
    line_segs: list[list[tuple[str, str]]],
    plain_lines: list[str],
    *,
    selection: Selection | None = None,
    mouse_handler: Callable[[Any], Any] | None = None,
) -> StyleAndTextTuples:
    """Emit prompt_toolkit fragments from per-line segments (cheap; no ANSI parse).

    Applies the *selection* highlight to copies of affected lines (never mutates
    the input, so cached segment lists stay clean) and attaches *mouse_handler*.
    """
    sel = _norm_selection(selection) if selection is not None else None
    y0 = sel[0][0] if sel else -1
    y1 = sel[1][0] if sel else -1
    out: StyleAndTextTuples = []
    for i, segs in enumerate(line_segs):
        if sel is not None and y0 <= i <= y1:
            lo, hi = _selection_bounds(i, sel, len(plain_lines[i]))
            segs = _highlight_line(segs, lo, hi)
        for style, text in segs:
            out.append((style, text, mouse_handler) if mouse_handler else (style, text))
        out.append(("", "\n"))
    return out


def flatten_blocks(
    blocks: list[Block],
    *,
    live_ansi: str | None,
    expanded: set[int],
    width: int,
    selection: Selection | None = None,
    mouse_handler: Callable[[Any], Any] | None = None,
) -> Flattened:
    """Render *blocks* (+ optional ``live_ansi`` slice) into formatted text.

    Kept pure and uncached for unit-testing; the live TUI uses the cached
    :func:`_ansi_to_line_segs` + :func:`emit_lines` path in ``ReplTUI``.
    """
    line_segs: list[list[tuple[str, str]]] = []
    block_lines: list[int] = []
    for i, blk in enumerate(blocks):
        block_lines.append(len(line_segs))
        line_segs.extend(_ansi_to_line_segs(blk.render(i in expanded, width)))
    if live_ansi:
        line_segs.extend(_ansi_to_line_segs(live_ansi))
    plain_lines = ["".join(t for _s, t in segs) for segs in line_segs]
    out = emit_lines(line_segs, plain_lines, selection=selection, mouse_handler=mouse_handler)
    return Flattened(out, block_lines, plain_lines)


# ── Full-screen REPL Application ───────────────────────────────────────────


class ReplTUI:
    """Full-screen REPL: scrollback (blocks + live region) + persistent input.

    The input box is **always typable**, even while the agent streams — the
    renderer repaints the scrollback via ``invalidate()`` while the input keeps
    focus. ``Tab`` completes slash-commands; the mouse scrolls / drag-selects /
    clicks the scrollback, and PgUp/PgDn scroll it too.

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
        on_interrupt: Callable[[], None] | None = None,
        footer_fn: Callable[[], str] | None = None,
        completer: Any = None,
        history: Any = None,
        style: Any = None,
        prompt: str = "leuk› ",
    ) -> None:
        self.renderer = renderer
        self._on_submit = on_submit
        self._on_interrupt = on_interrupt
        self._footer_fn = footer_fn or (lambda: "")
        self._completer = completer
        self._history = history
        self._style = style
        self._prompt = prompt

        self.expanded: set[int] = set()
        self._cursor_line = 0
        self._total_lines = 0
        self._follow = True  # auto-scroll to bottom while new content streams
        self._block_lines: list[int] = []
        self._plain_lines: list[str] = []
        self._body_window: Any = None
        self.app: Any = None
        # Active tool-approval request (overlay float), or None.
        self._approval: dict[str, Any] | None = None
        # Text selection in content coordinates (anchor / focus), drag state.
        self._sel_start: tuple[int, int] | None = None
        self._sel_end: tuple[int, int] | None = None
        self._selecting = False
        # Keyboard (line-granular) selection anchor/focus.
        self._kbd_anchor: int | None = None
        self._kbd_focus = 0

        # ── render caches (keep scrolling/dragging buttery smooth) ──
        # Per-block parsed line segments, keyed by (id(block), expanded). The
        # ANSI parse / rich render only runs when a block is new or toggled.
        self._seg_cache: dict[tuple[int, bool], list[list[tuple[str, str]]]] = {}
        self._cache_width = -1
        self._assembled_sig: Any = None
        self._assembled: tuple[list[list[tuple[str, str]]], list[int], list[str]] = ([], [], [])
        self._paint_sig: Any = None
        self._paint_frags: StyleAndTextTuples = []

    # ── width ─────────────────────────────────────────────────────────────

    def _width(self) -> int:
        return max(20, shutil.get_terminal_size((100, 30)).columns - 2)

    # ── scrollback control ────────────────────────────────────────────────

    def _assemble(
        self, blocks: list[Block], live_ansi: str | None, width: int
    ) -> tuple[list[list[tuple[str, str]]], list[int], list[str]]:
        """Build per-line segments, reusing the per-block parse cache."""
        line_segs: list[list[tuple[str, str]]] = []
        block_lines: list[int] = []
        for i, blk in enumerate(blocks):
            block_lines.append(len(line_segs))
            key = (id(blk), i in self.expanded)
            segs = self._seg_cache.get(key)
            if segs is None:
                segs = _ansi_to_line_segs(blk.render(i in self.expanded, width))
                self._seg_cache[key] = segs
            line_segs.extend(segs)
        if live_ansi:
            line_segs.extend(_ansi_to_line_segs(live_ansi))
        plain_lines = ["".join(t for _s, t in segs) for segs in line_segs]
        return line_segs, block_lines, plain_lines

    def _get_text(self) -> StyleAndTextTuples:
        # Animate the spinner on each repaint (driven by the app's refresh_interval).
        self.renderer.render_tick()
        width = self._width()
        if width != self._cache_width:  # terminal resized → drop the parse cache
            self._seg_cache.clear()
            self._cache_width = width
            self._assembled_sig = self._paint_sig = None

        blocks = self.renderer.blocks
        live = self.renderer.live_ansi
        # Content signature: only changes when a block is added/expanded or the
        # live region changes — NOT when merely scrolling.
        sig = (tuple((id(b), i in self.expanded) for i, b in enumerate(blocks)), live)
        if sig != self._assembled_sig:
            self._assembled = self._assemble(blocks, live, width)
            self._assembled_sig = sig

        line_segs, block_lines, plain_lines = self._assembled
        self._block_lines = block_lines
        self._plain_lines = plain_lines
        self._total_lines = len(plain_lines)
        if self._follow:
            self._cursor_line = max(0, self._total_lines - 1)

        sel: Selection | None = None
        if self._sel_start is not None and self._sel_end is not None:
            sel = (self._sel_start, self._sel_end)

        paint_sig = (sig, sel)
        if paint_sig == self._paint_sig:  # scrolling only → reuse fragments verbatim
            return self._paint_frags
        self._paint_frags = emit_lines(
            line_segs, plain_lines, selection=sel, mouse_handler=self._mouse
        )
        self._paint_sig = paint_sig
        return self._paint_frags

    def _cursor_position(self):  # noqa: ANN202 — prompt_toolkit Point
        from prompt_toolkit.data_structures import Point

        return Point(x=0, y=max(0, min(self._cursor_line, max(0, self._total_lines - 1))))

    def invalidate(self) -> None:
        """Request a repaint (wired into the renderer)."""
        if self.app is not None:
            self.app.invalidate()

    # ── tool approval (in-app overlay) ─────────────────────────────────────

    async def request_approval(self, reason: str, tool_call: Any) -> Any:
        """Show a modal approval float and await the user's choice.

        Returns an :class:`leuk.safety.ApprovalResult`. Used in place of the
        standalone approval dialog while the full-screen app is running (a
        nested prompt_toolkit Application can't run inside this one).
        """
        from leuk.safety import ApprovalResult

        if self.app is None:
            return ApprovalResult(approved=False)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self._approval = {"reason": reason, "tool_call": tool_call, "future": fut}
        self.invalidate()
        try:
            return await fut
        finally:
            self._approval = None
            self.invalidate()

    def _resolve_approval(self, *, approved: bool, remember: bool = False) -> None:
        from leuk.safety import ApprovalResult

        a = self._approval
        if a is not None and not a["future"].done():
            a["future"].set_result(ApprovalResult(approved=approved, remember=remember))

    def _approval_text(self):  # noqa: ANN202 — prompt_toolkit formatted text
        from leuk.cli.approval import humanise, primary_detail

        a = self._approval
        if a is None:
            return []
        tc = a["tool_call"]
        body = (
            f"🔐 Permission required\n\n"
            f"{humanise(tc)}\n{primary_detail(tc)}\n\n"
            f"{a['reason']}\n\n"
            f"↵ allow once   a always allow   d always deny   Esc deny"
        )
        return [("class:approval", body)]

    # ── scrolling ─────────────────────────────────────────────────────────

    def _scroll(self, delta: int) -> None:
        """Scroll the transcript by *delta* lines; re-engage follow at bottom."""
        if self._total_lines == 0:
            return
        self._cursor_line = max(0, min(self._total_lines - 1, self._cursor_line + delta))
        self._follow = self._cursor_line >= self._total_lines - 1

    def _page(self) -> int:
        ri = getattr(self._body_window, "render_info", None)
        if ri is not None:
            try:
                return max(1, ri.window_height - 1)
            except Exception:
                pass
        return 10

    def jump_to_bottom(self) -> None:
        self._follow = True
        self._cursor_line = max(0, self._total_lines - 1)

    def _autoscroll(self, y: int) -> None:
        """Scroll while dragging a selection past the visible top/bottom edge."""
        ri = getattr(self._body_window, "render_info", None)
        if ri is None:
            return
        try:
            top, bottom = ri.first_visible_line(), ri.last_visible_line()
        except Exception:
            return
        if y <= top:
            self._scroll(-2)
        elif y >= bottom:
            self._scroll(2)

    # ── mouse: scroll, drag-select, click-to-activate ─────────────────────

    def _mouse(self, event: Any) -> Any:
        from prompt_toolkit.mouse_events import MouseEventType as MET

        et = event.event_type
        if et == MET.SCROLL_UP:
            self._scroll(-3)
            return None
        if et == MET.SCROLL_DOWN:
            self._scroll(3)
            return None

        pos = (event.position.y, event.position.x)
        if et == MET.MOUSE_DOWN:
            self._sel_start = self._sel_end = pos
            self._selecting = True
            self._kbd_anchor = None  # a fresh mouse selection resets keyboard mode
            return None
        if et == MET.MOUSE_MOVE:
            if self._selecting:
                self._sel_end = pos
                self._autoscroll(pos[0])
                return None
            return NotImplemented
        if et == MET.MOUSE_UP:
            if self._selecting:
                self._selecting = False
                self._sel_end = pos
                if self._sel_start == self._sel_end:
                    self._sel_start = self._sel_end = None
                    self._click(pos[0])
                else:
                    self._copy_selection()
            return None
        return NotImplemented

    def _click(self, y: int) -> None:
        """A plain click (no drag) on content line *y*: open/expand its block."""
        import bisect

        if not self._block_lines:
            return
        idx = bisect.bisect_right(self._block_lines, y) - 1
        if idx < 0 or idx >= len(self.renderer.blocks):
            return
        block = self.renderer.blocks[idx]
        if block.on_activate is not None:
            try:
                block.on_activate()  # e.g. open an image externally
            except Exception:  # noqa: BLE001 — opening is best-effort
                pass
            return
        if block.expandable:
            self.expanded.discard(idx) if idx in self.expanded else self.expanded.add(idx)

    def _copy_selection(self) -> None:
        if self._sel_start is None or self._sel_end is None or not self._plain_lines:
            return
        (y0, x0), (y1, x1) = _norm_selection((self._sel_start, self._sel_end))
        n = len(self._plain_lines)
        y0, y1 = max(0, min(y0, n - 1)), max(0, min(y1, n - 1))
        parts: list[str] = []
        for i in range(y0, y1 + 1):
            ln = self._plain_lines[i]
            if y0 == y1:
                lo, hi = x0, x1
            elif i == y0:
                lo, hi = x0, len(ln)
            elif i == y1:
                lo, hi = 0, x1
            else:
                lo, hi = 0, len(ln)
            lo, hi = max(lo, len(_GUTTER)), max(hi, len(_GUTTER))  # drop the gutter
            parts.append(ln[lo:hi])
        text = "\n".join(parts).rstrip()
        if text:
            self._copy_to_clipboard(text)

    def _copy_to_clipboard(self, text: str) -> None:
        import base64

        if self.app is None:
            return
        try:
            from prompt_toolkit.clipboard import ClipboardData

            self.app.clipboard.set_data(ClipboardData(text))
        except Exception:  # noqa: BLE001 — internal clipboard is best-effort
            pass
        # System clipboard via OSC52 (works in Konsole and over SSH).
        try:
            b64 = base64.b64encode(text.encode("utf-8", "replace")).decode("ascii")
            self.app.output.write_raw(f"\x1b]52;c;{b64}\x07")
            self.app.output.flush()
        except Exception:  # noqa: BLE001 — OSC52 unsupported on some terminals
            pass

    def _jump_handler(self, event: Any) -> Any:
        from prompt_toolkit.mouse_events import MouseEventType as MET

        if event.event_type == MET.MOUSE_UP:
            self.jump_to_bottom()
            return None
        return NotImplemented

    # ── keyboard selection (Shift+Up/Down, line-granular) ─────────────────

    def _kbd_select(self, direction: int) -> None:
        """Extend a line-granular selection up/down with Shift+Arrow."""
        n = self._total_lines
        if n == 0:
            return
        if self._kbd_anchor is None:
            if self._sel_start is not None and self._sel_end is not None:
                # Continue from an existing (mouse) selection.
                (a0, _), (b0, _) = _norm_selection((self._sel_start, self._sel_end))
                self._kbd_anchor, self._kbd_focus = a0, b0
                self._kbd_focus = max(0, min(self._kbd_focus + direction, n - 1))
            else:
                ri = getattr(self._body_window, "render_info", None)
                try:
                    base = ri.last_visible_line() if ri else n - 1
                except Exception:
                    base = n - 1
                self._kbd_anchor = self._kbd_focus = max(0, min(base, n - 1))
        else:
            self._kbd_focus = max(0, min(self._kbd_focus + direction, n - 1))

        a, b = sorted((self._kbd_anchor, self._kbd_focus))
        self._sel_start = (a, 0)
        self._sel_end = (b, len(self._plain_lines[b]) if b < len(self._plain_lines) else 0)
        self._cursor_line = self._kbd_focus
        self._follow = False

    def _clear_selection(self) -> bool:
        """Drop any active selection; return True if there was one."""
        had = self._sel_start is not None or self._sel_end is not None
        self._sel_start = self._sel_end = None
        self._kbd_anchor = None
        return had

    def copy_or_interrupt(self) -> None:
        """Ctrl-C: copy the selection if there is one, else interrupt the turn."""
        if self._sel_start is not None and self._sel_end is not None and not self._selecting:
            self._copy_selection()
            self._clear_selection()
            return
        if self._on_interrupt is not None:
            self._on_interrupt()

    # ── application ───────────────────────────────────────────────────────

    def build_app(self):  # noqa: ANN201 — prompt_toolkit Application
        from prompt_toolkit.application import Application
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import (
            ConditionalContainer,
            Float,
            FloatContainer,
            HSplit,
            Layout,
            Window,
        )
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.layout.dimension import Dimension
        from prompt_toolkit.layout.menus import CompletionsMenu
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import Frame, TextArea

        kb = KeyBindings()
        approval_active = Condition(lambda: self._approval is not None)

        input_area = TextArea(
            prompt=[("class:prompt", self._prompt)],  # themed prompt label
            multiline=False,
            wrap_lines=True,
            height=1,
            completer=self._completer,
            complete_while_typing=True,
            history=self._history,  # Up/Down navigate REPL history
        )

        def _accept(buff) -> bool:  # noqa: ANN001
            text = buff.text
            self._follow = True
            result = self._on_submit(text)
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result)
            return False  # returning False resets (clears) the input buffer

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

        # Tab / Shift-Tab drive slash-command completion (not pane switching).
        @kb.add("tab", filter=~approval_active)
        def _(event) -> None:  # noqa: ANN001
            buf = event.app.current_buffer
            if buf.complete_state:
                buf.complete_next()
            else:
                buf.start_completion(select_first=False)

        @kb.add("s-tab", filter=~approval_active)
        def _(event) -> None:  # noqa: ANN001
            buf = event.app.current_buffer
            if buf.complete_state:
                buf.complete_previous()
            else:
                buf.start_completion(select_last=True)

        @kb.add("c-d")
        def _(event) -> None:  # noqa: ANN001 — Ctrl-D quits the session
            event.app.exit()

        @kb.add("c-c")
        def _(event) -> None:  # noqa: ANN001 — copy selection, else interrupt turn
            self.copy_or_interrupt()

        @kb.add("c-t")
        def _(event) -> None:  # noqa: ANN001 — expand/collapse the live reasoning panel
            self.renderer.toggle_thinking_expand()

        # Scroll the transcript without leaving the input (input stays focused).
        @kb.add("pageup", filter=~approval_active)
        def _(event) -> None:  # noqa: ANN001
            self._scroll(-self._page())

        @kb.add("pagedown", filter=~approval_active)
        def _(event) -> None:  # noqa: ANN001
            self._scroll(self._page())

        # Keyboard selection: Shift+Up/Down extend a line selection (^C copies).
        @kb.add("s-up", filter=~approval_active)
        def _(event) -> None:  # noqa: ANN001
            self._kbd_select(-1)

        @kb.add("s-down", filter=~approval_active)
        def _(event) -> None:  # noqa: ANN001
            self._kbd_select(1)

        @kb.add("escape", filter=~approval_active, eager=True)
        def _(event) -> None:  # noqa: ANN001 — clear an active selection
            self._clear_selection()

        # ── approval-overlay bindings (eager: win over input/navigation) ──
        @kb.add("enter", filter=approval_active, eager=True)
        def _(event) -> None:  # noqa: ANN001
            self._resolve_approval(approved=True)

        @kb.add("escape", filter=approval_active, eager=True)
        def _(event) -> None:  # noqa: ANN001
            self._resolve_approval(approved=False)

        @kb.add("a", filter=approval_active, eager=True)
        def _(event) -> None:  # noqa: ANN001
            self._resolve_approval(approved=True, remember=True)

        @kb.add("d", filter=approval_active, eager=True)
        def _(event) -> None:  # noqa: ANN001
            self._resolve_approval(approved=False, remember=True)

        approval_float = Float(
            content=ConditionalContainer(
                Frame(
                    Window(
                        FormattedTextControl(self._approval_text),
                        height=Dimension(min=5),
                    ),
                    title="Permission required",
                ),
                filter=approval_active,
            ),
        )
        # Slash-command completion dropdown, anchored to the input cursor.
        completion_float = Float(
            xcursor=True,
            ycursor=True,
            content=CompletionsMenu(max_height=12, scroll_offset=1),
        )

        # "Jump to latest" button — shown only while scrolled up off the bottom.
        jump_bar = ConditionalContainer(
            Window(
                FormattedTextControl(
                    lambda: [("class:jump", "  ⤓ Jump to latest  ", self._jump_handler)]
                ),
                height=1,
            ),
            filter=Condition(lambda: not self._follow),
        )

        # Use the theme-derived style when provided (so the prompt, footer,
        # completion menu, frame, etc. follow the active colour scheme); fall
        # back to a minimal default for standalone/test use.
        style = self._style or Style.from_dict(
            {
                "selection": "reverse",
                "help": "#928374",
                "approval": "#fabd2f",
                "jump": "bg:#504945 #fabd2f bold",
            }
        )
        root = FloatContainer(
            content=HSplit([body, jump_bar, input_area, footer]),
            floats=[completion_float, approval_float],
        )
        self.app = Application(
            layout=Layout(root, focused_element=input_area),
            key_bindings=kb,
            style=style,
            full_screen=True,
            mouse_support=True,
            refresh_interval=0.2,  # animate the spinner / drain streamed deltas
        )
        # Wire the renderer's repaint to this application.
        self.renderer._invalidate = self.invalidate
        return self.app

    async def run(self) -> None:
        """Run the full-screen application until exit."""
        if self.app is None:
            self.build_app()
        await self.app.run_async()


__all__ = ["ReplTUI", "TuiRenderer", "flatten_blocks", "render_tool_block"]
