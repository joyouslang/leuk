"""Interactive conversation history browser.

A small full-screen ``prompt_toolkit`` overlay, opened with **Tab** from the REPL
prompt (or ``/history``). The conversation is shown as a vertical list of blocks
— user turns, assistant replies, and tool / sub-agent calls. Tool & sub-agent
blocks are **collapsed** by default and expand to their full output:

* ``↑``/``↓`` (or ``k``/``j``) — move the selection
* ``Enter``/``Space`` — expand / collapse the selected tool / sub-agent block
* mouse click — select a block (and toggle its expansion)
* ``Tab``/``Esc``/``q`` — return to the REPL

This replaces the old ``/verbose`` toggle: nothing is permanently truncated, the
full output is always one keypress away.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.styles import Style

from leuk.cli.blocks import Block, build_blocks
from leuk.types import Message

__all__ = ["browse_history", "build_blocks"]


class _HistoryBrowser:
    """Scrolling is driven by a single *cursor line* (`_cursor_line`).

    `get_cursor_position` reports that line to the body `Window`, which then
    auto-scrolls to keep it visible — so moving the cursor past the bottom edge
    scrolls the view past the screen's contents. The mouse wheel and PgUp/PgDn
    just nudge the cursor line; arrows/clicks move the *selected block* and snap
    the cursor to it. Rich re-rendering only happens on selection/expansion
    changes, never on a plain scroll.
    """

    def __init__(self, blocks: list[Block]) -> None:
        self.blocks = blocks
        self.selected = max(0, len(blocks) - 1)  # start at the latest
        self.expanded: set[int] = set()
        self._ft: StyleAndTextTuples = []
        self._block_lines: list[int] = []  # first display row of each block
        self._cursor_line = 0  # the line the viewport follows
        self._total_lines = 0
        self._window: Window | None = None  # the scrollable body window
        self._refresh()
        self._cursor_line = max(0, self._total_lines - 1)  # start at the bottom

    # ── rendering ──────────────────────────────────────────────────
    def _width(self) -> int:
        return max(20, shutil.get_terminal_size((100, 30)).columns - 2)

    def _refresh(self) -> None:
        """Re-render all blocks into formatted text + record per-block line offsets.

        Only called on navigation/expansion (not every paint), so the Rich
        rendering cost stays bounded.
        """
        width = self._width()
        out: StyleAndTextTuples = []
        line_no = 0
        block_lines: list[int] = []
        for i, blk in enumerate(self.blocks):
            block_lines.append(line_no)
            selected = i == self.selected
            gutter_style = "class:sel" if selected else ""
            gutter = "▌ " if selected else "  "
            handler = self._mouse_handler(i)
            ansi = blk.render(i in self.expanded, width)
            fragments = to_formatted_text(ANSI(ansi))

            out.append((gutter_style, gutter, handler))
            for style, text, *_ in fragments:
                parts = text.split("\n")
                for j, part in enumerate(parts):
                    if j > 0:
                        out.append(("", "\n", handler))
                        out.append((gutter_style, gutter, handler))
                        line_no += 1
                    if part:
                        out.append((style, part, handler))
            out.append(("", "\n", handler))
            line_no += 1
        self._ft = out
        self._block_lines = block_lines
        self._total_lines = line_no

    def _mouse_handler(self, index: int) -> Callable[[MouseEvent], object]:
        def handler(event: MouseEvent) -> object:
            et = event.event_type
            if et == MouseEventType.SCROLL_UP:
                self._scroll(-3)
                return None
            if et == MouseEventType.SCROLL_DOWN:
                self._scroll(3)
                return None
            if et == MouseEventType.MOUSE_UP:
                blk = self.blocks[index]
                if self.selected == index and (blk.expandable or blk.on_activate is not None):
                    self._toggle()
                else:
                    self._select(index)
                return None
            return NotImplemented

        return handler

    def _get_text(self) -> StyleAndTextTuples:
        return self._ft

    def _cursor_position(self) -> Point:
        """Reported to the body Window so it scrolls to keep this line visible."""
        return Point(x=0, y=max(0, min(self._cursor_line, max(0, self._total_lines - 1))))

    # ── scrolling ──────────────────────────────────────────────────
    def _scroll(self, delta: int) -> None:
        """Nudge the viewport by *delta* lines (no re-render, no selection change)."""
        if self._total_lines == 0:
            return
        self._cursor_line = max(0, min(self._total_lines - 1, self._cursor_line + delta))

    def _scroll_page(self, direction: int) -> None:
        """Scroll by a screenful in *direction* (+1 down / -1 up)."""
        w = self._window
        height = w.render_info.window_height if (w and w.render_info) else 10
        self._scroll(direction * max(1, height - 1))

    # ── navigation ─────────────────────────────────────────────────
    def _select(self, index: int) -> None:
        self.selected = max(0, min(len(self.blocks) - 1, index))
        self._refresh()
        self._cursor_line = self._block_lines[self.selected]

    def _move(self, delta: int) -> None:
        if not self.blocks:
            return
        self._select(self.selected + delta)

    def _toggle(self) -> None:
        if not self.blocks:
            return
        block = self.blocks[self.selected]
        # Media block: Enter/click opens/plays it via the OS handler.
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
        self._refresh()
        self._cursor_line = self._block_lines[self.selected]

    # ── application ────────────────────────────────────────────────
    def build_app(self) -> Application:
        kb = KeyBindings()

        @kb.add("up")
        @kb.add("k")
        def _(_e):  # noqa: ANN001
            self._move(-1)

        @kb.add("down")
        @kb.add("j")
        def _(_e):  # noqa: ANN001
            self._move(1)

        @kb.add("pageup")
        def _(_e):  # noqa: ANN001
            self._scroll_page(-1)

        @kb.add("pagedown")
        def _(_e):  # noqa: ANN001
            self._scroll_page(1)

        @kb.add("home")
        def _(_e):  # noqa: ANN001
            self._move(-len(self.blocks))

        @kb.add("end")
        def _(_e):  # noqa: ANN001
            self._move(len(self.blocks))

        @kb.add("enter")
        @kb.add(" ")
        def _(_e):  # noqa: ANN001
            self._toggle()

        @kb.add("tab")
        @kb.add("escape")
        @kb.add("q")
        @kb.add("c-c")
        def _(event):  # noqa: ANN001
            event.app.exit()

        # rich pre-wraps each block at (width-2), so logical lines == display
        # rows: keep wrap_lines off so the cursor-line math is exact. The Window
        # follows the reported cursor position, so nudging `_cursor_line` (wheel,
        # PgUp/PgDn, arrows) scrolls the view — including past the screen.
        body = Window(
            content=FormattedTextControl(
                self._get_text,
                focusable=True,
                get_cursor_position=self._cursor_position,
            ),
            wrap_lines=False,
            always_hide_cursor=True,
        )
        self._window = body
        header = Window(
            FormattedTextControl([("class:title", " conversation history ")]),
            height=1,
        )
        footer = Window(
            FormattedTextControl(
                [
                    (
                        "class:help",
                        " ↑/↓ select · Enter expand / open media · scroll/PgUp/PgDn · "
                        "click a block · Tab/Esc back ",
                    )
                ]
            ),
            height=1,
        )
        style = Style.from_dict(
            {
                "sel": "bold #fabd2f",
                "title": "bold #282828 bg:#fabd2f",
                "help": "#928374",
            }
        )
        return Application(
            layout=Layout(HSplit([header, body, footer]), focused_element=body),
            key_bindings=kb,
            style=style,
            full_screen=True,
            mouse_support=True,
        )


async def browse_history(messages: list[Message], *, media_mode: str = "metadata") -> None:
    """Open the interactive history browser over *messages* (returns on exit)."""
    blocks = build_blocks(messages, media_mode=media_mode)
    if not blocks:
        return
    await _HistoryBrowser(blocks).build_app().run_async()
