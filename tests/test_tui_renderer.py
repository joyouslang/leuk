"""Tests for the persistent-input TUI renderer sink (no TTY required)."""

from __future__ import annotations

import asyncio

import pytest

from leuk.cli.blocks import Block, render_static
from leuk.cli.tui import TuiRenderer, flatten_blocks
from rich.text import Text as _RichText
from leuk.types import (
    Message,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
)


def _r() -> TuiRenderer:
    calls: list[int] = []
    r = TuiRenderer(invalidate=lambda: calls.append(1), markdown=True)
    r._invalidate_calls = calls  # type: ignore[attr-defined]
    return r


class TestText:
    def test_delta_sets_live_no_block_until_flush(self):
        r = _r()
        r.handle_event(StreamEvent(type=StreamEventType.TEXT_DELTA, content="hello "))
        r.handle_event(StreamEvent(type=StreamEventType.TEXT_DELTA, content="world"))
        assert r.live_ansi is not None
        assert "hello" in r.live_ansi
        assert r.blocks == []  # not finalized yet

    def test_flush_finalizes_block_and_clears_live(self):
        r = _r()
        r.handle_event(StreamEvent(type=StreamEventType.TEXT_DELTA, content="hi"))
        r.handle_event(StreamEvent(type=StreamEventType.MESSAGE_COMPLETE))
        assert r.live_ansi is None
        assert len(r.blocks) == 1
        assert not r.blocks[0].expandable

    def test_empty_text_makes_no_block(self):
        r = _r()
        r.handle_event(StreamEvent(type=StreamEventType.TEXT_DELTA, content="   "))
        r._flush_text()
        assert r.blocks == []


class TestThinking:
    def test_start_thinking_sets_live(self):
        r = _r()
        r.start_thinking()
        assert r.live_ansi is not None
        assert "Thinking" in r.live_ansi

    def test_text_delta_stops_thinking(self):
        r = _r()
        r.start_thinking()
        r.handle_event(StreamEvent(type=StreamEventType.TEXT_DELTA, content="x"))
        assert r._mode == "text"

    def test_tick_advances_spinner(self):
        r = _r()
        r.start_thinking()
        before = r._frame
        r.tick()
        assert r._frame != before


class TestTools:
    def test_tool_lifecycle_finalizes_expandable_block(self):
        r = _r()
        tc = ToolCall(id="t1", name="shell", arguments={"command": "ls"})
        r.handle_event(StreamEvent(type=StreamEventType.TOOL_CALL_START, tool_call=tc))
        assert r._mode == "tools"
        assert r.live_ansi is not None
        r.handle_event(StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=tc))
        r.handle_message(
            Message(
                role=Role.TOOL,
                tool_result=ToolResult(tool_call_id="t1", name="shell", content="out"),
            )
        )
        # Round finalized: one expandable tool block, live cleared.
        assert r.live_ansi is None
        assert len(r.blocks) == 1
        assert r.blocks[0].expandable


class TestFlatten:
    def _blocks(self, n: int) -> list[Block]:
        from functools import partial

        return [
            Block(False, partial(render_static, _RichText(f"line {i}"))) for i in range(n)
        ]

    def test_empty(self):
        flat = flatten_blocks([], live_ansi=None, expanded=set(), width=40)
        assert flat.fragments == []
        assert flat.block_lines == []
        assert flat.plain_lines == []

    def test_offsets_and_total(self):
        flat = flatten_blocks(self._blocks(3), live_ansi=None, expanded=set(), width=40)
        assert len(flat.block_lines) == 3
        assert flat.block_lines[0] == 0
        assert len(flat.plain_lines) >= 3  # at least one row per block

    def test_live_region_appended(self):
        no_live = flatten_blocks(self._blocks(2), live_ansi=None, expanded=set(), width=40)
        live = flatten_blocks(
            self._blocks(2), live_ansi="thinking…", expanded=set(), width=40
        )
        # The live slice adds rows but no new block offset.
        assert live.block_lines == no_live.block_lines
        assert len(live.plain_lines) > len(no_live.plain_lines)

    def test_selection_highlights_range(self):
        # Select within the single content line of the first block.
        flat = flatten_blocks(
            self._blocks(1),
            live_ansi=None,
            expanded=set(),
            width=40,
            selection=((0, 2), (0, 6)),
        )
        assert any("class:selection" in style for style, _t in _segs(flat.fragments))

    def test_mouse_handler_attached(self):
        sentinel = lambda _e: None  # noqa: E731
        flat = flatten_blocks(
            self._blocks(1), live_ansi=None, expanded=set(), width=40, mouse_handler=sentinel
        )
        assert any(len(frag) == 3 and frag[2] is sentinel for frag in flat.fragments)


def _segs(fragments):
    """Yield (style, text) ignoring any trailing mouse-handler element."""
    return [(f[0], f[1]) for f in fragments]


class TestSelection:
    def _tui(self, plain_lines):
        from leuk.cli.tui import ReplTUI

        tui = ReplTUI(TuiRenderer(), on_submit=lambda x: None)
        tui._plain_lines = plain_lines
        copied: list[str] = []
        tui._copy_to_clipboard = lambda t: copied.append(t)  # type: ignore[method-assign]
        return tui, copied

    def test_copy_strips_gutter_single_line(self):
        # Gutter is 2 chars ("  "); content "hello world" starts at col 2.
        tui, copied = self._tui(["  hello world"])
        tui._sel_start, tui._sel_end = (0, 2), (0, 7)  # "hello"
        tui._copy_selection()
        assert copied == ["hello"]

    def test_copy_multi_line(self):
        tui, copied = self._tui(["  first", "  second", "  third"])
        tui._sel_start, tui._sel_end = (0, 2), (2, 7)
        tui._copy_selection()
        assert copied == ["first\nsecond\nthird"]

    def test_copy_reversed_drag_normalizes(self):
        tui, copied = self._tui(["  alpha"])
        tui._sel_start, tui._sel_end = (0, 7), (0, 2)  # dragged right-to-left
        tui._copy_selection()
        assert copied == ["alpha"]


class TestRenderCaching:
    """The smoothness fix: scrolling must not re-render or rebuild fragments."""

    def _tui_with_blocks(self, render_counter):
        from functools import partial

        from leuk.cli.blocks import Block, render_static
        from leuk.cli.tui import ReplTUI

        def _counting_render(full, width):  # noqa: ANN001
            render_counter.append(1)
            return render_static(_RichText("hello world"), full, width)

        r = TuiRenderer()
        r.blocks = [Block(False, _counting_render), Block(False, partial(render_static, _RichText("two")))]
        return ReplTUI(r, on_submit=lambda x: None)

    def test_scroll_reuses_cached_fragments(self):
        tui = self._tui_with_blocks([])
        f1 = tui._get_text()
        tui._scroll(-1)  # scrolling changes only the viewport, not content
        f2 = tui._get_text()
        assert f1 is f2  # identical object → zero re-flatten on scroll

    def test_block_parsed_only_once_across_repaints(self):
        calls: list[int] = []
        tui = self._tui_with_blocks(calls)
        tui._get_text()
        tui._scroll(-1)
        tui._get_text()
        tui._get_text()
        assert sum(calls) == 1  # the block's expensive render ran exactly once


class TestKeyboardSelection:
    def _tui(self):
        from functools import partial

        from leuk.cli.blocks import Block, render_static
        from leuk.cli.tui import ReplTUI

        r = TuiRenderer()
        r.blocks = [
            Block(False, partial(render_static, _RichText("alpha"))),
            Block(False, partial(render_static, _RichText("beta"))),
        ]
        tui = ReplTUI(r, on_submit=lambda x: None)
        tui._get_text()  # populate _plain_lines / _total_lines
        return tui

    def test_shift_arrow_starts_selection(self):
        tui = self._tui()
        tui._kbd_select(-1)
        assert tui._sel_start is not None and tui._sel_end is not None

    def test_ctrl_c_copies_selection_then_clears(self):
        tui = self._tui()
        copied: list[str] = []
        tui._copy_to_clipboard = lambda t: copied.append(t)  # type: ignore[method-assign]
        tui._kbd_select(-1)  # select the last line
        interrupted: list[int] = []
        tui._on_interrupt = lambda: interrupted.append(1)
        tui.copy_or_interrupt()
        assert copied and copied[0].strip()  # something was copied
        assert tui._sel_start is None  # selection cleared
        assert interrupted == []  # did NOT interrupt while a selection existed

    def test_ctrl_c_interrupts_without_selection(self):
        tui = self._tui()
        interrupted: list[int] = []
        tui._on_interrupt = lambda: interrupted.append(1)
        tui.copy_or_interrupt()
        assert interrupted == [1]


class TestApp:
    def test_build_app_wires_renderer_and_submit(self):
        from leuk.cli.tui import ReplTUI

        submitted: list[str] = []
        r = TuiRenderer()
        tui = ReplTUI(r, on_submit=lambda t: submitted.append(t), prompt="t› ")
        app = tui.build_app()
        assert app is not None
        assert tui.app is app
        # The renderer now repaints through the app.
        assert r._invalidate == tui.invalidate

    def test_theme_style_is_applied(self):
        from leuk.cli.repl import _build_tui_style
        from leuk.cli.theme import PALETTE
        from leuk.cli.tui import ReplTUI

        style = _build_tui_style(PALETTE)
        tui = ReplTUI(TuiRenderer(), on_submit=lambda x: None, style=style)
        app = tui.build_app()
        assert app.style is style  # chrome follows the active colour scheme


class TestApproval:
    @pytest.mark.asyncio
    async def test_approval_resolves_allow_always(self):
        from leuk.cli.tui import ReplTUI
        from leuk.types import ToolCall

        tui = ReplTUI(TuiRenderer(), on_submit=lambda x: None)
        tui.build_app()
        tc = ToolCall(id="t1", name="shell", arguments={"command": "ls"})
        task = asyncio.ensure_future(tui.request_approval("because", tc))
        await asyncio.sleep(0)  # let request_approval register the overlay
        assert tui._approval is not None

        tui._resolve_approval(approved=True, remember=True)
        result = await task
        assert result.approved is True
        assert result.remember is True
        assert tui._approval is None  # overlay cleared

    @pytest.mark.asyncio
    async def test_approval_resolves_deny(self):
        from leuk.cli.tui import ReplTUI
        from leuk.types import ToolCall

        tui = ReplTUI(TuiRenderer(), on_submit=lambda x: None)
        tui.build_app()
        tc = ToolCall(id="t2", name="shell", arguments={"command": "rm -rf /"})
        task = asyncio.ensure_future(tui.request_approval("danger", tc))
        await asyncio.sleep(0)
        # The overlay text surfaces the command for the user to read.
        text = "".join(t for _s, t in tui._approval_text())
        assert "rm -rf /" in text

        tui._resolve_approval(approved=False)
        result = await task
        assert result.approved is False
        assert result.remember is False


class TestConsume:
    @pytest.mark.asyncio
    async def test_consume_until_turn_complete(self):
        r = _r()
        q: asyncio.Queue = asyncio.Queue()
        q.put_nowait(StreamEvent(type=StreamEventType.TEXT_DELTA, content="done"))
        q.put_nowait(StreamEvent(type=StreamEventType.MESSAGE_COMPLETE))
        q.put_nowait(StreamEvent(type=StreamEventType.TURN_COMPLETE))

        await r.consume(q)

        assert r.live_ansi is None
        assert len(r.blocks) == 1

    @pytest.mark.asyncio
    async def test_consume_stops_on_sentinel(self):
        r = _r()
        sentinel = object()
        q: asyncio.Queue = asyncio.Queue()
        q.put_nowait(StreamEvent(type=StreamEventType.TEXT_DELTA, content="partial"))
        q.put_nowait(sentinel)

        await r.consume(q, stop_sentinel=sentinel)

        # finally-block flushes the partial text into a block and clears live.
        assert r.live_ansi is None
        assert len(r.blocks) == 1
