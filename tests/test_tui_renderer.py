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
        out, offsets, total = flatten_blocks(
            [], live_ansi=None, selected=0, expanded=set(), width=40
        )
        assert out == []
        assert offsets == []
        assert total == 0

    def test_offsets_and_total(self):
        out, offsets, total = flatten_blocks(
            self._blocks(3), live_ansi=None, selected=1, expanded=set(), width=40
        )
        assert len(offsets) == 3
        assert offsets[0] == 0
        assert total >= 3  # at least one row per block
        # selected block gets the ▌ gutter somewhere in its fragments
        assert any("▌" in text for _style, text, *_ in out)

    def test_live_region_appended(self):
        _out, offsets, total_no_live = flatten_blocks(
            self._blocks(2), live_ansi=None, selected=0, expanded=set(), width=40
        )
        _out2, offsets2, total_live = flatten_blocks(
            self._blocks(2), live_ansi="thinking…", selected=0, expanded=set(), width=40
        )
        # The live slice adds rows but no new selectable block offset.
        assert offsets2 == offsets
        assert total_live > total_no_live


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
