"""Tests for the persistent-input TUI renderer sink (no TTY required)."""

from __future__ import annotations

import asyncio

import pytest

from leuk.cli.tui import TuiRenderer
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
