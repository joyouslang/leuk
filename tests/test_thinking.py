"""Tests for the thinking/reasoning stream (refactor-plan §5.3a)."""

from __future__ import annotations

import asyncio

import pytest

from leuk.cli.tui import TuiRenderer
from leuk.config import LLMConfig
from leuk.providers.anthropic import AnthropicProvider
from leuk.types import Message, Role, StreamEvent, StreamEventType


def _r() -> TuiRenderer:
    return TuiRenderer(markdown=True)


class TestThinkingParam:
    def test_off_by_default(self):
        cfg = LLMConfig(provider="anthropic", anthropic_api_key="x")
        assert AnthropicProvider(cfg)._thinking_param() is None

    def test_enabled_sends_budget(self):
        cfg = LLMConfig(
            provider="anthropic", anthropic_api_key="x", thinking=True, thinking_budget=4096
        )
        assert AnthropicProvider(cfg)._thinking_param() == {
            "type": "enabled",
            "budget_tokens": 4096,
        }


class TestThinkingReplay:
    def test_assistant_tool_call_replays_thinking_blocks(self):
        from leuk.types import ToolCall

        msg = Message(
            role=Role.ASSISTANT,
            content="Let me check.",
            tool_calls=[ToolCall(id="t1", name="shell", arguments={})],
            metadata={"_thinking_blocks": [{"thinking": "reasoning…", "signature": "sig"}]},
        )
        _system, out = AnthropicProvider._to_anthropic_messages([msg])
        blocks = out[0]["content"]
        # The thinking block must come first (API requirement for tool use).
        assert blocks[0] == {"type": "thinking", "thinking": "reasoning…", "signature": "sig"}
        assert blocks[1]["type"] == "text"
        assert blocks[2]["type"] == "tool_use"

    def test_unsigned_thinking_blocks_are_not_replayed(self):
        from leuk.types import ToolCall

        msg = Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="t1", name="shell", arguments={})],
            metadata={"_thinking_blocks": [{"thinking": "x", "signature": ""}]},
        )
        _system, out = AnthropicProvider._to_anthropic_messages([msg])
        assert all(b["type"] != "thinking" for b in out[0]["content"])


class TestTuiThinkingStream:
    def test_delta_shows_char_count_in_live(self):
        r = _r()
        r.handle_event(StreamEvent(type=StreamEventType.THINKING_DELTA, content="abcde"))
        assert r._mode == "thinking"
        assert r.live_ansi is not None
        assert "5 chars" in r.live_ansi

    def test_expand_shows_reasoning_tail(self):
        r = _r()
        r.handle_event(
            StreamEvent(type=StreamEventType.THINKING_DELTA, content="secret reasoning")
        )
        assert "secret reasoning" not in (r.live_ansi or "")  # collapsed
        r.toggle_thinking_expand()
        assert "secret reasoning" in (r.live_ansi or "")
        r.toggle_thinking_expand()
        assert "secret reasoning" not in (r.live_ansi or "")

    def test_text_delta_freezes_thinking_into_block(self):
        r = _r()
        r.handle_event(StreamEvent(type=StreamEventType.THINKING_DELTA, content="plan it"))
        r.handle_event(StreamEvent(type=StreamEventType.TEXT_DELTA, content="answer"))
        r.handle_event(StreamEvent(type=StreamEventType.MESSAGE_COMPLETE))
        # Two finalized blocks: thinking (expandable, first) then the answer.
        assert len(r.blocks) == 2
        assert r.blocks[0].expandable
        assert "thinking" in r.blocks[0].render(False, 60)
        assert "plan it" in r.blocks[0].render(True, 60)

    @pytest.mark.asyncio
    async def test_interrupt_mid_think_freezes_trace(self):
        r = _r()
        sentinel = object()
        q: asyncio.Queue = asyncio.Queue()
        q.put_nowait(StreamEvent(type=StreamEventType.THINKING_DELTA, content="partial thought"))
        q.put_nowait(sentinel)
        await r.consume(q, stop_sentinel=sentinel)
        assert r.live_ansi is None
        assert len(r.blocks) == 1
        assert "partial thought" in r.blocks[0].render(True, 60)


class TestHistoryAndPersistence:
    def test_build_blocks_renders_thinking(self):
        from leuk.cli.blocks import build_blocks

        msgs = [
            Message(role=Role.USER, content="q"),
            Message(role=Role.ASSISTANT, content="a", thinking="because reasons"),
        ]
        blocks = build_blocks(msgs)
        # user line, thinking block, assistant markdown
        assert len(blocks) == 3
        assert blocks[1].expandable
        assert "because reasons" in blocks[1].render(True, 60)

    @pytest.mark.asyncio
    async def test_thinking_roundtrips_through_sqlite(self, tmp_path):
        from leuk.config import SQLiteConfig
        from leuk.persistence.sqlite import SQLiteStore
        from leuk.types import Session

        store = SQLiteStore(SQLiteConfig(path=str(tmp_path / "t.db")))
        await store.init()
        sess = Session()
        await store.create_session(sess)
        await store.append_message(
            sess.id, Message(role=Role.ASSISTANT, content="a", thinking="deep thought")
        )
        (loaded,) = await store.get_messages(sess.id)
        assert loaded.thinking == "deep thought"
        assert "_thinking" not in loaded.metadata  # unpacked, not leaked
        await store.close()
