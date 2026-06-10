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


def _provider() -> AnthropicProvider:
    return AnthropicProvider(LLMConfig(provider="anthropic", anthropic_api_key="x"))


class TestThinkingParam:
    def test_on_by_default(self):
        # No flag — thinking is requested by default (half of max_tokens, capped).
        assert _provider()._thinking_param(16384, has_temperature=False) == {
            "type": "enabled",
            "budget_tokens": 8192,
        }

    def test_temperature_skips_thinking(self):
        # The API only allows temperature == 1 with thinking; skip rather than 400.
        assert _provider()._thinking_param(16384, has_temperature=True) is None

    def test_small_max_tokens_skips_thinking(self):
        # No room for a thinking budget plus an answer (e.g. title generation).
        assert _provider()._thinking_param(20, has_temperature=False) is None

    def test_remembered_rejection_skips_thinking(self):
        p = _provider()
        p._thinking_unsupported = True
        assert p._thinking_param(16384, has_temperature=False) is None


class TestDisableThinking:
    def test_thinking_rejection_strips_and_retries(self):
        p = _provider()
        kwargs = {
            "thinking": {"type": "enabled", "budget_tokens": 8192},
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "x", "signature": "s"},
                        {"type": "text", "text": "hi"},
                    ],
                }
            ],
        }
        exc = Exception("Extended thinking is not supported for this model")
        assert p._disable_thinking(exc, kwargs) is True
        assert p._thinking_unsupported is True
        assert "thinking" not in kwargs
        # Replayed thinking blocks are stripped from the messages too.
        assert kwargs["messages"][0]["content"] == [{"type": "text", "text": "hi"}]

    def test_unrelated_error_is_reraised(self):
        p = _provider()
        kwargs = {"thinking": {"type": "enabled", "budget_tokens": 8192}, "messages": []}
        exc = Exception("image exceeds 10 MB maximum")
        assert p._disable_thinking(exc, kwargs) is False
        assert p._thinking_unsupported is False

    def test_no_thinking_sent_means_no_retry(self):
        p = _provider()
        exc = Exception("thinking something")
        assert p._disable_thinking(exc, {"messages": []}) is False


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


class TestOpenAIReasoningRequest:
    def _provider(self):
        from leuk.providers.openai import OpenAIProvider

        return OpenAIProvider(LLMConfig(provider="openai", openai_api_key="x"))

    def test_rejection_strips_and_remembers(self):
        p = self._provider()
        kwargs = {"model": "m", "extra_body": {"reasoning": {}}}
        exc = Exception("Unrecognized request argument supplied: reasoning")
        assert p._disable_reasoning(exc, kwargs) is True
        assert p._reasoning_unsupported is True
        assert "extra_body" not in kwargs

    def test_unrelated_error_is_reraised(self):
        p = self._provider()
        kwargs = {"model": "m", "extra_body": {"reasoning": {}}}
        assert p._disable_reasoning(Exception("rate limited"), kwargs) is False
        assert p._reasoning_unsupported is False

    def test_no_reasoning_sent_means_no_retry(self):
        p = self._provider()
        assert p._disable_reasoning(Exception("reasoning broke"), {"model": "m"}) is False


class TestThinkingStatus:
    def test_anthropic_states(self):
        p = _provider()
        assert "requested" in p.thinking_status()
        p._thinking_unsupported = True
        assert "rejected" in p.thinking_status()

    def test_anthropic_temperature_explains_off(self):
        cfg = LLMConfig(provider="anthropic", anthropic_api_key="x", temperature=0.2)
        status = AnthropicProvider(cfg).thinking_status()
        assert "off" in status and "temperature" in status

    def test_openai_states(self):
        from leuk.providers.openai import OpenAIProvider

        p = OpenAIProvider(LLMConfig(provider="openai", openai_api_key="x"))
        assert "requested" in p.thinking_status()
        p._reasoning_unsupported = True
        assert "off" in p.thinking_status()


class TestTuiThinkingStream:
    def test_delta_shows_token_count_in_live(self):
        r = _r()
        r.handle_event(StreamEvent(type=StreamEventType.THINKING_DELTA, content="abcdefgh"))
        assert r._mode == "thinking"
        assert r.live_ansi is not None
        assert "~2 tok" in r.live_ansi  # 8 chars ≈ 2 tokens

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
