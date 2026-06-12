"""Tests for graceful context-overflow recovery (compact harder + retry)."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio

from leuk.agent.core import Agent, context_overflow_limit
from leuk.config import AgentConfig, Settings, SQLiteConfig
from leuk.persistence.memory import MemoryStore
from leuk.persistence.sqlite import SQLiteStore
from leuk.tools import ToolRegistry
from leuk.types import Message, Role, StreamEvent, StreamEventType

from tests.conftest import MockProvider

LLAMA_MSG = (
    "Error code: 400 - {'error': {'code': 400, 'message': 'request (21873 tokens) "
    "exceeds the available context size (17920 tokens), try increasing it', "
    "'type': 'invalid_request_error'}}"
)


class TestDetection:
    def test_llama_server_message(self):
        assert context_overflow_limit(Exception(LLAMA_MSG)) == 17920

    def test_openai_style(self):
        exc = Exception("This model's maximum context length is 8192 tokens, however...")
        assert context_overflow_limit(exc) == 8192

    def test_anthropic_style(self):
        exc = Exception("prompt is too long: 213071 tokens > 200000 maximum")
        assert context_overflow_limit(exc) == 200000

    def test_hint_without_number(self):
        assert context_overflow_limit(Exception("context_length_exceeded")) == 0

    def test_rate_limit_is_not_overflow(self):
        assert context_overflow_limit(Exception("Rate limit reached, retry in 20s")) is None

    def test_unrelated_error(self):
        assert context_overflow_limit(Exception("connection refused")) is None


class _OverflowOnceProvider(MockProvider):
    """Rejects the first stream() call with a context-overflow 400."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.stream_calls = 0

    async def stream(self, messages, tools=None, **kw) -> AsyncIterator[StreamEvent]:
        self.stream_calls += 1
        if self.stream_calls == 1:
            raise Exception(LLAMA_MSG)
        async for ev in super().stream(messages, tools, **kw):
            yield ev


@pytest_asyncio.fixture
async def overflow_agent(tmp_path: Path):
    provider = _OverflowOnceProvider()
    provider.add_response(Message(role=Role.ASSISTANT, content="Recovered fine."))
    sqlite = SQLiteStore(SQLiteConfig(path=str(tmp_path / "t.db")))
    settings = Settings(agent=AgentConfig(max_tool_rounds=3))
    agent = Agent(
        settings=settings,
        provider=provider,
        tool_registry=ToolRegistry(),
        sqlite=sqlite,
        hot_store=MemoryStore(),
    )
    await agent.init()
    yield agent, provider
    await sqlite.close()


class TestRecovery:
    @pytest.mark.asyncio
    async def test_stream_recovers_from_overflow(self, overflow_agent):
        agent, provider = overflow_agent
        events = [e async for e in agent.run_stream("hello there")]

        # The request was retried and completed.
        assert provider.stream_calls == 2
        completes = [
            e for e in events
            if isinstance(e, StreamEvent) and e.type == StreamEventType.MESSAGE_COMPLETE
        ]
        assert completes and completes[0].message.content == "Recovered fine."

        # The user saw what happened (a status event, not an ERROR).
        notices = [
            e for e in events
            if isinstance(e, StreamEvent)
            and e.type == StreamEventType.RATE_LIMITED
            and "Context overflow" in e.content
        ]
        assert notices

        # The learned clamp respects the server-reported limit with a margin.
        assert agent._window_clamp == int(17920 * 0.9)

        # Nothing was lost: the full history is still persisted.
        stored = await agent.sqlite.get_messages(agent.session.id)
        assert any((m.content or "") == "hello there" for m in stored)

    @pytest.mark.asyncio
    async def test_clamp_persists_for_later_turns(self, overflow_agent):
        agent, provider = overflow_agent
        async for _ in agent.run_stream("first"):
            pass
        provider.add_response(Message(role=Role.ASSISTANT, content="Second answer."))
        async for _ in agent.run_stream("second"):
            pass
        # No new overflow: stream called once for the second turn (3 total),
        # and the clamp is still in force.
        assert provider.stream_calls == 3
        assert agent._window_clamp == int(17920 * 0.9)


class TestLlamaCppProps:
    @pytest.mark.asyncio
    async def test_props_reports_serving_context(self, monkeypatch):
        import httpx

        from leuk.config import LLMConfig
        from leuk.providers.openai import OpenAIProvider

        class _Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"default_generation_settings": {"n_ctx": 17920}}

        async def _get(self_client, url, **kw):
            assert url.endswith("/props")
            return _Resp()

        monkeypatch.setattr(httpx.AsyncClient, "get", _get)
        p = OpenAIProvider(
            LLMConfig(provider="local", local_base_url="http://localhost:8080/v1"),
            base_url="http://localhost:8080/v1",
            api_key="none",  # llama-server needs no key
        )
        info = await p._llamacpp_props_info()
        assert info.context_window == 17920
