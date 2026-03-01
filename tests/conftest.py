"""Shared test fixtures."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio

from leuk.config import AgentConfig, LLMConfig, RedisConfig, Settings, SQLiteConfig
from leuk.persistence.memory import MemoryStore
from leuk.persistence.sqlite import SQLiteStore
from leuk.tools import create_default_registry
from leuk.tools.base import ToolRegistry
from leuk.types import Message, Role, StreamEvent, StreamEventType, ToolCall, ToolSpec


class MockProvider:
    """A mock LLM provider for testing."""

    def __init__(self, responses: list[Message] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_count = 0
        self.last_messages: list[Message] = []
        self.last_tools: list[ToolSpec] | None = None

    def add_response(self, msg: Message) -> None:
        self._responses.append(msg)

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Message:
        self.last_messages = messages
        self.last_tools = tools
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return Message(role=Role.ASSISTANT, content="(no more mock responses)")

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        msg = await self.generate(messages, tools, temperature=temperature, max_tokens=max_tokens)
        if msg.content:
            # Simulate streaming by yielding word by word
            words = msg.content.split()
            for w in words:
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, content=w + " ")
        if msg.tool_calls:
            for tc in msg.tool_calls:
                yield StreamEvent(type=StreamEventType.TOOL_CALL_START, tool_call=tc)
                yield StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=tc)
        yield StreamEvent(type=StreamEventType.MESSAGE_COMPLETE, message=msg)

    async def close(self) -> None:
        pass


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest_asyncio.fixture
async def sqlite_store(tmp_path: Path) -> AsyncIterator[SQLiteStore]:
    config = SQLiteConfig(path=str(tmp_path / "test.db"))
    store = SQLiteStore(config)
    await store.init()
    yield store
    await store.close()


@pytest.fixture
def memory_store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return create_default_registry()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm=LLMConfig(provider="mock"),
        redis=RedisConfig(),
        sqlite=SQLiteConfig(path=str(tmp_path / "test.db")),
        agent=AgentConfig(max_tool_rounds=5),
    )
