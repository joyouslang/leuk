"""Tests for the core agent."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from leuk.agent.core import Agent
from leuk.config import AgentConfig, Settings, SQLiteConfig
from leuk.persistence.memory import MemoryStore
from leuk.persistence.sqlite import SQLiteStore
from leuk.tools import create_default_registry
from leuk.types import Message, Role, Session, StreamEvent, StreamEventType, ToolCall

from tests.conftest import MockProvider


@pytest_asyncio.fixture
async def agent_setup(tmp_path: Path):
    """Create a fully wired agent for testing."""
    settings = Settings(
        sqlite=SQLiteConfig(path=str(tmp_path / "test.db")),
        agent=AgentConfig(max_tool_rounds=3),
    )

    provider = MockProvider()
    sqlite = SQLiteStore(settings.sqlite)
    hot_store = MemoryStore()
    tools = create_default_registry()

    agent = Agent(
        settings=settings,
        provider=provider,
        tool_registry=tools,
        sqlite=sqlite,
        hot_store=hot_store,
    )
    await agent.init()

    yield agent, provider, sqlite, hot_store

    await agent.shutdown()
    await sqlite.close()


class TestAgent:
    @pytest.mark.asyncio
    async def test_simple_conversation(self, agent_setup):
        agent, provider, sqlite, _ = agent_setup
        provider.add_response(Message(role=Role.ASSISTANT, content="Hello back!"))

        messages = []
        async for msg in agent.run("Hello"):
            messages.append(msg)

        assert len(messages) == 1
        assert messages[0].role == Role.ASSISTANT
        assert messages[0].content == "Hello back!"

    @pytest.mark.asyncio
    async def test_tool_call_and_response(self, agent_setup):
        agent, provider, sqlite, _ = agent_setup

        # First response: tool call
        tc = ToolCall(id="call_1", name="shell", arguments={"command": "echo test"})
        provider.add_response(
            Message(role=Role.ASSISTANT, content="Let me run that.", tool_calls=[tc])
        )
        # Second response: final text after tool result
        provider.add_response(
            Message(role=Role.ASSISTANT, content="The command output 'test'.")
        )

        messages = []
        async for msg in agent.run("Run echo test"):
            messages.append(msg)

        # Should be: assistant (with tool call), tool result, assistant (final)
        assert len(messages) == 3
        assert messages[0].role == Role.ASSISTANT
        assert messages[0].tool_calls is not None
        assert messages[1].role == Role.TOOL
        assert messages[1].tool_result is not None
        assert "test" in messages[1].tool_result.content
        assert messages[2].role == Role.ASSISTANT

    @pytest.mark.asyncio
    async def test_session_persistence(self, agent_setup):
        agent, provider, sqlite, _ = agent_setup
        provider.add_response(Message(role=Role.ASSISTANT, content="Saved!"))

        async for _ in agent.run("Save this"):
            pass

        # Check messages were persisted
        messages = await sqlite.get_messages(agent.session.id)
        # system + user + assistant
        assert len(messages) >= 2

    @pytest.mark.asyncio
    async def test_session_resume(self, tmp_path: Path):
        settings = Settings(
            sqlite=SQLiteConfig(path=str(tmp_path / "test.db")),
            agent=AgentConfig(max_tool_rounds=3),
        )
        sqlite = SQLiteStore(settings.sqlite)
        hot_store = MemoryStore()
        tools = create_default_registry()

        # Create first agent, have a conversation
        provider1 = MockProvider([Message(role=Role.ASSISTANT, content="First response")])
        agent1 = Agent(
            settings=settings,
            provider=provider1,
            tool_registry=tools,
            sqlite=sqlite,
            hot_store=hot_store,
        )
        await agent1.init()
        session_id = agent1.session.id
        async for _ in agent1.run("Hello"):
            pass
        await agent1.shutdown()

        # Create second agent with same session
        provider2 = MockProvider([Message(role=Role.ASSISTANT, content="Second response")])
        session = Session(id=session_id)
        agent2 = Agent(
            settings=settings,
            provider=provider2,
            tool_registry=tools,
            sqlite=sqlite,
            hot_store=hot_store,
            session=session,
        )
        await agent2.init()

        # Should have loaded previous messages
        assert len(agent2._messages) > 0
        await agent2.shutdown()
        await sqlite.close()

    @pytest.mark.asyncio
    async def test_max_tool_rounds(self, agent_setup):
        agent, provider, _, _ = agent_setup

        # Keep returning tool calls to hit the limit (max_tool_rounds=3)
        tc = ToolCall(id="call_1", name="shell", arguments={"command": "echo loop"})
        for _ in range(10):
            provider.add_response(
                Message(role=Role.ASSISTANT, content="Again.", tool_calls=[tc])
            )
        # The forced final response after exceeding rounds
        provider.add_response(Message(role=Role.ASSISTANT, content="Done."))

        messages = []
        async for msg in agent.run("Loop test"):
            messages.append(msg)

        # Should have 3 rounds of tool calls (each: assistant + tool result = 2 msgs)
        # then the forced text response
        assistant_msgs = [m for m in messages if m.role == Role.ASSISTANT]
        tool_msgs = [m for m in messages if m.role == Role.TOOL]
        # At least 3 tool call rounds happened
        assert len(tool_msgs) >= 3
        # A final response exists (either forced or from exceeding)
        assert len(assistant_msgs) >= 3

    @pytest.mark.asyncio
    async def test_unknown_tool(self, agent_setup):
        agent, provider, _, _ = agent_setup

        tc = ToolCall(id="call_1", name="nonexistent_tool", arguments={})
        provider.add_response(Message(role=Role.ASSISTANT, tool_calls=[tc]))
        provider.add_response(Message(role=Role.ASSISTANT, content="Oops."))

        messages = []
        async for msg in agent.run("Try unknown"):
            messages.append(msg)

        tool_msgs = [m for m in messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_result.is_error


class TestAgentStreaming:
    @pytest.mark.asyncio
    async def test_stream_simple(self, agent_setup):
        agent, provider, _, _ = agent_setup
        provider.add_response(Message(role=Role.ASSISTANT, content="Streamed response"))

        events = []
        async for evt in agent.run_stream("Hello"):
            events.append(evt)

        text_deltas = [e for e in events if isinstance(e, StreamEvent) and e.type == StreamEventType.TEXT_DELTA]
        complete = [e for e in events if isinstance(e, StreamEvent) and e.type == StreamEventType.MESSAGE_COMPLETE]

        assert len(text_deltas) > 0
        assert len(complete) == 1
        assert complete[0].message.content == "Streamed response"
