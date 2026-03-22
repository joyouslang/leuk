"""Tests for the AgentSession background loop."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from leuk.agent.core import Agent
from leuk.agent.session import AgentSession, _STOP_SENTINEL
from leuk.config import AgentConfig, Settings, SQLiteConfig
from leuk.persistence.memory import MemoryStore
from leuk.persistence.sqlite import SQLiteStore
from leuk.tools import create_default_registry
from leuk.types import (
    AgentState,
    Message,
    Role,
    StreamEvent,
    StreamEventType,
)

from tests.conftest import MockProvider


# ── Fixtures ──────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def session_setup(tmp_path: Path):
    """Create a fully wired AgentSession for testing."""
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

    agent_session = AgentSession(agent)

    yield agent_session, provider, sqlite, hot_store

    if agent_session.running:
        await agent_session.stop()
    await agent.shutdown()
    await sqlite.close()


# ── Helpers ───────────────────────────────────────────────────────


async def _collect_events(
    queue: asyncio.Queue, *, timeout: float = 5.0
) -> list[StreamEvent | Message]:
    """Collect events from a queue until TURN_COMPLETE or timeout."""
    events: list[StreamEvent | Message] = []
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        try:
            event = await asyncio.wait_for(queue.get(), timeout=remaining)
        except asyncio.TimeoutError:
            break
        if event is _STOP_SENTINEL:
            break
        if isinstance(event, StreamEvent) and event.type == StreamEventType.TURN_COMPLETE:
            events.append(event)
            break
        events.append(event)
    return events


# ── Tests ─────────────────────────────────────────────────────────


class TestAgentSessionLifecycle:
    """Test start/stop/running lifecycle."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self, session_setup):
        agent_session, provider, _, _ = session_setup
        assert not agent_session.running

        agent_session.start()
        assert agent_session.running
        assert agent_session.state == AgentState.IDLE

        await agent_session.stop()
        assert not agent_session.running
        assert agent_session.state == AgentState.STOPPED

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self, session_setup):
        agent_session, _, _, _ = session_setup
        agent_session.start()
        agent_session.start()  # should not crash
        assert agent_session.running
        await agent_session.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, session_setup):
        agent_session, _, _, _ = session_setup
        await agent_session.stop()  # should not crash

    @pytest.mark.asyncio
    async def test_session_property(self, session_setup):
        agent_session, _, _, _ = session_setup
        assert agent_session.session is agent_session.agent.session


class TestAgentSessionConversation:
    """Test pushing messages and receiving events."""

    @pytest.mark.asyncio
    async def test_simple_response(self, session_setup):
        agent_session, provider, _, _ = session_setup
        provider.add_response(Message(role=Role.ASSISTANT, content="Hello!"))

        agent_session.start()
        agent_session.push("Hi there")

        events = await _collect_events(agent_session.event_queue)

        # Should have state changes, text deltas, message complete, turn complete
        text_deltas = [
            e for e in events if isinstance(e, StreamEvent) and e.type == StreamEventType.TEXT_DELTA
        ]
        turn_complete = [
            e
            for e in events
            if isinstance(e, StreamEvent) and e.type == StreamEventType.TURN_COMPLETE
        ]

        assert len(text_deltas) > 0
        assert len(turn_complete) == 1

        await agent_session.stop()

    @pytest.mark.asyncio
    async def test_multiple_turns(self, session_setup):
        agent_session, provider, _, _ = session_setup
        provider.add_response(Message(role=Role.ASSISTANT, content="First"))
        provider.add_response(Message(role=Role.ASSISTANT, content="Second"))

        agent_session.start()

        # Turn 1
        agent_session.push("Hello")
        events1 = await _collect_events(agent_session.event_queue)
        assert any(
            isinstance(e, StreamEvent) and e.type == StreamEventType.TURN_COMPLETE for e in events1
        )

        # Turn 2
        agent_session.push("Again")
        events2 = await _collect_events(agent_session.event_queue)
        assert any(
            isinstance(e, StreamEvent) and e.type == StreamEventType.TURN_COMPLETE for e in events2
        )

        await agent_session.stop()

    @pytest.mark.asyncio
    async def test_tool_call_events(self, session_setup):
        from leuk.types import ToolCall

        agent_session, provider, _, _ = session_setup

        tc = ToolCall(id="call_1", name="shell", arguments={"command": "echo test"})
        provider.add_response(
            Message(role=Role.ASSISTANT, content="Let me check.", tool_calls=[tc])
        )
        provider.add_response(Message(role=Role.ASSISTANT, content="Done."))

        agent_session.start()
        agent_session.push("Run echo test")

        events = await _collect_events(agent_session.event_queue)

        # Should have tool call start/end events, tool result messages, and text
        tool_starts = [
            e
            for e in events
            if isinstance(e, StreamEvent) and e.type == StreamEventType.TOOL_CALL_START
        ]
        tool_ends = [
            e
            for e in events
            if isinstance(e, StreamEvent) and e.type == StreamEventType.TOOL_CALL_END
        ]
        tool_results = [e for e in events if isinstance(e, Message) and e.role == Role.TOOL]

        assert len(tool_starts) >= 1
        assert len(tool_ends) >= 1
        assert len(tool_results) >= 1

        await agent_session.stop()


class TestAgentSessionStateChanges:
    """Test state change events."""

    @pytest.mark.asyncio
    async def test_state_transitions(self, session_setup):
        agent_session, provider, _, _ = session_setup
        provider.add_response(Message(role=Role.ASSISTANT, content="OK"))

        agent_session.start()
        # Wait for IDLE state
        await asyncio.sleep(0.05)
        assert agent_session.state == AgentState.IDLE

        agent_session.push("Go")
        events = await _collect_events(agent_session.event_queue)

        state_changes = [
            e
            for e in events
            if isinstance(e, StreamEvent) and e.type == StreamEventType.STATE_CHANGE
        ]
        # Should have at least a THINKING state change
        states = [e.content for e in state_changes]
        assert AgentState.THINKING.value in states

        await agent_session.stop()


class TestAgentSessionInterrupt:
    """Test interrupt capability."""

    @pytest.mark.asyncio
    async def test_interrupt_sets_state(self, session_setup):
        agent_session, provider, _, _ = session_setup

        # Set up a slow response (many tool rounds)
        from leuk.types import ToolCall

        tc = ToolCall(id="call_slow", name="shell", arguments={"command": "sleep 10"})
        for _ in range(10):
            provider.add_response(
                Message(role=Role.ASSISTANT, content="Working...", tool_calls=[tc])
            )

        agent_session.start()
        agent_session.push("Do something slow")

        # Wait a bit for the agent to start working
        await asyncio.sleep(0.1)

        # Interrupt
        agent_session.interrupt()

        # Collect remaining events
        events = await _collect_events(agent_session.event_queue, timeout=2.0)

        # Should get a TURN_COMPLETE eventually
        turn_complete = [
            e
            for e in events
            if isinstance(e, StreamEvent) and e.type == StreamEventType.TURN_COMPLETE
        ]
        assert len(turn_complete) == 1

        await agent_session.stop()


class TestAgentSessionQueueRendering:
    """Test that render_queue works with AgentSession events."""

    @pytest.mark.asyncio
    async def test_render_queue_simple(self, session_setup, capsys):
        from rich.console import Console
        from leuk.cli.render import StreamRenderer

        agent_session, provider, _, _ = session_setup
        provider.add_response(Message(role=Role.ASSISTANT, content="Rendered!"))

        con = Console(file=open("/dev/null", "w"), force_terminal=True)
        renderer = StreamRenderer(con)

        agent_session.start()
        agent_session.push("Render me")

        await renderer.render_queue(agent_session.event_queue)

        # Text buffer should contain the response
        assert renderer._text_buffer  # not empty
        text = "".join(renderer._text_buffer)
        assert "Rendered" in text

        await agent_session.stop()
