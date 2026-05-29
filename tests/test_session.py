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


class TestAgentSessionErrorRecovery:
    """Test that the session survives provider errors and exposes /retry state."""

    @pytest.mark.asyncio
    async def test_provider_error_emits_error_and_turn_complete(self, session_setup):
        agent_session, provider, _, _ = session_setup

        async def boom(*args, **kwargs):
            raise TimeoutError("Request timed out or interrupted.")
            yield  # pragma: no cover — make this an async generator

        provider.stream = boom

        agent_session.start()
        agent_session.push("Hello there")

        events = await _collect_events(agent_session.event_queue)

        errors = [
            e for e in events if isinstance(e, StreamEvent) and e.type == StreamEventType.ERROR
        ]
        turn_complete = [
            e
            for e in events
            if isinstance(e, StreamEvent) and e.type == StreamEventType.TURN_COMPLETE
        ]

        assert len(errors) == 1, "ERROR event must be emitted on provider failure"
        assert "TimeoutError" in errors[0].content
        assert len(turn_complete) == 1, "TURN_COMPLETE must fire even after error"

        await agent_session.stop()

    @pytest.mark.asyncio
    async def test_session_survives_error_and_accepts_next_turn(self, session_setup):
        agent_session, provider, _, _ = session_setup

        call_count = {"n": 0}
        original_stream = provider.stream

        async def stream_then_succeed(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ConnectionError("network down")
                yield  # pragma: no cover
            else:
                async for ev in original_stream(*args, **kwargs):
                    yield ev

        provider.stream = stream_then_succeed
        provider.add_response(Message(role=Role.ASSISTANT, content="Recovered"))

        agent_session.start()

        agent_session.push("first")
        await _collect_events(agent_session.event_queue)
        assert agent_session.running, "session must stay alive after a turn-level error"

        agent_session.push("second")
        events = await _collect_events(agent_session.event_queue)
        text_deltas = [
            e for e in events if isinstance(e, StreamEvent) and e.type == StreamEventType.TEXT_DELTA
        ]
        assert len(text_deltas) > 0, "subsequent turns must work after a recovered error"

        await agent_session.stop()

    @pytest.mark.asyncio
    async def test_error_after_tool_call_heals_orphans(self, session_setup):
        """Provider error mid-tool-round must not leave orphan tool_calls in _messages."""
        from leuk.types import ToolCall, ToolResult

        agent_session, provider, _, _ = session_setup
        agent = agent_session.agent

        # Inject a fake interrupted state: an assistant message with a
        # tool_call but no matching tool_result. Mirrors what happens when
        # the user Ctrl-C's during _execute_tool.
        tc = ToolCall(id="orphan_1", name="shell", arguments={"command": "ls"})
        agent._messages.append(
            Message(role=Role.ASSISTANT, content="Looking", tool_calls=[tc])
        )

        await agent._heal_orphaned_tool_calls()

        # The healing pass should append a placeholder ToolResult.
        results = [m for m in agent._messages if m.tool_result and m.tool_result.tool_call_id == "orphan_1"]
        assert len(results) == 1
        assert results[0].tool_result.is_error
        assert "interrupted" in results[0].tool_result.content.lower()

        # Idempotent — a second pass adds nothing.
        await agent._heal_orphaned_tool_calls()
        results2 = [m for m in agent._messages if m.tool_result and m.tool_result.tool_call_id == "orphan_1"]
        assert len(results2) == 1

    @pytest.mark.asyncio
    async def test_last_user_input_tracked(self, session_setup):
        agent_session, provider, _, _ = session_setup
        provider.add_response(Message(role=Role.ASSISTANT, content="ack"))

        assert agent_session.last_user_input is None

        agent_session.start()
        agent_session.push("remember me")
        await _collect_events(agent_session.event_queue)

        assert agent_session.last_user_input == "remember me"

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
