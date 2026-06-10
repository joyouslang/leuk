"""Tests for the history tool (full-conversation navigation after compaction)."""

from __future__ import annotations

import pytest

from leuk.tools.history import HistoryTool
from leuk.types import Message, Role, ToolCall, ToolResult


def _convo() -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content="system prompt"),
        Message(role=Role.USER, content="please fix the websocket reconnect bug"),
        Message(
            role=Role.ASSISTANT,
            content="Looking into it.",
            tool_calls=[ToolCall(id="t1", name="shell", arguments={"command": "grep -r reconnect"})],
        ),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id="t1", name="shell", content="src/ws.py:42 reconnect()"),
        ),
        Message(role=Role.ASSISTANT, content="Fixed: the backoff was never reset."),
    ]


def _tool(messages: list[Message] | None = None) -> HistoryTool:
    t = HistoryTool()
    msgs = _convo() if messages is None else messages

    async def _source() -> list[Message]:
        return msgs

    t.set_source(_source)
    return t


class TestSearch:
    @pytest.mark.asyncio
    async def test_finds_matches_with_indices(self):
        out = await _tool().execute({"action": "search", "query": "reconnect"})
        assert "#1" in out  # the user message
        assert "#2" in out  # the tool call arguments
        assert "#3" in out  # the tool result
        assert "match(es)" in out

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        out = await _tool().execute({"action": "search", "query": "BACKOFF"})
        assert "#4" in out

    @pytest.mark.asyncio
    async def test_no_match(self):
        out = await _tool().execute({"action": "search", "query": "kubernetes"})
        assert "No messages match" in out

    @pytest.mark.asyncio
    async def test_empty_query_errors(self):
        out = await _tool().execute({"action": "search", "query": "  "})
        assert "[ERROR]" in out


class TestRead:
    @pytest.mark.asyncio
    async def test_reads_range_with_headers(self):
        out = await _tool().execute({"action": "read", "start": 1, "count": 2})
        assert "websocket reconnect bug" in out
        assert "#1 [user" in out and "#2 [assistant" in out
        assert "#3" not in out  # range respected

    @pytest.mark.asyncio
    async def test_long_content_truncated(self):
        msgs = [Message(role=Role.USER, content="x" * 5000)]
        out = await _tool(msgs).execute({"action": "read", "start": 0, "count": 1})
        assert "+3000 chars" in out

    @pytest.mark.asyncio
    async def test_out_of_range_clamped(self):
        out = await _tool().execute({"action": "read", "start": 999, "count": 5})
        assert "backoff" in out  # clamps to the last message


class TestWiring:
    @pytest.mark.asyncio
    async def test_unwired_tool_errors_cleanly(self):
        out = await HistoryTool().execute({"action": "search", "query": "x"})
        assert "[ERROR]" in out

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        out = await _tool().execute({"action": "delete"})
        assert "[ERROR]" in out

    def test_registered_by_default(self):
        from leuk.tools import create_default_registry

        reg = create_default_registry()
        assert reg.get("history") is not None


class TestCompactionNotes:
    def test_emergency_drop_mentions_history_tool(self):
        from leuk.agent.context import _emergency_drop

        big = [Message(role=Role.USER, content="word " * 4000) for _ in range(6)]
        out = _emergency_drop([], big, max_tokens=2000)
        notes = [m for m in out if m.content and "history" in (m.content or "")]
        assert notes, "drop note must point at the history tool"
