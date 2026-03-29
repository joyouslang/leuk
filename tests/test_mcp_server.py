"""Tests for the MCP server (src/leuk/mcp/server.py)."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from leuk.config import SQLiteConfig
from leuk.mcp.server import LeukMCPServer
from leuk.persistence.memory import MemoryStore
from leuk.persistence.sqlite import SQLiteStore
from leuk.types import Message, Role, Session, SessionStatus


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def sqlite_store(tmp_path: Path) -> AsyncIterator[SQLiteStore]:
    config = SQLiteConfig(path=str(tmp_path / "mcp_test.db"))
    store = SQLiteStore(config)
    await store.init()
    yield store
    await store.close()


@pytest.fixture
def hot_store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def sessions() -> dict:
    return {}


@pytest_asyncio.fixture
async def server(
    sqlite_store: SQLiteStore,
    hot_store: MemoryStore,
    sessions: dict,
) -> LeukMCPServer:
    return LeukMCPServer(sqlite_store, hot_store, sessions)


# ── helper: call an mcp tool by name ────────────────────────────────────────


async def call_tool(server: LeukMCPServer, name: str, **kwargs):
    """Invoke a registered FastMCP tool by name and return the Python result.

    FastMCP.call_tool returns ``(content_list, meta)`` where:
    - list-returning tools → ``meta = {"result": [...]}``
    - dict-returning tools → ``meta = {the_dict_directly}``
    """
    _, meta = await server.mcp.call_tool(name, kwargs)
    # Unwrap list results (meta has a single "result" key)
    if tuple(meta.keys()) == ("result",):
        return meta["result"]
    return meta


# ── list_sessions ────────────────────────────────────────────────────────────


class TestListSessions:
    async def test_empty(self, server: LeukMCPServer):
        result = await call_tool(server, "list_sessions")
        assert result == []

    async def test_returns_sessions(
        self, server: LeukMCPServer, sqlite_store: SQLiteStore
    ):
        s1 = Session(system_prompt="hello")
        s2 = Session(system_prompt="world")
        await sqlite_store.create_session(s1)
        await sqlite_store.create_session(s2)

        result = await call_tool(server, "list_sessions")
        assert len(result) == 2
        ids = {r["session_id"] for r in result}
        assert s1.id in ids
        assert s2.id in ids

    async def test_fields(self, server: LeukMCPServer, sqlite_store: SQLiteStore):
        s = Session(system_prompt="test prompt")
        await sqlite_store.create_session(s)

        result = await call_tool(server, "list_sessions")
        assert len(result) == 1
        row = result[0]
        assert row["session_id"] == s.id
        assert row["status"] == SessionStatus.ACTIVE.value
        assert row["system_prompt"] == "test prompt"
        assert row["is_running"] is False

    async def test_is_running_flag(
        self, server: LeukMCPServer, sqlite_store: SQLiteStore, sessions: dict
    ):
        s = Session()
        await sqlite_store.create_session(s)
        sessions[s.id] = MagicMock()  # simulate a running AgentSession

        result = await call_tool(server, "list_sessions")
        assert result[0]["is_running"] is True

    async def test_limit(self, server: LeukMCPServer, sqlite_store: SQLiteStore):
        for _ in range(5):
            await sqlite_store.create_session(Session())

        result = await call_tool(server, "list_sessions", limit=2)
        assert len(result) == 2

    async def test_system_prompt_truncated(
        self, server: LeukMCPServer, sqlite_store: SQLiteStore
    ):
        long_prompt = "x" * 500
        s = Session(system_prompt=long_prompt)
        await sqlite_store.create_session(s)

        result = await call_tool(server, "list_sessions")
        assert len(result[0]["system_prompt"]) == 200


# ── get_history ──────────────────────────────────────────────────────────────


class TestGetHistory:
    async def test_unknown_session(self, server: LeukMCPServer):
        result = await call_tool(server, "get_history", session_id="doesnotexist")
        assert result[0]["error"].startswith("Session")

    async def test_empty_history(
        self, server: LeukMCPServer, sqlite_store: SQLiteStore
    ):
        s = Session()
        await sqlite_store.create_session(s)

        result = await call_tool(server, "get_history", session_id=s.id)
        assert result == []

    async def test_returns_messages(
        self, server: LeukMCPServer, sqlite_store: SQLiteStore
    ):
        s = Session()
        await sqlite_store.create_session(s)
        await sqlite_store.append_message(s.id, Message(role=Role.USER, content="hi"))
        await sqlite_store.append_message(
            s.id, Message(role=Role.ASSISTANT, content="hello")
        )

        result = await call_tool(server, "get_history", session_id=s.id)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hi"
        assert result[1]["role"] == "assistant"

    async def test_limit(self, server: LeukMCPServer, sqlite_store: SQLiteStore):
        s = Session()
        await sqlite_store.create_session(s)
        for i in range(10):
            await sqlite_store.append_message(
                s.id, Message(role=Role.USER, content=f"msg {i}")
            )

        result = await call_tool(server, "get_history", session_id=s.id, limit=3)
        assert len(result) == 3
        # Should be the last 3 messages
        assert result[-1]["content"] == "msg 9"


# ── send_message ─────────────────────────────────────────────────────────────


class TestSendMessage:
    async def test_unknown_session(self, server: LeukMCPServer):
        result = await call_tool(
            server, "send_message", session_id="nope", message="hello"
        )
        assert result["success"] is False
        assert "not currently running" in result["error"]

    async def test_injects_message(
        self, server: LeukMCPServer, sqlite_store: SQLiteStore, sessions: dict
    ):
        s = Session()
        await sqlite_store.create_session(s)

        mock_agent_session = MagicMock()
        sessions[s.id] = mock_agent_session

        result = await call_tool(
            server, "send_message", session_id=s.id, message="do something"
        )
        assert result["success"] is True
        assert result["session_id"] == s.id
        mock_agent_session.push.assert_called_once_with("do something")


# ── create_session ────────────────────────────────────────────────────────────


class TestCreateSession:
    async def test_creates_record(
        self, server: LeukMCPServer, sqlite_store: SQLiteStore
    ):
        result = await call_tool(server, "create_session")
        assert "session_id" in result
        assert result["status"] == SessionStatus.ACTIVE.value

        # Verify persisted
        loaded = await sqlite_store.get_session(result["session_id"])
        assert loaded is not None

    async def test_with_system_prompt(
        self, server: LeukMCPServer, sqlite_store: SQLiteStore
    ):
        result = await call_tool(
            server, "create_session", system_prompt="custom prompt"
        )
        loaded = await sqlite_store.get_session(result["session_id"])
        assert loaded is not None
        assert loaded.system_prompt == "custom prompt"

    async def test_default_system_prompt(
        self, server: LeukMCPServer, sqlite_store: SQLiteStore
    ):
        result = await call_tool(server, "create_session")
        loaded = await sqlite_store.get_session(result["session_id"])
        assert loaded is not None
        assert loaded.system_prompt == ""

    async def test_unique_ids(self, server: LeukMCPServer):
        r1 = await call_tool(server, "create_session")
        r2 = await call_tool(server, "create_session")
        assert r1["session_id"] != r2["session_id"]
