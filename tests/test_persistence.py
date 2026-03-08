"""Tests for persistence layer."""

from __future__ import annotations

import pytest

from leuk.persistence.memory import MemoryStore
from leuk.persistence.sqlite import SQLiteStore
from leuk.types import Message, Role, Session, SessionStatus, ToolCall, ToolResult


# ── SQLite store ────────────────────────────────────────────────────


class TestSQLiteStore:
    @pytest.mark.asyncio
    async def test_create_and_get_session(self, sqlite_store: SQLiteStore):
        session = Session(system_prompt="test prompt")
        await sqlite_store.create_session(session)

        loaded = await sqlite_store.get_session(session.id)
        assert loaded is not None
        assert loaded.id == session.id
        assert loaded.system_prompt == "test prompt"
        assert loaded.status == SessionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_update_session(self, sqlite_store: SQLiteStore):
        session = Session()
        await sqlite_store.create_session(session)

        session.status = SessionStatus.COMPLETED
        await sqlite_store.update_session(session)

        loaded = await sqlite_store.get_session(session.id)
        assert loaded is not None
        assert loaded.status == SessionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_list_sessions(self, sqlite_store: SQLiteStore):
        for i in range(5):
            await sqlite_store.create_session(Session(system_prompt=f"session {i}"))

        sessions = await sqlite_store.list_sessions(limit=3)
        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, sqlite_store: SQLiteStore):
        result = await sqlite_store.get_session("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_session(self, sqlite_store: SQLiteStore):
        session = Session(system_prompt="to be deleted")
        await sqlite_store.create_session(session)
        msg = Message(role=Role.USER, content="hello")
        await sqlite_store.append_message(session.id, msg)

        await sqlite_store.delete_session(session.id)
        assert await sqlite_store.get_session(session.id) is None
        assert await sqlite_store.get_messages(session.id) == []

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, sqlite_store: SQLiteStore):
        """Deleting a session that doesn't exist should not raise."""
        await sqlite_store.delete_session("nonexistent")

    @pytest.mark.asyncio
    async def test_session_metadata_name(self, sqlite_store: SQLiteStore):
        """Session names are stored in metadata['name']."""
        session = Session(metadata={"name": "my-project"})
        await sqlite_store.create_session(session)

        loaded = await sqlite_store.get_session(session.id)
        assert loaded is not None
        assert loaded.metadata.get("name") == "my-project"

        # Rename via update
        session.metadata["name"] = "renamed"
        await sqlite_store.update_session(session)
        loaded = await sqlite_store.get_session(session.id)
        assert loaded is not None
        assert loaded.metadata["name"] == "renamed"

    @pytest.mark.asyncio
    async def test_append_and_get_messages(self, sqlite_store: SQLiteStore):
        session = Session()
        await sqlite_store.create_session(session)

        msg1 = Message(role=Role.USER, content="Hello")
        msg2 = Message(role=Role.ASSISTANT, content="Hi there!")
        await sqlite_store.append_message(session.id, msg1)
        await sqlite_store.append_message(session.id, msg2)

        messages = await sqlite_store.get_messages(session.id)
        assert len(messages) == 2
        assert messages[0].role == Role.USER
        assert messages[0].content == "Hello"
        assert messages[1].role == Role.ASSISTANT

    @pytest.mark.asyncio
    async def test_message_with_tool_calls(self, sqlite_store: SQLiteStore):
        session = Session()
        await sqlite_store.create_session(session)

        tc = ToolCall(id="call_1", name="shell", arguments={"command": "ls"})
        msg = Message(role=Role.ASSISTANT, content="Running command.", tool_calls=[tc])
        await sqlite_store.append_message(session.id, msg)

        messages = await sqlite_store.get_messages(session.id)
        assert len(messages) == 1
        assert messages[0].tool_calls is not None
        assert messages[0].tool_calls[0].name == "shell"

    @pytest.mark.asyncio
    async def test_message_with_tool_result(self, sqlite_store: SQLiteStore):
        session = Session()
        await sqlite_store.create_session(session)

        tr = ToolResult(tool_call_id="call_1", name="shell", content="file.txt")
        msg = Message(role=Role.TOOL, tool_result=tr)
        await sqlite_store.append_message(session.id, msg)

        messages = await sqlite_store.get_messages(session.id)
        assert len(messages) == 1
        assert messages[0].tool_result is not None
        assert messages[0].tool_result.content == "file.txt"


# ── Memory store ────────────────────────────────────────────────────


class TestMemoryStore:
    @pytest.mark.asyncio
    async def test_context(self):
        store = MemoryStore()
        await store.set_context("s1", '{"test": true}')
        result = await store.get_context("s1")
        assert result == '{"test": true}'

    @pytest.mark.asyncio
    async def test_context_missing(self):
        store = MemoryStore()
        result = await store.get_context("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_context(self):
        store = MemoryStore()
        await store.set_context("s1", "data")
        await store.delete_context("s1")
        assert await store.get_context("s1") is None

    @pytest.mark.asyncio
    async def test_active_session(self):
        store = MemoryStore()
        await store.set_active_session("abc123")
        assert await store.get_active_session() == "abc123"

    @pytest.mark.asyncio
    async def test_close(self):
        store = MemoryStore()
        await store.set_context("s1", "data")
        await store.close()
        assert await store.get_context("s1") is None
