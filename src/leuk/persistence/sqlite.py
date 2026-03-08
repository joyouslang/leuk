"""SQLite-backed durable store."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from leuk.config import SQLiteConfig
from leuk.types import Message, Role, Session, SessionStatus, ToolCall, ToolResult


class SQLiteStore:
    """Durable persistence using SQLite via aiosqlite."""

    def __init__(self, config: SQLiteConfig) -> None:
        self._path = Path(config.path).expanduser()
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "Store not initialised -- call init() first"
        return self._db

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def create_session(self, session: Session) -> None:
        await self.db.execute(
            """INSERT INTO sessions (id, status, created_at, updated_at, system_prompt, metadata, parent_session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session.id,
                session.status.value,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                session.system_prompt,
                json.dumps(session.metadata),
                session.parent_session_id,
            ),
        )
        await self.db.commit()

    async def get_session(self, session_id: str) -> Session | None:
        cursor = await self.db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_session(row)

    async def update_session(self, session: Session) -> None:
        session.updated_at = datetime.now(timezone.utc)
        await self.db.execute(
            """UPDATE sessions SET status=?, updated_at=?, system_prompt=?, metadata=?, parent_session_id=?
               WHERE id=?""",
            (
                session.status.value,
                session.updated_at.isoformat(),
                session.system_prompt,
                json.dumps(session.metadata),
                session.parent_session_id,
                session.id,
            ),
        )
        await self.db.commit()

    async def list_sessions(self, *, limit: int = 20) -> list[Session]:
        cursor = await self.db.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
        )
        return [_row_to_session(row) for row in await cursor.fetchall()]

    async def delete_session(self, session_id: str) -> None:
        await self.db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await self.db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await self.db.commit()

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    async def append_message(self, session_id: str, message: Message) -> None:
        await self.db.execute(
            """INSERT INTO messages (session_id, role, content, tool_calls, tool_result, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                message.role.value,
                message.content,
                json.dumps([_tc_to_dict(tc) for tc in message.tool_calls])
                if message.tool_calls
                else None,
                json.dumps(_tr_to_dict(message.tool_result)) if message.tool_result else None,
                message.timestamp.isoformat(),
                json.dumps(message.metadata),
            ),
        )
        await self.db.commit()

    async def get_messages(self, session_id: str) -> list[Message]:
        cursor = await self.db.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY rowid ASC",
            (session_id,),
        )
        return [_row_to_message(row) for row in await cursor.fetchall()]

    async def close(self) -> None:
        if self._db:
            await self._db.close()


# ======================================================================
# Schema
# ======================================================================

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    system_prompt TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    parent_session_id TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_calls TEXT,
    tool_result TEXT,
    timestamp TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


# ======================================================================
# Conversion helpers
# ======================================================================


def _tc_to_dict(tc: ToolCall) -> dict:
    return {"id": tc.id, "name": tc.name, "arguments": tc.arguments}


def _tr_to_dict(tr: ToolResult) -> dict:
    return {
        "tool_call_id": tr.tool_call_id,
        "name": tr.name,
        "content": tr.content,
        "is_error": tr.is_error,
    }


def _row_to_session(row: aiosqlite.Row) -> Session:
    return Session(
        id=row["id"],
        status=SessionStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        system_prompt=row["system_prompt"],
        metadata=json.loads(row["metadata"]),
        parent_session_id=row["parent_session_id"],
    )


def _row_to_message(row: aiosqlite.Row) -> Message:
    tool_calls = None
    if row["tool_calls"]:
        raw = json.loads(row["tool_calls"])
        tool_calls = [ToolCall(**tc) for tc in raw]

    tool_result = None
    if row["tool_result"]:
        tool_result = ToolResult(**json.loads(row["tool_result"]))

    return Message(
        role=Role(row["role"]),
        content=row["content"],
        tool_calls=tool_calls,
        tool_result=tool_result,
        timestamp=datetime.fromisoformat(row["timestamp"]),
        metadata=json.loads(row["metadata"]),
    )
