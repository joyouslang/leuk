"""SQLite-backed durable store."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from leuk.config import SQLiteConfig
from leuk.types import MediaPart, Message, Role, Session, SessionStatus, ToolCall, ToolResult


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

    async def list_sessions(
        self, *, limit: int = 20, top_level_only: bool = False
    ) -> list[Session]:
        """List sessions, most recently updated first.

        When *top_level_only* is set, sub-agent sessions (those with a
        ``parent_session_id``) are excluded — the REPL uses this so spawned
        worker sessions don't clutter ``/sessions``.
        """
        where = "WHERE parent_session_id IS NULL " if top_level_only else ""
        cursor = await self.db.execute(
            f"SELECT * FROM sessions {where}ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        return [_row_to_session(row) for row in await cursor.fetchall()]

    async def list_child_sessions(
        self, parent_id: str | None = None, *, limit: int = 50
    ) -> list[Session]:
        """List sub-agent (child) sessions, most recently updated first.

        With *parent_id*, only that session's direct children are returned;
        otherwise every session that has a ``parent_session_id`` is returned.
        """
        if parent_id is None:
            cursor = await self.db.execute(
                "SELECT * FROM sessions WHERE parent_session_id IS NOT NULL "
                "ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM sessions WHERE parent_session_id = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (parent_id, limit),
            )
        return [_row_to_session(row) for row in await cursor.fetchall()]

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and its messages, cascading to sub-agent children.

        Sub-agent sessions are kept (archived) after they finish so the user
        can inspect them; deleting their parent removes them too. The cascade
        is recursive in case a sub-agent spawned its own sub-agents.
        """
        cursor = await self.db.execute(
            "SELECT id FROM sessions WHERE parent_session_id = ?", (session_id,)
        )
        child_ids = [row["id"] for row in await cursor.fetchall()]
        for child_id in child_ids:
            await self.delete_session(child_id)

        await self.db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await self.db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await self.db.commit()

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    async def append_message(self, session_id: str, message: Message) -> None:
        meta = dict(message.metadata)
        if message.attachments:
            meta["_attachments"] = [a.to_dict() for a in message.attachments]
        if message.thinking:
            meta["_thinking"] = message.thinking
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
                json.dumps(meta),
            ),
        )
        await self.db.commit()

    async def get_messages(self, session_id: str) -> list[Message]:
        cursor = await self.db.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY rowid ASC",
            (session_id,),
        )
        return [_row_to_message(row) for row in await cursor.fetchall()]

    async def delete_last_exchange(self, session_id: str) -> int:
        """Delete the last user message and everything after it (used by /undo).

        ``[SYSTEM]`` housekeeping messages don't count as the exchange start.
        Returns the number of rows deleted (0 when there is no user turn).
        """
        cursor = await self.db.execute(
            """DELETE FROM messages WHERE session_id = ? AND rowid >= (
                   SELECT MAX(rowid) FROM messages
                   WHERE session_id = ? AND role = 'user'
                     AND content IS NOT NULL AND content != ''
                     AND content NOT LIKE '[SYSTEM]%'
               )""",
            (session_id, session_id),
        )
        await self.db.commit()
        return cursor.rowcount or 0

    # ------------------------------------------------------------------
    # Tool approvals
    # ------------------------------------------------------------------

    async def add_tool_approval(
        self,
        tool: str,
        pattern: str,
        action: str = "allow",
        created_by: str = "",
    ) -> None:
        """Insert or replace a persistent tool approval rule."""
        await self.db.execute(
            """INSERT OR REPLACE INTO tool_approvals (tool, pattern, action, created_by, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (tool, pattern, action, created_by, datetime.now(timezone.utc).isoformat()),
        )
        await self.db.commit()

    async def list_tool_approvals(self) -> list[dict]:
        """Return all persistent tool approval rules."""
        cursor = await self.db.execute(
            "SELECT id, tool, pattern, action, created_by, created_at "
            "FROM tool_approvals ORDER BY id ASC"
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def remove_tool_approval(self, approval_id: int) -> None:
        """Delete a single tool approval by its ID."""
        await self.db.execute("DELETE FROM tool_approvals WHERE id = ?", (approval_id,))
        await self.db.commit()

    async def clear_tool_approvals(self) -> int:
        """Delete all tool approval rules. Returns count deleted."""
        cursor = await self.db.execute("SELECT COUNT(*) FROM tool_approvals")
        row = await cursor.fetchone()
        count = row[0] if row else 0
        await self.db.execute("DELETE FROM tool_approvals")
        await self.db.commit()
        return count

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

CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    schedule_type TEXT NOT NULL,
    schedule_expr TEXT NOT NULL,
    pre_check_script TEXT NOT NULL DEFAULT '',
    context_mode TEXT NOT NULL DEFAULT 'fresh',
    enabled INTEGER NOT NULL DEFAULT 1,
    last_run TEXT,
    next_run TEXT,
    session_id TEXT
);

CREATE TABLE IF NOT EXISTS task_run_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    session_id TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    success INTEGER NOT NULL DEFAULT 0,
    summary TEXT,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_task_run_logs_task ON task_run_logs(task_id);

CREATE TABLE IF NOT EXISTS tool_approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool TEXT NOT NULL,
    pattern TEXT NOT NULL,
    action TEXT NOT NULL DEFAULT 'allow',
    created_by TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(tool, pattern)
);
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

    meta = json.loads(row["metadata"])
    raw_atts = meta.pop("_attachments", None)
    attachments = (
        [MediaPart.from_dict(a) for a in raw_atts] if isinstance(raw_atts, list) else None
    )
    thinking = meta.pop("_thinking", None)

    return Message(
        role=Role(row["role"]),
        content=row["content"],
        tool_calls=tool_calls,
        tool_result=tool_result,
        timestamp=datetime.fromisoformat(row["timestamp"]),
        metadata=meta,
        attachments=attachments,
        thinking=thinking if isinstance(thinking, str) else None,
    )
