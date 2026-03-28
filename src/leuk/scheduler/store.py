"""SQLite CRUD for scheduled_tasks and task_run_logs."""

from __future__ import annotations

from datetime import datetime, timezone

import aiosqlite

from leuk.persistence.sqlite import SQLiteStore
from leuk.scheduler.task import ScheduledTask


class SchedulerStore:
    """CRUD wrapper around the scheduled_tasks and task_run_logs tables.

    Uses the same database connection owned by a :class:`~leuk.persistence.sqlite.SQLiteStore`.
    The caller is responsible for calling ``SQLiteStore.init()`` before using this class.
    """

    def __init__(self, sqlite: SQLiteStore) -> None:
        self._sqlite = sqlite

    @property
    def _db(self) -> aiosqlite.Connection:
        return self._sqlite.db

    # ------------------------------------------------------------------
    # scheduled_tasks
    # ------------------------------------------------------------------

    async def create_task(self, task: ScheduledTask) -> None:
        await self._db.execute(
            """INSERT INTO scheduled_tasks
               (id, name, prompt, schedule_type, schedule_expr,
                pre_check_script, context_mode, enabled, last_run, next_run, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task.id,
                task.name,
                task.prompt,
                task.schedule_type,
                task.schedule_expr,
                task.pre_check_script,
                task.context_mode,
                1 if task.enabled else 0,
                task.last_run.isoformat() if task.last_run else None,
                task.next_run.isoformat() if task.next_run else None,
                task.session_id,
            ),
        )
        await self._db.commit()

    async def get_task(self, task_id: str) -> ScheduledTask | None:
        cursor = await self._db.execute(
            "SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)
        )
        row = await cursor.fetchone()
        return _row_to_task(row) if row else None

    async def list_tasks(self) -> list[ScheduledTask]:
        cursor = await self._db.execute(
            "SELECT * FROM scheduled_tasks ORDER BY name ASC"
        )
        return [_row_to_task(row) for row in await cursor.fetchall()]

    async def update_task(self, task: ScheduledTask) -> None:
        await self._db.execute(
            """UPDATE scheduled_tasks
               SET name=?, prompt=?, schedule_type=?, schedule_expr=?,
                   pre_check_script=?, context_mode=?, enabled=?,
                   last_run=?, next_run=?, session_id=?
               WHERE id=?""",
            (
                task.name,
                task.prompt,
                task.schedule_type,
                task.schedule_expr,
                task.pre_check_script,
                task.context_mode,
                1 if task.enabled else 0,
                task.last_run.isoformat() if task.last_run else None,
                task.next_run.isoformat() if task.next_run else None,
                task.session_id,
                task.id,
            ),
        )
        await self._db.commit()

    async def delete_task(self, task_id: str) -> None:
        await self._db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await self._db.commit()

    async def get_due_tasks(self, now: datetime) -> list[ScheduledTask]:
        """Return enabled tasks whose next_run is at or before *now*."""
        cursor = await self._db.execute(
            """SELECT * FROM scheduled_tasks
               WHERE enabled = 1
                 AND next_run IS NOT NULL
                 AND next_run <= ?""",
            (now.isoformat(),),
        )
        return [_row_to_task(row) for row in await cursor.fetchall()]

    # ------------------------------------------------------------------
    # task_run_logs
    # ------------------------------------------------------------------

    async def log_run(
        self,
        task_id: str,
        session_id: str | None,
        started_at: datetime,
        finished_at: datetime | None,
        success: bool,
        summary: str | None = None,
        error: str | None = None,
    ) -> None:
        await self._db.execute(
            """INSERT INTO task_run_logs
               (task_id, session_id, started_at, finished_at, success, summary, error)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                session_id,
                started_at.isoformat(),
                finished_at.isoformat() if finished_at else None,
                1 if success else 0,
                summary,
                error,
            ),
        )
        await self._db.commit()

    async def get_run_logs(
        self, task_id: str, *, limit: int = 20
    ) -> list[dict]:
        cursor = await self._db.execute(
            """SELECT * FROM task_run_logs WHERE task_id = ?
               ORDER BY started_at DESC LIMIT ?""",
            (task_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


# ======================================================================
# Helpers
# ======================================================================


def _row_to_task(row: aiosqlite.Row) -> ScheduledTask:
    def _dt(val: str | None) -> datetime | None:
        if val is None:
            return None
        dt = datetime.fromisoformat(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    return ScheduledTask(
        id=row["id"],
        name=row["name"],
        prompt=row["prompt"],
        schedule_type=row["schedule_type"],
        schedule_expr=row["schedule_expr"],
        pre_check_script=row["pre_check_script"] or "",
        context_mode=row["context_mode"],
        enabled=bool(row["enabled"]),
        last_run=_dt(row["last_run"]),
        next_run=_dt(row["next_run"]),
        session_id=row["session_id"],
    )
