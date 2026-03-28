"""Tests for the scheduler package."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio

from leuk.config import AgentConfig, LLMConfig, SQLiteConfig, Settings
from leuk.persistence.memory import MemoryStore
from leuk.persistence.sqlite import SQLiteStore
from leuk.scheduler import (
    ScheduledTask,
    SchedulerStore,
    TaskScheduler,
    compute_initial_next_run,
    compute_next_run,
)
from leuk.tools.base import ToolRegistry
from leuk.types import Message, Role


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def sqlite_store(tmp_path: Path) -> AsyncIterator[SQLiteStore]:
    config = SQLiteConfig(path=str(tmp_path / "test_sched.db"))
    store = SQLiteStore(config)
    await store.init()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def sched_store(sqlite_store: SQLiteStore) -> SchedulerStore:
    return SchedulerStore(sqlite_store)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm=LLMConfig(provider="mock"),
        sqlite=SQLiteConfig(path=str(tmp_path / "test_sched.db")),
        agent=AgentConfig(max_tool_rounds=5),
    )


_NOW = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ── ScheduledTask ──────────────────────────────────────────────────────


class TestScheduledTask:
    def test_defaults(self):
        task = ScheduledTask(
            name="daily-report",
            prompt="Summarise logs",
            schedule_type="cron",
            schedule_expr="0 9 * * *",
        )
        assert task.enabled is True
        assert task.context_mode == "fresh"
        assert task.pre_check_script == ""
        assert task.last_run is None
        assert task.next_run is None
        assert task.session_id is None
        assert len(task.id) == 36  # UUID4

    def test_unique_ids(self):
        t1 = ScheduledTask(name="a", prompt="p", schedule_type="once", schedule_expr="2025-01-01T00:00:00")
        t2 = ScheduledTask(name="b", prompt="p", schedule_type="once", schedule_expr="2025-01-01T00:00:00")
        assert t1.id != t2.id


# ── compute_next_run ──────────────────────────────────────────────────


class TestComputeNextRun:
    def test_interval(self):
        task = ScheduledTask(
            name="t", prompt="p", schedule_type="interval", schedule_expr="3600"
        )
        nxt = compute_next_run(task, _NOW)
        assert nxt == _NOW + timedelta(seconds=3600)

    def test_once_returns_none(self):
        task = ScheduledTask(
            name="t", prompt="p", schedule_type="once", schedule_expr="2025-01-01T12:00:00"
        )
        assert compute_next_run(task, _NOW) is None

    def test_cron(self):
        pytest.importorskip("croniter")
        task = ScheduledTask(
            name="t", prompt="p", schedule_type="cron", schedule_expr="0 10 * * *"
        )
        # After noon on 2025-01-01, next 10:00 is 2025-01-02
        nxt = compute_next_run(task, _NOW)
        assert nxt is not None
        assert nxt.hour == 10
        assert nxt.day == 2

    def test_unknown_type_raises(self):
        task = ScheduledTask(
            name="t", prompt="p", schedule_type="once", schedule_expr="x"
        )
        task.schedule_type = "bogus"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown schedule_type"):
            compute_next_run(task, _NOW)

    def test_compute_initial_once(self):
        future = datetime(2099, 6, 1, 9, 0, 0, tzinfo=timezone.utc)
        task = ScheduledTask(
            name="t", prompt="p", schedule_type="once",
            schedule_expr=future.isoformat()
        )
        nxt = compute_initial_next_run(task)
        assert nxt == future

    def test_compute_initial_interval(self):
        task = ScheduledTask(
            name="t", prompt="p", schedule_type="interval", schedule_expr="60"
        )
        nxt = compute_initial_next_run(task)
        assert nxt is not None
        assert nxt > datetime.now(timezone.utc)


# ── SchedulerStore ────────────────────────────────────────────────────


class TestSchedulerStore:
    @pytest.mark.asyncio
    async def test_create_and_get(self, sched_store: SchedulerStore):
        task = ScheduledTask(
            name="my-task",
            prompt="Do something",
            schedule_type="interval",
            schedule_expr="300",
            next_run=_NOW,
        )
        await sched_store.create_task(task)

        loaded = await sched_store.get_task(task.id)
        assert loaded is not None
        assert loaded.name == "my-task"
        assert loaded.prompt == "Do something"
        assert loaded.schedule_type == "interval"
        assert loaded.schedule_expr == "300"
        assert loaded.enabled is True
        assert loaded.next_run is not None

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, sched_store: SchedulerStore):
        assert await sched_store.get_task("no-such-id") is None

    @pytest.mark.asyncio
    async def test_list(self, sched_store: SchedulerStore):
        for i in range(3):
            await sched_store.create_task(
                ScheduledTask(name=f"task-{i}", prompt="p", schedule_type="interval", schedule_expr="60")
            )
        tasks = await sched_store.list_tasks()
        assert len(tasks) == 3

    @pytest.mark.asyncio
    async def test_update(self, sched_store: SchedulerStore):
        task = ScheduledTask(
            name="old-name", prompt="p", schedule_type="interval", schedule_expr="60"
        )
        await sched_store.create_task(task)
        task.name = "new-name"
        task.enabled = False
        await sched_store.update_task(task)

        loaded = await sched_store.get_task(task.id)
        assert loaded is not None
        assert loaded.name == "new-name"
        assert loaded.enabled is False

    @pytest.mark.asyncio
    async def test_delete(self, sched_store: SchedulerStore):
        task = ScheduledTask(name="t", prompt="p", schedule_type="interval", schedule_expr="60")
        await sched_store.create_task(task)
        await sched_store.delete_task(task.id)
        assert await sched_store.get_task(task.id) is None

    @pytest.mark.asyncio
    async def test_get_due_tasks(self, sched_store: SchedulerStore):
        past = _NOW - timedelta(hours=1)
        future = _NOW + timedelta(hours=1)

        due_task = ScheduledTask(
            name="due", prompt="p", schedule_type="interval", schedule_expr="60",
            next_run=past
        )
        not_due_task = ScheduledTask(
            name="not-due", prompt="p", schedule_type="interval", schedule_expr="60",
            next_run=future
        )
        disabled_task = ScheduledTask(
            name="disabled", prompt="p", schedule_type="interval", schedule_expr="60",
            next_run=past, enabled=False
        )

        for t in [due_task, not_due_task, disabled_task]:
            await sched_store.create_task(t)

        due = await sched_store.get_due_tasks(_NOW)
        names = {t.name for t in due}
        assert "due" in names
        assert "not-due" not in names
        assert "disabled" not in names

    @pytest.mark.asyncio
    async def test_log_run(self, sched_store: SchedulerStore):
        task = ScheduledTask(name="t", prompt="p", schedule_type="interval", schedule_expr="60")
        await sched_store.create_task(task)

        now = datetime.now(timezone.utc)
        await sched_store.log_run(
            task_id=task.id,
            session_id="sess-123",
            started_at=now,
            finished_at=now,
            success=True,
            summary="All done",
        )

        logs = await sched_store.get_run_logs(task.id)
        assert len(logs) == 1
        assert logs[0]["success"] == 1
        assert logs[0]["summary"] == "All done"
        assert logs[0]["session_id"] == "sess-123"

    @pytest.mark.asyncio
    async def test_log_run_failure(self, sched_store: SchedulerStore):
        task = ScheduledTask(name="t", prompt="p", schedule_type="interval", schedule_expr="60")
        await sched_store.create_task(task)

        now = datetime.now(timezone.utc)
        await sched_store.log_run(
            task_id=task.id,
            session_id=None,
            started_at=now,
            finished_at=now,
            success=False,
            error="something went wrong",
        )

        logs = await sched_store.get_run_logs(task.id)
        assert logs[0]["success"] == 0
        assert logs[0]["error"] == "something went wrong"

    @pytest.mark.asyncio
    async def test_task_with_nulls(self, sched_store: SchedulerStore):
        """Tasks with no last_run/next_run/session_id round-trip correctly."""
        task = ScheduledTask(
            name="t", prompt="p", schedule_type="interval", schedule_expr="60"
        )
        await sched_store.create_task(task)
        loaded = await sched_store.get_task(task.id)
        assert loaded is not None
        assert loaded.last_run is None
        assert loaded.next_run is None
        assert loaded.session_id is None


# ── TaskScheduler ─────────────────────────────────────────────────────


class TestTaskSchedulerLifecycle:
    """Test that the scheduler starts/stops cleanly without actually running tasks."""

    @pytest.mark.asyncio
    async def test_start_stop(self, sqlite_store: SQLiteStore, settings: Settings):
        from tests.conftest import MockProvider

        scheduler = TaskScheduler(
            settings=settings,
            sqlite=sqlite_store,
            hot_store=MemoryStore(),
            provider=MockProvider(),
            tool_registry=ToolRegistry(),
        )
        await scheduler.start()
        assert scheduler._task is not None
        assert not scheduler._task.done()

        await scheduler.stop()
        assert scheduler._task is None

    @pytest.mark.asyncio
    async def test_stop_without_start(self, sqlite_store: SQLiteStore, settings: Settings):
        from tests.conftest import MockProvider

        scheduler = TaskScheduler(
            settings=settings,
            sqlite=sqlite_store,
            hot_store=MemoryStore(),
            provider=MockProvider(),
            tool_registry=ToolRegistry(),
        )
        # Should not raise
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self, sqlite_store: SQLiteStore, settings: Settings):
        from tests.conftest import MockProvider

        scheduler = TaskScheduler(
            settings=settings,
            sqlite=sqlite_store,
            hot_store=MemoryStore(),
            provider=MockProvider(),
            tool_registry=ToolRegistry(),
        )
        await scheduler.start()
        task_ref = scheduler._task
        await scheduler.start()  # should not spawn a second task
        assert scheduler._task is task_ref
        await scheduler.stop()


class TestTaskSchedulerExecution:
    """Integration-style tests: verify a due task gets run and logs are written."""

    @pytest.mark.asyncio
    async def test_interval_task_runs(self, sqlite_store: SQLiteStore, settings: Settings):
        from tests.conftest import MockProvider

        provider = MockProvider(
            responses=[Message(role=Role.ASSISTANT, content="Task done")]
        )

        scheduler = TaskScheduler(
            settings=settings,
            sqlite=sqlite_store,
            hot_store=MemoryStore(),
            provider=provider,
            tool_registry=ToolRegistry(),
        )

        # Create a task that was due 1 minute ago
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        task = ScheduledTask(
            name="test-run",
            prompt="Hello agent",
            schedule_type="interval",
            schedule_expr="3600",
            next_run=past,
        )
        await scheduler.store.create_task(task)

        # Run a single poll (no loop)
        await scheduler._poll_once()

        # Give background task time to execute
        await asyncio.sleep(0.2)

        # The task's last_run should now be set and next_run advanced
        updated = await scheduler.store.get_task(task.id)
        assert updated is not None
        assert updated.last_run is not None
        assert updated.next_run is not None
        assert updated.next_run > past

        # A run log entry should exist
        logs = await scheduler.store.get_run_logs(task.id)
        assert len(logs) == 1
        assert logs[0]["success"] == 1

    @pytest.mark.asyncio
    async def test_once_task_disabled_after_run(self, sqlite_store: SQLiteStore, settings: Settings):
        from tests.conftest import MockProvider

        provider = MockProvider(
            responses=[Message(role=Role.ASSISTANT, content="Done once")]
        )

        scheduler = TaskScheduler(
            settings=settings,
            sqlite=sqlite_store,
            hot_store=MemoryStore(),
            provider=provider,
            tool_registry=ToolRegistry(),
        )

        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        task = ScheduledTask(
            name="once-task",
            prompt="Run once",
            schedule_type="once",
            schedule_expr=past.isoformat(),
            next_run=past,
        )
        await scheduler.store.create_task(task)
        await scheduler._poll_once()
        await asyncio.sleep(0.2)

        updated = await scheduler.store.get_task(task.id)
        assert updated is not None
        assert updated.enabled is False
        assert updated.next_run is None
