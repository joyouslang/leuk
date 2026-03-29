"""TaskScheduler: background poll loop that executes scheduled agent tasks."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from leuk.agent.core import Agent
from leuk.config import Settings
from leuk.persistence.base import HotStore
from leuk.persistence.sqlite import SQLiteStore
from leuk.providers.base import LLMProvider
from leuk.safety import SafetyGuard
from leuk.scheduler.store import SchedulerStore
from leuk.scheduler.task import ScheduledTask
from leuk.tools.base import ToolRegistry
from leuk.types import Role, Session

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Polls for due :class:`~leuk.scheduler.task.ScheduledTask` objects and runs them.

    Each task spawns a real :class:`~leuk.agent.core.Agent` session, so runs appear
    in ``/sessions`` just like interactive sessions.

    Usage::

        scheduler = TaskScheduler(settings, sqlite, hot_store, provider, tools)
        await scheduler.start()
        # ... later ...
        await scheduler.stop()
    """

    def __init__(
        self,
        settings: Settings,
        sqlite: SQLiteStore,
        hot_store: HotStore,
        provider: LLMProvider,
        tool_registry: ToolRegistry,
        safety_guard: SafetyGuard | None = None,
    ) -> None:
        self._settings = settings
        self._sqlite = sqlite
        self._hot_store = hot_store
        self._provider = provider
        self._tools = tool_registry
        self._safety_guard = safety_guard
        self._store = SchedulerStore(sqlite)
        self._poll_interval = settings.scheduler.poll_interval
        self._task: asyncio.Task[None] | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background poll loop."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._poll_loop(), name="scheduler-poll")
        logger.info("TaskScheduler started (poll interval: %ds)", self._poll_interval)

    async def stop(self) -> None:
        """Cancel the poll loop and wait for it to exit."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("TaskScheduler stopped")

    # ── Store delegation ──────────────────────────────────────────────

    @property
    def store(self) -> SchedulerStore:
        """Direct access to the underlying :class:`SchedulerStore`."""
        return self._store

    # ── Internal ──────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduler poll error")
            await asyncio.sleep(self._poll_interval)

    async def _poll_once(self) -> None:
        now = datetime.now(timezone.utc)
        due = await self._store.get_due_tasks(now)
        for task in due:
            asyncio.create_task(
                self._run_task(task), name=f"scheduler-task-{task.id[:8]}"
            )

    async def _run_task(self, task: ScheduledTask) -> None:
        started_at = datetime.now(timezone.utc)
        session_id: str | None = None
        success = False
        summary: str | None = None
        error: str | None = None

        try:
            # Pre-check script
            if task.pre_check_script:
                ok = await _run_pre_check(task.pre_check_script)
                if not ok:
                    logger.info(
                        "Skipping task %r — pre-check script returned non-zero", task.name
                    )
                    # Advance next_run so we don't re-check immediately
                    task.next_run = compute_next_run(task, started_at)
                    task.last_run = started_at
                    await self._store.update_task(task)
                    return

            # Build session
            if task.context_mode == "resume" and task.session_id:
                session = Session(id=task.session_id, system_prompt=self._settings.agent.system_prompt)
            else:
                session = Session(system_prompt=self._settings.agent.system_prompt)

            session.metadata["scheduled_task_id"] = task.id
            session.metadata["scheduled_task_name"] = task.name

            agent = Agent(
                settings=self._settings,
                provider=self._provider,
                tool_registry=self._tools,
                sqlite=self._sqlite,
                hot_store=self._hot_store,
                session=session,
                safety_guard=self._safety_guard,
            )
            await agent.init()
            session_id = agent.session.id

            # Collect output
            text_parts: list[str] = []
            async for msg in agent.run(task.prompt):
                if msg.role == Role.ASSISTANT and msg.content:
                    text_parts.append(msg.content)

            await agent.shutdown()

            summary = "\n\n".join(text_parts) if text_parts else None
            success = True

            # Persist session_id for resume mode
            if task.context_mode == "resume":
                task.session_id = session_id

            logger.info(
                "Task %r completed successfully (session %s)", task.name, session_id
            )

        except Exception as exc:
            error = str(exc)
            logger.exception("Task %r failed", task.name)

        finally:
            finished_at = datetime.now(timezone.utc)

            # Update task state
            task.last_run = started_at
            task.next_run = compute_next_run(task, started_at)
            if task.schedule_type == "once":
                task.enabled = False

            try:
                await self._store.update_task(task)
                await self._store.log_run(
                    task_id=task.id,
                    session_id=session_id,
                    started_at=started_at,
                    finished_at=finished_at,
                    success=success,
                    summary=summary,
                    error=error,
                )
            except Exception:
                logger.exception("Failed to persist run log for task %r", task.name)


# ======================================================================
# Scheduling helpers
# ======================================================================


def compute_next_run(task: ScheduledTask, after: datetime) -> datetime | None:
    """Return the next scheduled run time for *task* after *after*.

    Returns ``None`` for ``"once"`` tasks (they are disabled after their first run).
    """
    if task.schedule_type == "once":
        return None

    if task.schedule_type == "interval":
        seconds = int(task.schedule_expr)
        from datetime import timedelta
        return after + timedelta(seconds=seconds)

    if task.schedule_type == "cron":
        try:
            from croniter import croniter  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "croniter is required for cron-type tasks. "
                "Install it with: pip install croniter"
            ) from exc
        it = croniter(task.schedule_expr, after)
        return it.get_next(datetime).replace(tzinfo=timezone.utc)

    raise ValueError(f"Unknown schedule_type: {task.schedule_type!r}")


def compute_initial_next_run(task: ScheduledTask) -> datetime | None:
    """Compute the first ``next_run`` for a newly created task."""
    now = datetime.now(timezone.utc)

    if task.schedule_type == "once":
        dt = datetime.fromisoformat(task.schedule_expr)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    # For cron/interval, compute first run from now
    return compute_next_run(task, now)


async def _run_pre_check(script: str) -> bool:
    """Run *script* in a shell. Returns True iff exit code is 0."""
    try:
        proc = await asyncio.create_subprocess_shell(
            script,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return proc.returncode == 0
    except Exception:
        logger.exception("Pre-check script failed to execute: %r", script)
        return False
