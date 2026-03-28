# Skill: /add-scheduler

Enable scheduled task execution in leuk — run agent prompts on a cron schedule,
fixed interval, or one-time delay.  Results are stored in SQLite alongside normal
sessions.

---

## Prerequisites

1. `croniter` for cron expression parsing: `uv add croniter`.
2. Phase 1.1 (sub-agent concurrency limits) is strongly recommended: scheduled tasks
   can overlap with interactive sessions.

---

## Step 1 — Add dependency

Edit `pyproject.toml`: add `croniter` to `[project.dependencies]` or create an
optional group:

```toml
[project.optional-dependencies]
scheduler = [
    "croniter>=2.0",
]
```

Run: `uv sync --extra scheduler`

---

## Step 2 — Add config section

File: `src/leuk/config.py`

Add after `MCPServerConfig`:

```python
class SchedulerConfig(BaseSettings):
    """Scheduled task runner configuration."""

    model_config = SettingsConfigDict(env_prefix="LEUK_SCHEDULER_", extra="ignore")

    enabled: bool = Field(default=False, description="Enable the task scheduler")
    poll_interval: int = Field(default=60, description="Seconds between scheduler polls")
```

Add to `Settings`:

```python
scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
```

---

## Step 3 — Create scheduler package

### `src/leuk/scheduler/__init__.py`

```python
"""Scheduled task runner."""
```

### `src/leuk/scheduler/task.py`

```python
"""ScheduledTask dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class ScheduledTask:
    id: str
    name: str
    prompt: str
    schedule_type: Literal["cron", "interval", "once"]
    schedule_expr: str          # cron expr, seconds (interval), or ISO datetime (once)
    context_mode: Literal["fresh", "resume"] = "fresh"
    pre_check_script: str = ""  # shell command; non-zero exit skips this run
    enabled: bool = True
    last_run: datetime | None = None
    next_run: datetime | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
```

### `src/leuk/scheduler/store.py`

CRUD layer over `src/leuk/persistence/sqlite.py`'s `SQLiteStore`.

```python
"""SQLite persistence for scheduled tasks."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leuk.persistence.sqlite import SQLiteStore

from leuk.scheduler.task import ScheduledTask


class SchedulerStore:
    def __init__(self, sqlite: "SQLiteStore") -> None:
        self._db = sqlite

    async def migrate(self) -> None:
        """Create tables if they don't exist."""
        async with self._db._conn() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    schedule_type TEXT NOT NULL,
                    schedule_expr TEXT NOT NULL,
                    context_mode TEXT NOT NULL DEFAULT 'fresh',
                    pre_check_script TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    last_run TEXT,
                    next_run TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS task_run_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    session_id TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    success INTEGER,
                    summary TEXT,
                    FOREIGN KEY(task_id) REFERENCES scheduled_tasks(id)
                )
            """)
            await conn.commit()

    async def list_tasks(self) -> list[ScheduledTask]:
        ...  # SELECT * FROM scheduled_tasks

    async def get_task(self, task_id: str) -> ScheduledTask | None:
        ...

    async def save_task(self, task: ScheduledTask) -> None:
        ...

    async def delete_task(self, task_id: str) -> None:
        ...
```

`SQLiteStore._conn()` is a private helper — see `src/leuk/persistence/sqlite.py` for
the actual context manager pattern; adapt accordingly or add a public
`execute(sql, params)` method.

### `src/leuk/scheduler/runner.py`

```python
"""TaskScheduler: polls SQLite and runs due tasks."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from croniter import croniter

from leuk.scheduler.store import SchedulerStore
from leuk.scheduler.task import ScheduledTask

logger = logging.getLogger(__name__)


class TaskScheduler:
    def __init__(self, store: SchedulerStore, poll_interval: int = 60) -> None:
        self._store = store
        self._poll_interval = poll_interval
        self._running = False

    async def start(self, agent_factory) -> None:
        """Start polling. agent_factory(prompt) -> coroutine that runs the agent."""
        self._running = True
        while self._running:
            await self._poll(agent_factory)
            await asyncio.sleep(self._poll_interval)

    async def stop(self) -> None:
        self._running = False

    async def _poll(self, agent_factory) -> None:
        now = datetime.now(timezone.utc)
        tasks = await self._store.list_tasks()
        for task in tasks:
            if not task.enabled:
                continue
            if task.next_run and task.next_run <= now:
                asyncio.create_task(self._run_task(task, agent_factory))

    async def _run_task(self, task: ScheduledTask, agent_factory) -> None:
        # Optional pre-check
        if task.pre_check_script:
            proc = await asyncio.create_subprocess_shell(
                task.pre_check_script,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            if proc.returncode != 0:
                logger.info("Skipping task %s: pre-check failed", task.name)
                return

        logger.info("Running scheduled task: %s", task.name)
        await agent_factory(task.prompt)

        # Update next_run
        task.last_run = datetime.now(timezone.utc)
        if task.schedule_type == "cron":
            task.next_run = croniter(task.schedule_expr, task.last_run).get_next(datetime)
        elif task.schedule_type == "interval":
            task.next_run = datetime.fromtimestamp(
                task.last_run.timestamp() + float(task.schedule_expr),
                tz=timezone.utc,
            )
        else:
            task.enabled = False  # one-time task

        await self._store.save_task(task)
```

---

## Step 4 — Wire into the REPL

File: `src/leuk/cli/repl.py`

After creating `AgentSession`, add:

```python
if settings.scheduler.enabled:
    from leuk.scheduler.store import SchedulerStore
    from leuk.scheduler.runner import TaskScheduler

    sched_store = SchedulerStore(sqlite)
    await sched_store.migrate()
    scheduler = TaskScheduler(sched_store, settings.scheduler.poll_interval)

    async def _run_scheduled_prompt(prompt: str) -> None:
        await agent_session.submit(prompt)

    asyncio.create_task(
        scheduler.start(_run_scheduled_prompt), name="scheduler"
    )
```

Also add a `/tasks` REPL command to `_handle_slash_command()` in
`src/leuk/cli/repl.py`:

```python
elif cmd == "/tasks":
    if settings.scheduler.enabled:
        tasks = await sched_store.list_tasks()
        for t in tasks:
            console.print(f"[bold]{t.name}[/bold] ({t.schedule_type}: {t.schedule_expr}) next={t.next_run}")
    else:
        console.print("[yellow]Scheduler not enabled. Set LEUK_SCHEDULER_ENABLED=true.[/yellow]")
```

---

## Step 5 — Enable and create a sample task

```bash
export LEUK_SCHEDULER_ENABLED=true
leuk
```

Then in the REPL, use the shell tool to insert a sample task:

```
> shell: python - <<'EOF'
import asyncio, uuid
from datetime import datetime, timezone, timedelta
from leuk.persistence.sqlite import SQLiteStore
from leuk.scheduler.store import SchedulerStore
from leuk.scheduler.task import ScheduledTask

async def main():
    sqlite = SQLiteStore()
    await sqlite.setup()
    store = SchedulerStore(sqlite)
    await store.migrate()
    task = ScheduledTask(
        id=str(uuid.uuid4()),
        name="daily-summary",
        prompt="Summarize what files changed in this repo today.",
        schedule_type="cron",
        schedule_expr="0 9 * * *",   # 9 AM daily
    )
    from croniter import croniter
    task.next_run = croniter(task.schedule_expr, datetime.now(timezone.utc)).get_next(datetime)
    await store.save_task(task)
    print("Task created.")

asyncio.run(main())
EOF
```

---

## Verification

```bash
# Check tables exist
sqlite3 ~/.config/leuk/leuk.db ".tables"
# Expected: ... scheduled_tasks task_run_logs ...

# List tasks
leuk
# > /tasks
```
