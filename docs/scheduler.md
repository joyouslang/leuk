[Home](README.md) › Scheduler

# Scheduler

An optional background scheduler runs agent tasks on a cadence. Enable with
`LEUK_SCHEDULER_ENABLED=1`.

## Pieces — `src/leuk/scheduler/`

| Module | Role |
|--------|------|
| `task.py` | `ScheduledTask` dataclass (schedule type/expression, prompt) |
| `store.py` | `SchedulerStore` — SQLite CRUD for tasks |
| `runner.py` | `TaskScheduler` — background poll loop that runs due tasks |

The runner shares the provider, tool registry, and [SafetyGuard](safety.md) with
the REPL. Each run executes an agent turn for the task's prompt.

`/tasks` lists scheduled tasks and their next/last run.

## See also

- [Configuration](configuration.md) · [Architecture Overview](architecture.md)
