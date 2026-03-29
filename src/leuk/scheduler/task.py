"""ScheduledTask dataclass."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class ScheduledTask:
    """A task that runs an agent prompt on a schedule.

    Fields
    ------
    id:
        Unique task identifier (UUID).
    name:
        Human-readable task name.
    prompt:
        The prompt sent to the agent on each run.
    schedule_type:
        One of ``"cron"``, ``"interval"``, or ``"once"``.
    schedule_expr:
        Interpretation depends on schedule_type:
        - ``"cron"``: standard 5-field cron expression (e.g. ``"0 9 * * 1"``).
        - ``"interval"``: number of seconds between runs as a string (e.g. ``"3600"``).
        - ``"once"``: ISO 8601 datetime string for the single run time.
    pre_check_script:
        Optional shell command. If exit code != 0, the agent invocation is skipped
        for that cycle (useful to avoid API cost when preconditions aren't met).
    context_mode:
        ``"fresh"`` creates a new session per run; ``"resume"`` continues
        the same session (stored in ``session_id``).
    enabled:
        When False the task is not executed during polls.
    last_run:
        UTC datetime of the most recent run, or None if never run.
    next_run:
        UTC datetime of the next scheduled run.
    session_id:
        Persisted session ID used when ``context_mode == "resume"``.
    """

    name: str
    prompt: str
    schedule_type: Literal["cron", "interval", "once"]
    schedule_expr: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pre_check_script: str = ""
    context_mode: Literal["fresh", "resume"] = "fresh"
    enabled: bool = True
    last_run: datetime | None = None
    next_run: datetime | None = None
    session_id: str | None = None
