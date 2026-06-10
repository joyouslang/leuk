"""Git-snapshot undo for agent turns (refactor-plan §5.7).

Before each turn the project's working tree (tracked **and** untracked files,
minus anything ``.gitignore``'d) is captured as a hidden git commit object
built through a *temporary index* — the user's index, ``HEAD``, stash, and
refs are never touched. ``/undo`` then restores every file that changed since
the snapshot and deletes files the turn created.

Why git (vs. file copies / FS snapshots): git is already present in the
projects leuk targets, captures the whole tree atomically, and handles
deletions and binary files. The snapshot commit is parentless and unreferenced;
within a session that is safe (``git gc`` keeps unreachable objects for weeks).

Shell side-effects (processes, network) are **not** undoable; callers must say
so in the summary they print. Outside a git repo, snapshotting returns ``None``
and ``/undo`` degrades to context-only.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from leuk.types import Message, Role

logger = logging.getLogger(__name__)

_GIT_TIMEOUT = 30  # seconds; snapshots must never hang a turn

# Ephemeral identity so commit-tree works even with no git config (never
# touches the user's configuration).
_GIT_ID = ["-c", "user.name=leuk-undo", "-c", "user.email=undo@leuk.local"]


@dataclass(slots=True)
class UndoSnapshot:
    """A pre-turn snapshot: which repo, which tree, and what the turn was."""

    commit_sha: str
    workdir: Path  # repo toplevel at snapshot time (survives a later /cd)
    user_input: str = ""


def _git(workdir: Path, *args: str, env: dict[str, str] | None = None) -> tuple[int, str]:
    """Run ``git -C workdir *args``; return ``(returncode, stdout)``."""
    import os

    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    try:
        proc = subprocess.run(
            ["git", "-C", str(workdir), *_GIT_ID, *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            env=full_env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("git %s failed: %s", args[:2], exc)
        return 1, ""
    return proc.returncode, proc.stdout


def _toplevel(workdir: Path) -> Path | None:
    """The repo root containing *workdir*, or None when not in a git repo."""
    rc, out = _git(workdir, "rev-parse", "--show-toplevel")
    if rc != 0 or not out.strip():
        return None
    return Path(out.strip())


def snapshot_worktree(workdir: Path) -> UndoSnapshot | None:
    """Capture the working tree as a hidden commit; None outside a git repo.

    Uses a throwaway ``GIT_INDEX_FILE`` so ``git add -A`` stages the full tree
    (tracked + untracked, respecting .gitignore) without disturbing the user's
    real index.
    """
    top = _toplevel(workdir)
    if top is None:
        return None
    # The temp index must NOT pre-exist (git rejects a 0-byte index file), so
    # point GIT_INDEX_FILE at a fresh name inside a throwaway directory.
    with tempfile.TemporaryDirectory(prefix="leuk-undo-") as tmpdir:
        env = {"GIT_INDEX_FILE": str(Path(tmpdir) / "index")}
        rc, _ = _git(top, "add", "-A", env=env)
        if rc != 0:
            return None
        rc, tree = _git(top, "write-tree", env=env)
        if rc != 0 or not tree.strip():
            return None
        rc, sha = _git(top, "commit-tree", tree.strip(), "-m", "leuk undo snapshot")
    if rc != 0 or not sha.strip():
        return None
    return UndoSnapshot(commit_sha=sha.strip(), workdir=top)


def revert_worktree(snap: UndoSnapshot) -> str:
    """Restore the working tree to *snap*'s state; return a human summary.

    Files modified or deleted since the snapshot are restored from it; files
    created since are removed. The user's index/HEAD are left alone (restores
    go through ``git restore --worktree``).
    """
    top = snap.workdir
    current = snapshot_worktree(top)
    if current is None:
        return "Could not snapshot the current tree — files NOT reverted."
    rc, out = _git(
        top, "diff", "--name-status", "--no-renames", "-z",
        snap.commit_sha, current.commit_sha,
    )
    if rc != 0:
        return "git diff failed — files NOT reverted."

    parts = out.split("\0")
    restore: list[str] = []
    deleted = 0
    # -z output alternates STATUS, path (a trailing empty element remains).
    for status, path in zip(parts[::2], parts[1::2]):
        if not path:
            continue
        if status == "A":  # created during the turn → remove
            try:
                (top / path).unlink(missing_ok=True)
                deleted += 1
            except OSError:
                logger.debug("undo: could not delete %s", path, exc_info=True)
        else:  # modified or deleted during the turn → restore from snapshot
            restore.append(path)

    # Chunked so a huge turn can't overflow the OS argv limit.
    for i in range(0, len(restore), 500):
        rc, _ = _git(
            top, "restore", "--source", snap.commit_sha, "--worktree", "--",
            *restore[i : i + 500],
        )
        if rc != 0:
            return "git restore failed — files may be partially reverted."

    total = len(restore) + deleted
    if total == 0:
        return "No file changes to revert."
    bits = []
    if restore:
        bits.append(f"{len(restore)} restored")
    if deleted:
        bits.append(f"{deleted} deleted")
    return f"Reverted {total} file change(s): {', '.join(bits)}."


def last_exchange_start(messages: list[Message]) -> int | None:
    """Index of the last real user message (the start of the last exchange).

    Returns None when there is no user turn to pop. ``[SYSTEM]`` housekeeping
    messages don't count as user turns.
    """
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.role is Role.USER:
            content = (m.content or "").strip()
            if content and not content.startswith("[SYSTEM]"):
                return i
    return None
