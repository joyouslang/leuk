"""Git-snapshot undo for agent turns (refactor-plan §5.7).

Before each turn the project's working tree (tracked **and** untracked files,
minus anything ``.gitignore``'d) is captured as a hidden git commit object
built through a *temporary index*, together with where ``HEAD`` and the
current branch pointed. ``/undo`` then restores every file that changed since
the snapshot, deletes files the turn created, and — when the turn made
commits, switched branches, or detached ``HEAD`` — moves the refs back to
their pre-turn position (the turn's commits stay reachable via the reflog).

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
    """A pre-turn snapshot: repo, tree, ref positions, and what the turn was."""

    commit_sha: str
    workdir: Path  # repo toplevel at snapshot time (survives a later /cd)
    user_input: str = ""
    head_sha: str = ""  # commit HEAD pointed at ("" = unborn branch / no commits)
    branch_ref: str = ""  # symbolic HEAD, e.g. "refs/heads/main" ("" = detached)


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
    return UndoSnapshot(
        commit_sha=sha.strip(),
        workdir=top,
        head_sha=_head_sha(top),
        branch_ref=_branch_ref(top),
    )


def _head_sha(top: Path) -> str:
    """The commit HEAD resolves to, or "" on an unborn branch."""
    rc, out = _git(top, "rev-parse", "--verify", "-q", "HEAD")
    return out.strip() if rc == 0 else ""


def _branch_ref(top: Path) -> str:
    """The symbolic ref HEAD points at (e.g. refs/heads/main), or "" if detached."""
    rc, out = _git(top, "symbolic-ref", "-q", "HEAD")
    return out.strip() if rc == 0 else ""


def _revert_refs(snap: UndoSnapshot) -> str:
    """Move HEAD/branch back to the snapshot's position; "" when untouched.

    Handles the turn committing on the current branch (most common), switching
    to / creating another branch, detaching HEAD, and making the first commit
    of an unborn repo. Only refs move — the worktree restore is handled by the
    tree snapshot, and the turn's commits remain reachable via the reflog.
    """
    top = snap.workdir
    cur_ref = _branch_ref(top)
    cur_head = _head_sha(top)
    if cur_ref == snap.branch_ref and cur_head == snap.head_sha:
        return ""

    bits: list[str] = []

    # Re-attach HEAD to the original branch if the turn switched/detached it.
    if cur_ref != snap.branch_ref and snap.branch_ref:
        rc, _ = _git(top, "symbolic-ref", "HEAD", snap.branch_ref)
        if rc != 0:
            return "Could not re-attach HEAD to the original branch — refs NOT reverted."
        bits.append(
            f"switched back to {snap.branch_ref.removeprefix('refs/heads/')}"
        )
        cur_head = _head_sha(top)

    if snap.head_sha and cur_head != snap.head_sha:
        # Count the commits being unwound (when current is a descendant).
        rc, n = _git(top, "rev-list", "--count", f"{snap.head_sha}..{cur_head}")
        count = n.strip() if rc == 0 and n.strip().isdigit() else ""
        # --mixed: move the ref and reset the index to the pre-turn commit;
        # the worktree itself is restored from the tree snapshot.
        rc, _ = _git(top, "reset", "--mixed", "-q", snap.head_sha)
        if rc != 0:
            bits.append("could NOT move the branch back")
        else:
            what = f"{count} commit(s)" if count else "commits"
            bits.append(
                f"unwound {what} (branch reset to {snap.head_sha[:7]}; "
                "they remain in the reflog)"
            )
    elif not snap.head_sha and cur_head:
        # The turn created the repo's very first commit(s): return to unborn.
        if snap.branch_ref:
            _git(top, "update-ref", "-d", snap.branch_ref)
        _git(top, "read-tree", "--empty")
        bits.append("removed the turn's initial commit(s)")

    return ("Refs: " + "; ".join(bits) + ".") if bits else ""


def revert_worktree(snap: UndoSnapshot) -> str:
    """Restore the repo to *snap*'s state; return a human summary.

    Files modified or deleted since the snapshot are restored from it; files
    created since are removed; commits/branch switches the turn made are
    unwound back to the recorded ref positions (kept in the reflog).
    """
    top = snap.workdir
    current = snapshot_worktree(top)
    if current is None:
        return "Could not snapshot the current tree — files NOT reverted."

    # Refs first (commits made during the turn), then the file restore — the
    # tree diff below is snapshot-vs-snapshot, independent of where HEAD is.
    ref_summary = _revert_refs(snap)
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
        files_summary = "No file changes to revert."
    else:
        bits = []
        if restore:
            bits.append(f"{len(restore)} restored")
        if deleted:
            bits.append(f"{deleted} deleted")
        files_summary = f"Reverted {total} file change(s): {', '.join(bits)}."
    return f"{files_summary} {ref_summary}".strip()


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
