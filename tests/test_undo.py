"""Tests for git-snapshot undo (refactor-plan §5.7)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from leuk.agent.undo import (
    UndoSnapshot,
    last_exchange_start,
    revert_worktree,
    snapshot_worktree,
)
from leuk.types import Message, Role


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)


def _git(path: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True, text=True, check=True,
    ).stdout
    return out.strip()


def _commit_all(path: Path, msg: str) -> str:
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", msg)
    return _git(path, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("original\n")
    return tmp_path


class TestSnapshot:
    def test_not_a_repo_returns_none(self, tmp_path: Path):
        assert snapshot_worktree(tmp_path / "nowhere") is None

    def test_snapshot_in_repo(self, repo: Path):
        snap = snapshot_worktree(repo)
        assert snap is not None
        assert len(snap.commit_sha) == 40
        assert snap.workdir == repo

    def test_snapshot_works_without_commits(self, repo: Path):
        # Fresh repo, no HEAD yet — snapshot must still capture the tree.
        snap = snapshot_worktree(repo)
        assert snap is not None

    def test_snapshot_does_not_touch_user_index(self, repo: Path):
        snapshot_worktree(repo)
        out = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout
        # a.txt must still be untracked (??) — not staged by the snapshot.
        assert "?? a.txt" in out


class TestRevert:
    def test_modified_file_restored(self, repo: Path):
        snap = snapshot_worktree(repo)
        (repo / "a.txt").write_text("clobbered by agent\n")
        summary = revert_worktree(snap)
        assert (repo / "a.txt").read_text() == "original\n"
        assert "1 restored" in summary

    def test_created_file_deleted(self, repo: Path):
        snap = snapshot_worktree(repo)
        (repo / "junk.txt").write_text("agent made this\n")
        summary = revert_worktree(snap)
        assert not (repo / "junk.txt").exists()
        assert "1 deleted" in summary

    def test_deleted_file_restored(self, repo: Path):
        snap = snapshot_worktree(repo)
        (repo / "a.txt").unlink()
        summary = revert_worktree(snap)
        assert (repo / "a.txt").read_text() == "original\n"
        assert "1 restored" in summary

    def test_mixed_changes(self, repo: Path):
        (repo / "b.txt").write_text("keep me\n")
        snap = snapshot_worktree(repo)
        (repo / "a.txt").write_text("changed\n")
        (repo / "b.txt").unlink()
        (repo / "new.txt").write_text("created\n")
        summary = revert_worktree(snap)
        assert (repo / "a.txt").read_text() == "original\n"
        assert (repo / "b.txt").read_text() == "keep me\n"
        assert not (repo / "new.txt").exists()
        assert "3 file change(s)" in summary

    def test_no_changes(self, repo: Path):
        snap = snapshot_worktree(repo)
        assert revert_worktree(snap) == "No file changes to revert."

    def test_gitignored_files_untouched(self, repo: Path):
        (repo / ".gitignore").write_text("*.log\n")
        snap = snapshot_worktree(repo)
        (repo / "debug.log").write_text("ignored artifact\n")
        revert_worktree(snap)
        # Ignored files are outside the snapshot — never deleted/restored.
        assert (repo / "debug.log").exists()


class TestRevertRefs:
    """/undo must unwind commits/branch moves the agent made during a turn."""

    def test_commit_during_turn_is_unwound(self, repo: Path):
        base = _commit_all(repo, "base")
        snap = snapshot_worktree(repo)
        assert snap.head_sha == base

        # The "agent" creates a file and commits it (the poetry.txt case).
        (repo / "poem.txt").write_text("roses are red\n")
        _commit_all(repo, "Add poem")

        summary = revert_worktree(snap)
        assert _git(repo, "rev-parse", "HEAD") == base  # branch moved back
        assert not (repo / "poem.txt").exists()  # file removed too
        assert "unwound 1 commit(s)" in summary
        assert "reflog" in summary
        # Worktree is clean relative to the restored HEAD.
        assert _git(repo, "status", "--porcelain") == ""

    def test_multiple_commits_unwound(self, repo: Path):
        base = _commit_all(repo, "base")
        snap = snapshot_worktree(repo)
        (repo / "one.txt").write_text("1\n")
        _commit_all(repo, "one")
        (repo / "two.txt").write_text("2\n")
        _commit_all(repo, "two")

        summary = revert_worktree(snap)
        assert _git(repo, "rev-parse", "HEAD") == base
        assert "unwound 2 commit(s)" in summary

    def test_branch_switch_is_reverted(self, repo: Path):
        base = _commit_all(repo, "base")
        original = _git(repo, "symbolic-ref", "HEAD")
        snap = snapshot_worktree(repo)

        _git(repo, "checkout", "-q", "-b", "feature")
        (repo / "feat.txt").write_text("x\n")
        _commit_all(repo, "feature work")

        summary = revert_worktree(snap)
        assert _git(repo, "symbolic-ref", "HEAD") == original  # back on main
        assert _git(repo, "rev-parse", "HEAD") == base
        assert not (repo / "feat.txt").exists()
        assert "switched back to" in summary

    def test_first_commit_in_unborn_repo_is_removed(self, repo: Path):
        # No commits yet — snapshot records an unborn branch.
        snap = snapshot_worktree(repo)
        assert snap.head_sha == ""
        _commit_all(repo, "very first")

        summary = revert_worktree(snap)
        # HEAD is unborn again (rev-parse HEAD fails).
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", "-q", "HEAD"],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0
        assert "initial commit" in summary

    def test_no_ref_changes_no_ref_summary(self, repo: Path):
        _commit_all(repo, "base")
        snap = snapshot_worktree(repo)
        (repo / "a.txt").write_text("edited but not committed\n")
        summary = revert_worktree(snap)
        assert "Refs:" not in summary
        assert (repo / "a.txt").read_text() == "original\n"


class TestLastExchangeStart:
    def test_finds_last_user_turn(self):
        msgs = [
            Message(role=Role.USER, content="first"),
            Message(role=Role.ASSISTANT, content="reply"),
            Message(role=Role.USER, content="second"),
            Message(role=Role.ASSISTANT, content="reply 2"),
        ]
        assert last_exchange_start(msgs) == 2

    def test_skips_system_housekeeping(self):
        msgs = [
            Message(role=Role.USER, content="real"),
            Message(role=Role.ASSISTANT, content="reply"),
            Message(role=Role.USER, content="[SYSTEM] housekeeping"),
        ]
        assert last_exchange_start(msgs) == 0

    def test_no_user_turn(self):
        assert last_exchange_start([Message(role=Role.SYSTEM, content="sys")]) is None
        assert last_exchange_start([]) is None


class TestSqliteDeleteLastExchange:
    @pytest.mark.asyncio
    async def test_deletes_from_last_user_message(self, tmp_path: Path):
        from leuk.config import SQLiteConfig
        from leuk.persistence.sqlite import SQLiteStore
        from leuk.types import Session

        store = SQLiteStore(SQLiteConfig(path=str(tmp_path / "t.db")))
        await store.init()
        sess = Session()
        await store.create_session(sess)
        for m in (
            Message(role=Role.USER, content="first"),
            Message(role=Role.ASSISTANT, content="reply"),
            Message(role=Role.USER, content="second"),
            Message(role=Role.ASSISTANT, content="reply 2"),
        ):
            await store.append_message(sess.id, m)

        deleted = await store.delete_last_exchange(sess.id)
        assert deleted == 2
        remaining = await store.get_messages(sess.id)
        assert [m.content for m in remaining] == ["first", "reply"]
        await store.close()

    @pytest.mark.asyncio
    async def test_no_user_message_deletes_nothing(self, tmp_path: Path):
        from leuk.config import SQLiteConfig
        from leuk.persistence.sqlite import SQLiteStore
        from leuk.types import Session

        store = SQLiteStore(SQLiteConfig(path=str(tmp_path / "t.db")))
        await store.init()
        sess = Session()
        await store.create_session(sess)
        await store.append_message(sess.id, Message(role=Role.ASSISTANT, content="hi"))

        assert await store.delete_last_exchange(sess.id) == 0
        assert len(await store.get_messages(sess.id)) == 1
        await store.close()


class TestUndoSnapshotSentinel:
    def test_sentinel_has_empty_sha(self, tmp_path: Path):
        # The REPL pushes a sentinel outside git repos (context-only undo).
        snap = UndoSnapshot(commit_sha="", workdir=tmp_path, user_input="msg")
        assert not snap.commit_sha
        assert snap.user_input == "msg"
