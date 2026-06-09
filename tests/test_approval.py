"""Tests for the tool-approval helpers (pure logic; dialog is interactive)."""

from __future__ import annotations

from leuk.cli.approval import choice_to_result, humanise, primary_detail
from leuk.safety import ApprovalResult
from leuk.types import ToolCall


def _tc(name: str, **args) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=args)


class TestChoiceToResult:
    def test_allow_once(self):
        assert choice_to_result("allow") == ApprovalResult(approved=True)

    def test_allow_always(self):
        assert choice_to_result("allow_always") == ApprovalResult(approved=True, remember=True)

    def test_deny_once(self):
        assert choice_to_result("deny") == ApprovalResult(approved=False)

    def test_deny_always(self):
        assert choice_to_result("deny_always") == ApprovalResult(approved=False, remember=True)

    def test_cancel_denies(self):
        # Esc/q on the dialog → None → deny once (the safe default).
        assert choice_to_result(None) == ApprovalResult(approved=False)
        assert choice_to_result("") == ApprovalResult(approved=False)


class TestHumanise:
    def test_known_tool(self):
        assert humanise(_tc("shell")) == "Run a shell command"
        assert humanise(_tc("file_edit")) == "Edit a file"

    def test_unknown_tool(self):
        assert "frobnicate" in humanise(_tc("frobnicate"))


class TestPrimaryDetail:
    def test_prefers_command(self):
        assert primary_detail(_tc("shell", command="ls -la")) == "command: ls -la"

    def test_path_then_url(self):
        assert primary_detail(_tc("file_edit", path="/a/b")) == "path: /a/b"
        assert primary_detail(_tc("web_fetch", url="https://x")) == "url: https://x"

    def test_fallback_to_all_args(self):
        out = primary_detail(_tc("misc", foo=1, bar=2))
        assert "foo=1" in out and "bar=2" in out

    def test_no_args(self):
        assert primary_detail(_tc("noop")) == "(no arguments)"
