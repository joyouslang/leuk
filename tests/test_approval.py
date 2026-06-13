"""Tests for the tool-approval helpers (pure logic; dialog is interactive)."""

from __future__ import annotations

from leuk.cli.approval import (
    amendable_field,
    approval_scope,
    choice_to_result,
    humanise,
    primary_detail,
    risk_assessment,
)
from leuk.safety import ApprovalResult
from leuk.types import ToolCall


def _tc(name: str, **args) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=args)


class TestChoiceToResult:
    def test_allow_once(self):
        assert choice_to_result("allow") == ApprovalResult(approved=True)

    def test_allow_always_carries_scope(self):
        r = choice_to_result("allow_always", "pkg-config *")
        assert r == ApprovalResult(approved=True, remember=True, scope_pattern="pkg-config *")

    def test_deny_once(self):
        assert choice_to_result("deny") == ApprovalResult(approved=False)

    def test_deny_always_carries_scope(self):
        r = choice_to_result("deny_always", "rm *")
        assert r == ApprovalResult(approved=False, remember=True, scope_pattern="rm *")

    def test_cancel_denies(self):
        # Esc/q on the dialog → None → deny once (the safe default).
        assert choice_to_result(None) == ApprovalResult(approved=False)
        assert choice_to_result("") == ApprovalResult(approved=False)


class TestApprovalScope:
    def test_shell_scopes_to_program(self):
        pat, label = approval_scope(_tc("shell", command="/usr/bin/find /x -name y"))
        assert pat == "find *" and "find" in label

    def test_file_edit_scopes_to_directory(self):
        pat, label = approval_scope(_tc("file_edit", path="src/game/main.c"))
        assert pat == "src/game/*" and "src/game/" in label

    def test_web_fetch_scopes_to_host(self):
        pat, label = approval_scope(_tc("web_fetch", url="https://docs.python.org/3/x"))
        assert pat == "*docs.python.org*" and "docs.python.org" in label

    def test_input_control_scopes_to_action(self):
        pat, label = approval_scope(_tc("input_control", action="click", x=1, y=2))
        assert pat == "click*" and "click" in label

    def test_unknown_falls_back_to_whole_tool(self):
        pat, label = approval_scope(_tc("memory_write", key="k"))
        assert pat == "*" and "memory_write" in label


class TestRiskAssessment:
    def test_dangerous_shell_is_high(self):
        level, _ = risk_assessment(_tc("shell", command="rm -rf /"), "dangerous")
        assert level == "high"

    def test_plain_shell_is_low(self):
        level, _ = risk_assessment(_tc("shell", command="ls -la"), "ask")
        assert level == "low"

    def test_input_control_action_is_high(self):
        level, _ = risk_assessment(_tc("input_control", action="click"), "ask")
        assert level == "high"

    def test_read_is_low(self):
        level, _ = risk_assessment(_tc("file_read", path="/x"), "ask")
        assert level == "low"

    def test_overwrite_is_medium(self):
        level, _ = risk_assessment(_tc("file_edit", path="/x", overwrite=True), "ask")
        assert level == "medium"


class TestAmendableField:
    def test_shell_command_editable(self):
        assert amendable_field(_tc("shell", command="ls")) == "command"

    def test_missing_arg_not_editable(self):
        assert amendable_field(_tc("shell")) is None

    def test_memory_write_not_editable(self):
        assert amendable_field(_tc("memory_write", key="k")) is None


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
