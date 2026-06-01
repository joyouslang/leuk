"""Desktop-control safety + browser action surface tests."""

from __future__ import annotations

import pytest

from leuk.config import PermissionAction, SafetyConfig, ToolRule
from leuk.safety import SafetyGuard
from leuk.tools.browser import BrowserTool
from leuk.types import ToolCall


async def _never_confirm(reason, tool_call):  # pragma: no cover - shouldn't run
    raise AssertionError("confirm should not be called in these checks")


class TestDesktopApprovalRule:
    def test_input_control_asks_with_rule(self):
        cfg = SafetyConfig(rules=[ToolRule(tool="input_control", pattern="*", action="ask")])
        guard = SafetyGuard(cfg, confirm_callback=_never_confirm)
        verdict = guard.check(ToolCall(id="1", name="input_control", arguments={"action": "click"}))
        assert verdict.verdict == PermissionAction.ASK

    def test_input_control_allowed_without_rule_under_auto(self):
        # Auto-approve = no input_control ask rule; auto policy allows everything.
        from leuk.config import ReviewPolicy

        cfg = SafetyConfig(review_policy=ReviewPolicy.AUTO, rules=[])
        guard = SafetyGuard(cfg, confirm_callback=_never_confirm)
        verdict = guard.check(ToolCall(id="2", name="input_control", arguments={"action": "click"}))
        assert verdict.verdict == PermissionAction.ALLOW


class TestBrowserSurface:
    def test_spec_has_spa_actions(self):
        enum = BrowserTool().spec.parameters["properties"]["action"]["enum"]
        for a in ("read_page", "find", "fill", "press", "wait_for", "wait_for_network_idle"):
            assert a in enum

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        out = await BrowserTool().execute({"action": "definitely_not_real"})
        assert "[ERROR] Unknown action" in out
