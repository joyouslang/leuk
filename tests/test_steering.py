"""Tests for the steering helpers (pure functions) and config wiring."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from leuk.agent.steering import (
    STEERING_INSTRUCTIONS,
    compose_system_prompt,
    continue_nudge,
    detect_circling,
    parse_reflection,
    parse_text_tool_calls,
    steering_active,
    tool_call_signature,
    truncation_nudge,
)
from leuk.config import SteeringConfig, load_settings
from leuk.types import ToolCall


# ── steering_active ────────────────────────────────────────────────────────


def test_steering_active_auto_is_local_only():
    cfg = SteeringConfig(enabled="auto")
    assert steering_active(cfg, "local") is True
    assert steering_active(cfg, "anthropic") is False
    assert steering_active(cfg, "zen") is False
    assert steering_active(cfg, "openai") is False


def test_steering_active_forced():
    assert steering_active(SteeringConfig(enabled="on"), "anthropic") is True
    assert steering_active(SteeringConfig(enabled="on"), "local") is True
    assert steering_active(SteeringConfig(enabled="off"), "local") is False
    assert steering_active(SteeringConfig(enabled="off"), "anthropic") is False


# ── compose_system_prompt ──────────────────────────────────────────────────


def test_compose_noop_when_inactive():
    base = "BASE PROMPT"
    # auto + non-local → inactive → exact passthrough (no added tokens).
    assert compose_system_prompt(base, SteeringConfig(enabled="auto"), "anthropic") == base
    # explicit off → inactive even for local.
    assert compose_system_prompt(base, SteeringConfig(enabled="off"), "local") == base


def test_compose_appends_when_active():
    base = "BASE PROMPT"
    out = compose_system_prompt(base, SteeringConfig(enabled="auto"), "local")
    assert out.startswith(base)
    assert STEERING_INSTRUCTIONS in out
    assert "Never refuse a task you have the tools for." in out  # stable phrase


def test_compose_extra_instructions_last():
    base = "BASE PROMPT"
    cfg = SteeringConfig(enabled="on", extra_instructions="USEREXTRA prefer ripgrep")
    out = compose_system_prompt(base, cfg, "anthropic")
    # base first, built-in block next, user extra last.
    assert out.index(base) < out.index("Operating discipline") < out.index("USEREXTRA")


def test_compose_blank_extra_instructions_ignored():
    base = "BASE PROMPT"
    cfg = SteeringConfig(enabled="on", extra_instructions="   ")
    out = compose_system_prompt(base, cfg, "anthropic")
    assert out == f"{base}\n\n{STEERING_INSTRUCTIONS}"


def test_compose_adds_desktop_percentage_guidance():
    cfg = SteeringConfig(enabled="on")
    out = compose_system_prompt("BASE", cfg, "local", desktop_control=True)
    assert "xpct" in out and "ypct" in out and "percentage" in out.lower()
    # Absent when desktop control isn't available…
    assert "xpct" not in compose_system_prompt("BASE", cfg, "local", desktop_control=False)
    # …and never injected when steering is inactive, even with desktop control.
    off = SteeringConfig(enabled="off")
    assert compose_system_prompt("BASE", off, "local", desktop_control=True) == "BASE"


# ── parse_reflection ───────────────────────────────────────────────────────


def test_parse_reflection_done():
    assert parse_reflection("DONE") == (False, "")
    assert parse_reflection("Done — everything is complete.") == (False, "")


def test_parse_reflection_continue_with_hint():
    cont, hint = parse_reflection("CONTINUE\nclick the first cell")
    assert cont is True
    assert hint == "click the first cell"


def test_parse_reflection_continue_inline_hint():
    cont, hint = parse_reflection("CONTINUE: open the menu")
    assert cont is True
    assert hint == "open the menu"


def test_parse_reflection_negation_cues():
    # Weak-model phrasing without the literal keyword still means "keep going".
    assert parse_reflection("I am not done yet, still need to win")[0] is True
    assert parse_reflection("The task is incomplete.")[0] is True


def test_parse_reflection_unparseable_accepts():
    # Safe default: an unparseable / empty reply accepts the stop (bounded anyway).
    assert parse_reflection("") == (False, "")
    assert parse_reflection(None) == (False, "")
    assert parse_reflection("hmm, not sure what you mean")[0] is False


# ── nudge builders ─────────────────────────────────────────────────────────


def test_truncation_nudge_is_plain_continue():
    assert truncation_nudge() == "continue"


def test_continue_nudge_includes_hint():
    msg = continue_nudge("click cell 3")
    assert msg.startswith("[STEERING]")
    assert "click cell 3" in msg
    # Without a hint it still produces a usable steering message.
    assert continue_nudge("").startswith("[STEERING]")
    assert "click cell 3" not in continue_nudge("")


# ── circle detection ───────────────────────────────────────────────────────


def _sig(cmd: str) -> str:
    return tool_call_signature([ToolCall(id="x", name="shell", arguments={"command": cmd})])


def test_tool_call_signature_canonical():
    # Argument key order doesn't change the signature.
    a = tool_call_signature([ToolCall(id="1", name="t", arguments={"a": 1, "b": 2})])
    b = tool_call_signature([ToolCall(id="2", name="t", arguments={"b": 2, "a": 1})])
    assert a == b
    # Different name or args → different signature.
    assert a != tool_call_signature([ToolCall(id="3", name="t", arguments={"a": 9, "b": 2})])


def test_detect_circling_identical_repeat():
    a = _sig("x")
    assert detect_circling([a, a, a, a], min_rounds=4) is True


def test_detect_circling_abab_cycle():
    a, b = _sig("x"), _sig("y")
    assert detect_circling([a, b, a, b], min_rounds=4) is True


def test_detect_circling_diversified_not_flagged():
    a, b = _sig("x"), _sig("y")
    # Three repeats then a different call → the model is varying; not a circle.
    assert detect_circling([a, a, a, b], min_rounds=4) is False
    assert detect_circling([a, b, "c", a], min_rounds=4) is False


def test_detect_circling_lengthy_gate():
    a = _sig("x")
    # Below min_rounds, even a pure repeat is not yet flagged.
    assert detect_circling([a, a, a], min_rounds=4) is False
    assert detect_circling([a, a, a, a], min_rounds=6) is False


# ── text tool-call salvage ───────────────────────────────────────────────────

_NAMES = {"browser", "shell"}


def test_salvage_pseudo_xml_user_example():
    # The exact shape the user reported (messy whitespace, no closing param tags).
    content = (
        "Let me open it.\n"
        "<tool_call> <function=browser> <parameter=action> navigate  <parameter=url>\n"
        "https://minesweeper.online/game/new?difficulty=beginner   </tool_call>"
    )
    calls = parse_text_tool_calls(content, _NAMES)
    assert len(calls) == 1
    assert calls[0].name == "browser"
    assert calls[0].arguments == {
        "action": "navigate",
        "url": "https://minesweeper.online/game/new?difficulty=beginner",
    }


def test_salvage_hermes_json():
    content = '<tool_call>{"name": "shell", "arguments": {"command": "ls -la"}}</tool_call>'
    calls = parse_text_tool_calls(content, _NAMES)
    assert [(c.name, c.arguments) for c in calls] == [("shell", {"command": "ls -la"})]


def test_salvage_bare_json_parameters_key():
    content = 'sure: {"name": "shell", "parameters": {"command": "pwd"}}'
    calls = parse_text_tool_calls(content, _NAMES)
    assert [(c.name, c.arguments) for c in calls] == [("shell", {"command": "pwd"})]


def test_salvage_openai_nested_with_stringified_args():
    content = '<tool_call>{"function": {"name": "browser", "arguments": "{\\"action\\": \\"read\\"}"}}</tool_call>'
    calls = parse_text_tool_calls(content, _NAMES)
    assert [(c.name, c.arguments) for c in calls] == [("browser", {"action": "read"})]


def test_salvage_multiple_calls():
    content = (
        "<function=shell><parameter=command>echo a</parameter></function>"
        "<function=shell><parameter=command>echo b</parameter></function>"
    )
    calls = parse_text_tool_calls(content, _NAMES)
    assert [c.arguments["command"] for c in calls] == ["echo a", "echo b"]


def test_salvage_filters_unknown_tool():
    content = "<tool_call><function=not_a_tool><parameter=x>1</parameter></function></tool_call>"
    assert parse_text_tool_calls(content, _NAMES) == []


def test_salvage_no_false_positive_on_prose():
    assert (
        parse_text_tool_calls("I cannot proceed; the DOM differs from expectations.", _NAMES) == []
    )
    assert parse_text_tool_calls("", _NAMES) == []
    assert parse_text_tool_calls(None, _NAMES) == []


def test_salvage_coerces_scalar_args():
    content = "<function=shell><parameter=command>echo hi</parameter><parameter=timeout>5</parameter></function>"
    calls = parse_text_tool_calls(content, {"shell"})
    assert calls[0].arguments == {"command": "echo hi", "timeout": 5}  # "5" → int


# ── config overlay / precedence ────────────────────────────────────────────


def test_steering_config_defaults():
    cfg = SteeringConfig()
    assert cfg.enabled == "auto"
    assert cfg.max_continuations == 3
    assert cfg.reminder_interval == 8
    assert cfg.reflect_only_after_tool_use is True
    assert cfg.nudge_on_truncation is True
    assert cfg.enrich_tool_errors is True
    assert cfg.loop_detection is True
    assert cfg.loop_min_rounds == 4
    assert cfg.loop_max_interventions == 2
    assert cfg.salvage_text_tool_calls is True


def test_steering_config_json_overlay(tmp_path: Path):
    cf = tmp_path / "config.json"
    cf.write_text(json.dumps({"steering": {"enabled": "on", "reminder_interval": 4}}))
    creds = tmp_path / "credentials.json"
    with (
        patch("leuk.config.persistent_config_path", return_value=cf),
        patch("leuk.config.credentials_path", return_value=creds),
    ):
        s = load_settings()
        assert s.steering.enabled == "on"
        assert s.steering.reminder_interval == 4
        # Untouched fields keep their defaults.
        assert s.steering.max_continuations == 3


def test_env_overrides_steering_config_json(tmp_path: Path, monkeypatch):
    cf = tmp_path / "config.json"
    cf.write_text(json.dumps({"steering": {"enabled": "on"}}))
    creds = tmp_path / "credentials.json"
    monkeypatch.setenv("LEUK_STEERING_ENABLED", "off")
    with (
        patch("leuk.config.persistent_config_path", return_value=cf),
        patch("leuk.config.credentials_path", return_value=creds),
    ):
        s = load_settings()
        assert s.steering.enabled == "off"  # env wins over config.json
