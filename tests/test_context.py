"""Tests for context window management."""

from __future__ import annotations

from leuk.agent.context import (
    _emergency_drop,
    _fix_orphaned_pairs,
    _safe_split_index,
    estimate_message_tokens,
    estimate_total_tokens,
    mask_observations,
    truncate_tool_results,
)
from leuk.types import Message, Role, ToolCall, ToolResult


def _msg(role: Role, content: str) -> Message:
    return Message(role=role, content=content)


def _tool_msg(name: str, content: str) -> Message:
    tr = ToolResult(tool_call_id="c1", name=name, content=content)
    return Message(role=Role.TOOL, tool_result=tr)


# ── Token estimation ──────────────────────────────────────────────────────


def test_estimate_tokens():
    msg = _msg(Role.USER, "Hello world")
    tokens = estimate_message_tokens(msg)
    # "Hello world" = 11 chars / 4 = 2 + 4 overhead = 6
    assert tokens > 0


def test_estimate_total():
    msgs = [_msg(Role.USER, "a" * 400), _msg(Role.ASSISTANT, "b" * 400)]
    total = estimate_total_tokens(msgs)
    assert total > 100  # 800 chars / 4 = 200 + overhead


# ── Stage 1: truncate ────────────────────────────────────────────────────


def test_truncate_tool_results():
    long_content = "x" * 100_000
    tr = ToolResult(tool_call_id="c1", name="shell", content=long_content)
    msg = Message(role=Role.TOOL, tool_result=tr)

    truncated = truncate_tool_results([msg], max_result_tokens=100)
    assert len(truncated) == 1
    assert len(truncated[0].tool_result.content) < len(long_content)
    assert "truncated" in truncated[0].tool_result.content


def test_truncate_tool_results_no_change():
    tr = ToolResult(tool_call_id="c1", name="shell", content="short")
    msg = Message(role=Role.TOOL, tool_result=tr)

    truncated = truncate_tool_results([msg])
    assert truncated[0].tool_result.content == "short"


# ── Stage 2: observation masking ─────────────────────────────────────────


def test_mask_observations_under_threshold():
    """No masking when under the threshold."""
    msgs = [_msg(Role.USER, "hi"), _tool_msg("shell", "x" * 1000)]
    result = mask_observations(msgs, max_tokens=100_000)
    # Should return same objects — no masking applied
    assert result[1] is msgs[1]


def test_mask_observations_masks_old_tool_results():
    """Old tool results are masked when over threshold."""
    msgs = [
        _msg(Role.SYSTEM, "sys"),
        _tool_msg("shell", "x" * 10_000),  # old, large — should be masked
        _tool_msg("file_read", "y" * 10_000),  # old, large — should be masked
        _msg(Role.USER, "latest"),
    ]
    # Set threshold so masking kicks in immediately (threshold=0.0)
    result = mask_observations(msgs, max_tokens=100_000, threshold=0.0)

    # The two tool results should now have short placeholder content
    assert "masked" in result[1].tool_result.content
    assert "masked" in result[2].tool_result.content
    # User message is untouched
    assert result[3].content == "latest"


def test_mask_observations_skips_small_results():
    """Tool results under 200 chars are not masked."""
    msgs = [_tool_msg("shell", "tiny")]
    result = mask_observations(msgs, max_tokens=1, threshold=0.0)
    assert result[0].tool_result.content == "tiny"


def test_mask_observations_preserves_error_results():
    """Error tool results are never masked — they contain diagnostic info."""
    tr = ToolResult(tool_call_id="c1", name="shell", content="x" * 5000, is_error=True)
    msgs = [Message(role=Role.TOOL, tool_result=tr)]
    result = mask_observations(msgs, max_tokens=1, threshold=0.0)
    # Error results should not be masked
    assert result[0] is msgs[0]


# ── Stage 4: emergency drop ──────────────────────────────────────────────


def test_emergency_drop_fits():
    system = [_msg(Role.SYSTEM, "sys")]
    rest = [_msg(Role.USER, "hi")]
    result = _emergency_drop(system, rest, max_tokens=10_000)
    assert len(result) == 2


def test_emergency_drop_drops_old():
    system = [_msg(Role.SYSTEM, "sys")]
    rest = [_msg(Role.USER, f"message {'x' * 1000} {i}") for i in range(100)]

    result = _emergency_drop(system, list(rest), max_tokens=1_000)
    assert len(result) < len(rest) + 1
    assert result[0].role == Role.SYSTEM
    has_note = any("removed" in (m.content or "").lower() for m in result)
    assert has_note


def test_emergency_drop_preserves_system():
    system = [_msg(Role.SYSTEM, "system prompt")]
    rest = [
        _msg(Role.USER, "x" * 10_000),
        _msg(Role.ASSISTANT, "y" * 10_000),
        _msg(Role.USER, "latest"),
    ]
    result = _emergency_drop(system, list(rest), max_tokens=100)
    assert result[0].role == Role.SYSTEM
    assert result[0].content == "system prompt"


def test_emergency_drop_keeps_tool_pairs():
    """Dropping an assistant with tool_calls must also drop its tool_results."""
    tc = ToolCall(id="tc1", name="shell", arguments={"command": "ls"})
    tr = ToolResult(tool_call_id="tc1", name="shell", content="x" * 5000)
    system = [_msg(Role.SYSTEM, "sys")]
    rest = [
        Message(role=Role.ASSISTANT, content=None, tool_calls=[tc]),
        Message(role=Role.TOOL, tool_result=tr),
        _msg(Role.USER, "latest"),
    ]
    result = _emergency_drop(system, list(rest), max_tokens=100)
    # Both assistant+tool_call and tool_result should be dropped together
    for msg in result:
        assert msg.role != Role.TOOL or msg.tool_result is None or msg.tool_result.tool_call_id != "tc1"


# ── Tool-use/tool-result pairing ────────────────────────────────────────


def test_safe_split_avoids_breaking_pair():
    """Split index adjusts forward past tool_results after a tool_use."""
    tc = ToolCall(id="tc1", name="shell", arguments={"command": "ls"})
    tr1 = ToolResult(tool_call_id="tc1", name="shell", content="output1")
    msgs = [
        _msg(Role.USER, "do something"),
        Message(role=Role.ASSISTANT, content=None, tool_calls=[tc]),
        Message(role=Role.TOOL, tool_result=tr1),
        _msg(Role.ASSISTANT, "done"),
        _msg(Role.USER, "thanks"),
    ]
    # Naive split at 2 would land right on the tool_result
    idx = _safe_split_index(msgs, 2)
    # Should move past the tool_result to index 3
    assert idx == 3


def test_safe_split_at_boundary():
    """Split at message boundary that's already safe stays put."""
    msgs = [
        _msg(Role.USER, "hi"),
        _msg(Role.ASSISTANT, "hello"),
        _msg(Role.USER, "bye"),
    ]
    assert _safe_split_index(msgs, 2) == 2


def test_fix_orphaned_pairs_injects_placeholder():
    """Orphaned tool_use gets a placeholder tool_result."""
    tc = ToolCall(id="tc_orphan", name="shell", arguments={"command": "ls"})
    msgs = [
        _msg(Role.USER, "hi"),
        Message(role=Role.ASSISTANT, content=None, tool_calls=[tc]),
        # Missing tool_result for tc_orphan!
        _msg(Role.USER, "what happened?"),
    ]
    fixed = _fix_orphaned_pairs(msgs)
    # Should have injected a placeholder tool_result
    assert len(fixed) == 4
    assert fixed[2].role == Role.TOOL
    assert fixed[2].tool_result.tool_call_id == "tc_orphan"
    assert "compaction" in fixed[2].tool_result.content


def test_fix_orphaned_pairs_no_change_when_paired():
    """No changes when all tool_use/tool_result pairs are intact."""
    tc = ToolCall(id="tc1", name="shell", arguments={"command": "ls"})
    tr = ToolResult(tool_call_id="tc1", name="shell", content="output")
    msgs = [
        Message(role=Role.ASSISTANT, content=None, tool_calls=[tc]),
        Message(role=Role.TOOL, tool_result=tr),
    ]
    fixed = _fix_orphaned_pairs(msgs)
    assert fixed is msgs  # same object — no changes
