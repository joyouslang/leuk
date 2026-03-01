"""Tests for context window management."""

from __future__ import annotations

import pytest

from leuk.agent.context import (
    estimate_message_tokens,
    estimate_total_tokens,
    sliding_window,
    truncate_tool_results,
)
from leuk.types import Message, Role, ToolResult


def _msg(role: Role, content: str) -> Message:
    return Message(role=role, content=content)


def test_estimate_tokens():
    msg = _msg(Role.USER, "Hello world")
    tokens = estimate_message_tokens(msg)
    # "Hello world" = 11 chars / 4 = 2 + 4 overhead = 6
    assert tokens > 0


def test_estimate_total():
    msgs = [_msg(Role.USER, "a" * 400), _msg(Role.ASSISTANT, "b" * 400)]
    total = estimate_total_tokens(msgs)
    assert total > 100  # 800 chars / 4 = 200 + overhead


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


def test_sliding_window_fits():
    msgs = [_msg(Role.SYSTEM, "sys"), _msg(Role.USER, "hi")]
    result = sliding_window(msgs, max_tokens=10_000)
    assert len(result) == 2


def test_sliding_window_drops_old():
    msgs = [_msg(Role.SYSTEM, "sys")]
    # Add many messages to exceed budget
    for i in range(100):
        msgs.append(_msg(Role.USER, f"message {'x' * 1000} {i}"))
        msgs.append(_msg(Role.ASSISTANT, f"reply {'y' * 1000} {i}"))

    result = sliding_window(msgs, max_tokens=1_000)
    # Should have system + summary + some recent messages
    assert len(result) < len(msgs)
    # System prompt should be preserved
    assert result[0].role == Role.SYSTEM
    # Summary should be injected
    has_summary = any("trimmed" in (m.content or "").lower() for m in result)
    assert has_summary


def test_sliding_window_preserves_system():
    msgs = [
        _msg(Role.SYSTEM, "system prompt"),
        _msg(Role.USER, "x" * 10_000),
        _msg(Role.ASSISTANT, "y" * 10_000),
        _msg(Role.USER, "latest"),
    ]
    result = sliding_window(msgs, max_tokens=100)
    assert result[0].role == Role.SYSTEM
    assert result[0].content == "system prompt"
