"""Tests for core data types."""

from leuk.types import (
    Message,
    Role,
    Session,
    SessionStatus,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
    ToolSpec,
)


def test_role_values():
    assert Role.SYSTEM == "system"
    assert Role.USER == "user"
    assert Role.ASSISTANT == "assistant"
    assert Role.TOOL == "tool"


def test_session_defaults():
    s = Session()
    assert s.status == SessionStatus.ACTIVE
    assert len(s.id) == 32  # uuid4 hex
    assert s.parent_session_id is None


def test_message_with_tool_calls():
    tc = ToolCall(id="call_1", name="shell", arguments={"command": "ls"})
    msg = Message(role=Role.ASSISTANT, content="Let me check.", tool_calls=[tc])
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "shell"


def test_tool_result():
    tr = ToolResult(tool_call_id="call_1", name="shell", content="file.txt")
    assert not tr.is_error


def test_tool_result_error():
    tr = ToolResult(
        tool_call_id="call_1", name="shell", content="not found", is_error=True
    )
    assert tr.is_error


def test_stream_event_types():
    assert StreamEventType.TEXT_DELTA == "text_delta"
    assert StreamEventType.MESSAGE_COMPLETE == "message_complete"


def test_stream_event():
    evt = StreamEvent(type=StreamEventType.TEXT_DELTA, content="Hello")
    assert evt.content == "Hello"
    assert evt.message is None


def test_tool_spec():
    spec = ToolSpec(name="test", description="A test tool", parameters={"type": "object"})
    assert spec.name == "test"
