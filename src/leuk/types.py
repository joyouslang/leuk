"""Core data types used throughout the agent system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """Message role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class SessionStatus(StrEnum):
    """Lifecycle status of an agent session."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentState(StrEnum):
    """Runtime state of an AgentSession's background loop."""

    IDLE = "idle"  # waiting for user input
    THINKING = "thinking"  # LLM is generating
    TOOL_RUNNING = "tool_running"  # executing tool calls
    INTERRUPTED = "interrupted"  # generation was interrupted by user
    STOPPED = "stopped"  # session loop has exited


@dataclass(slots=True)
class ToolCall:
    """A tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]
    # Provider-specific metadata (e.g. Google's thought_signature).
    # Not persisted — only used within a single agent turn.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    """The result of executing a tool call."""

    tool_call_id: str
    name: str
    content: str
    # Provider-specific metadata forwarded from the originating ToolCall.
    metadata: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False


@dataclass(slots=True)
class Message:
    """A single message in a conversation."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_result: ToolResult | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.metadata.get("id", "")


@dataclass(slots=True)
class Session:
    """A persistent agent session that survives restarts."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    system_prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_session_id: str | None = None  # For sub-agent tracking


@dataclass(slots=True)
class ToolSpec:
    """JSON-schema-style tool specification sent to the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


class StreamEventType(StrEnum):
    """Types of events emitted during streaming."""

    TEXT_DELTA = "text_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    MESSAGE_COMPLETE = "message_complete"
    # AgentSession-level events
    STATE_CHANGE = "state_change"  # content = new AgentState value
    TURN_COMPLETE = "turn_complete"  # agent finished responding, waiting for input
    ERROR = "error"  # content = error description
    RATE_LIMITED = "rate_limited"  # content = "Rate limited, retrying in Ns..."


@dataclass(slots=True)
class StreamEvent:
    """A single event in a streaming response.

    - TEXT_DELTA: partial text token (content holds the delta)
    - TOOL_CALL_START: tool call begun (tool_call has id+name, arguments may be partial)
    - TOOL_CALL_DELTA: partial arguments JSON string (content holds the delta)
    - TOOL_CALL_END: tool call arguments fully received
    - MESSAGE_COMPLETE: final assembled Message (message field is set)
    """

    type: StreamEventType
    content: str = ""
    tool_call: ToolCall | None = None
    message: Message | None = None
