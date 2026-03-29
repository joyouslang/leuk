"""Channel abstraction: protocol + normalized message type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable


@dataclass
class ChannelMessage:
    """A normalized inbound message from any channel.

    All channel backends convert their native message types into this
    structure before invoking the registered callback.
    """

    text: str
    chat_id: str
    sender: str
    channel: str
    metadata: dict[str, Any] = field(default_factory=dict)


# Callback signature: async def handler(msg: ChannelMessage) -> None
MessageCallback = Callable[[ChannelMessage], Awaitable[None]]


@runtime_checkable
class Channel(Protocol):
    """Protocol that every channel backend must satisfy.

    Implementations self-register via ``register_channel(name, factory)``
    at import time so the :class:`ChannelRegistry` discovers them
    automatically through the barrel import pattern.
    """

    name: str

    async def connect(self) -> None:
        """Establish connection to the messaging platform."""
        ...

    async def send(self, chat_id: str, text: str) -> None:
        """Send a reply to a specific chat/conversation."""
        ...

    def on_message(self, callback: MessageCallback) -> None:
        """Register the callback invoked for each incoming message."""
        ...

    async def disconnect(self) -> None:
        """Gracefully close the connection and release resources."""
        ...
