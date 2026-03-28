"""Multi-channel messaging: registry + self-registering channel factories.

Usage
-----
Channel modules call :func:`register_channel` at import time::

    # In channels/telegram.py
    register_channel("telegram", _make_telegram)

The :class:`ChannelRegistry` triggers all registrations by importing the
channel sub-modules (barrel import pattern), then instantiates each enabled
channel and wires it to per-chat :class:`~leuk.agent.session.AgentSession`s.

Lifecycle::

    registry = ChannelRegistry(session_factory, config)
    await registry.start()   # connects all enabled channels
    # ... run until shutdown ...
    await registry.stop()    # disconnects and stops all sessions
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from leuk.channels.base import Channel, ChannelMessage, MessageCallback

logger = logging.getLogger(__name__)

# ── Self-registration table ────────────────────────────────────────────────

# name -> factory(config: ChannelsConfig) -> Channel | None
_factories: dict[str, Callable[[Any], Channel | None]] = {}


def register_channel(name: str, factory: Callable[[Any], Channel | None]) -> None:
    """Register a channel factory.

    Called at module import time by each channel implementation.  The
    *factory* receives the :class:`~leuk.config.ChannelsConfig` instance
    and must return a :class:`Channel` or ``None`` if the channel cannot
    be activated (missing credentials or optional dependency).
    """
    _factories[name] = factory


# ── Registry ──────────────────────────────────────────────────────────────

# Sentinel pushed into a session's event_queue when it stops.
_STOP_SENTINEL = object()

# Type alias for the session factory callable.
# (channel_name: str, chat_id: str) -> AgentSession
_SessionFactory = Callable[[str, str], Awaitable[Any]]


class ChannelRegistry:
    """Manages active channels and per-chat :class:`~leuk.agent.session.AgentSession`s.

    Each unique ``(channel_name, chat_id)`` pair gets its own
    ``AgentSession``.  Responses from the agent are forwarded back to the
    originating channel.
    """

    def __init__(self, session_factory: _SessionFactory, config: Any) -> None:
        """
        Parameters
        ----------
        session_factory:
            Async callable ``(channel_name, chat_id) -> AgentSession``.
            The registry calls this the first time a message arrives from a
            new chat.
        config:
            A :class:`~leuk.config.ChannelsConfig` instance (or any object
            with per-channel credential attributes).
        """
        self._session_factory = session_factory
        self._config = config
        self._channels: dict[str, Channel] = {}
        # (channel_name, chat_id) -> AgentSession
        self._sessions: dict[tuple[str, str], Any] = {}
        self._forward_tasks: list[asyncio.Task[Any]] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Import channel modules (triggering self-registration), then
        instantiate and connect every enabled channel."""
        _import_channels()

        for name, factory in _factories.items():
            channel = factory(self._config)
            if channel is None:
                logger.debug("Channel %r skipped (no credentials or optional dependency missing)", name)
                continue

            channel.on_message(self._handle_message)
            try:
                await channel.connect()
                self._channels[name] = channel
                logger.info("Channel %r connected", name)
            except Exception as exc:
                logger.warning("Channel %r failed to connect: %s", name, exc)

    async def stop(self) -> None:
        """Stop all agent sessions and disconnect all channels."""
        for session in list(self._sessions.values()):
            try:
                await session.stop()
            except Exception:
                pass
        self._sessions.clear()

        # Cancel forwarding tasks
        for task in self._forward_tasks:
            if not task.done():
                task.cancel()
        if self._forward_tasks:
            await asyncio.gather(*self._forward_tasks, return_exceptions=True)
        self._forward_tasks.clear()

        for name, channel in list(self._channels.items()):
            try:
                await channel.disconnect()
                logger.debug("Channel %r disconnected", name)
            except Exception:
                pass
        self._channels.clear()

    # ── Routing ───────────────────────────────────────────────────────────

    async def _handle_message(self, msg: ChannelMessage) -> None:
        """Route an incoming :class:`ChannelMessage` to the right session."""
        key = (msg.channel, msg.chat_id)
        session = self._sessions.get(key)

        if session is None:
            session = await self._session_factory(msg.channel, msg.chat_id)
            session.start()
            self._sessions[key] = session
            task = asyncio.create_task(
                self._forward_events(msg.channel, msg.chat_id, session),
                name=f"channel-fwd-{msg.channel}-{msg.chat_id}",
            )
            self._forward_tasks.append(task)

        session.push(msg.text)

    async def _forward_events(
        self, channel_name: str, chat_id: str, session: Any
    ) -> None:
        """Consume events from an AgentSession and send text replies back.

        Text deltas are accumulated until a TURN_COMPLETE event arrives,
        then the full response is sent as a single message.
        """
        from leuk.types import AgentState, StreamEvent, StreamEventType

        channel = self._channels.get(channel_name)
        if channel is None:
            return

        response_parts: list[str] = []

        while True:
            try:
                event = await session.event_queue.get()
            except Exception:
                break

            # Stop sentinel from AgentSession.stop()
            if event is _STOP_SENTINEL or (
                hasattr(event, "__class__") and event.__class__.__name__ == "object"
            ):
                # Check against the AgentSession's own sentinel by inspecting state
                if session.state == AgentState.STOPPED:
                    break
                # It's an unknown object — ignore and continue
                continue

            if not isinstance(event, StreamEvent):
                continue

            if event.type == StreamEventType.TEXT_DELTA and event.content:
                response_parts.append(event.content)
            elif event.type == StreamEventType.TURN_COMPLETE:
                if response_parts:
                    text = "".join(response_parts)
                    response_parts.clear()
                    try:
                        await channel.send(chat_id, text)
                    except Exception as exc:
                        logger.error(
                            "Failed to send response on channel %r chat %r: %s",
                            channel_name,
                            chat_id,
                            exc,
                        )
                if session.state == AgentState.STOPPED:
                    break
            elif event.type == StreamEventType.STATE_CHANGE:
                if event.content == AgentState.STOPPED:
                    break

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def active_channels(self) -> list[str]:
        """Names of currently connected channels."""
        return list(self._channels.keys())

    @property
    def session_count(self) -> int:
        """Number of active per-chat agent sessions."""
        return len(self._sessions)


# ── Barrel import ─────────────────────────────────────────────────────────


def _import_channels() -> None:
    """Import all channel sub-modules so they self-register.

    The REPL channel has no optional dependencies and is always imported.
    Telegram, Slack, and Discord are wrapped in try/except so a missing
    library never prevents the others from loading.
    """
    from leuk.channels import repl as _repl  # noqa: F401

    try:
        from leuk.channels import telegram as _telegram  # noqa: F401
    except ImportError:
        logger.debug("Telegram channel unavailable (aiogram not installed)")

    try:
        from leuk.channels import slack as _slack  # noqa: F401
    except ImportError:
        logger.debug("Slack channel unavailable (slack-bolt not installed)")

    try:
        from leuk.channels import discord as _discord  # noqa: F401
    except ImportError:
        logger.debug("Discord channel unavailable (discord.py not installed)")


__all__ = [
    "Channel",
    "ChannelMessage",
    "ChannelRegistry",
    "MessageCallback",
    "register_channel",
]
