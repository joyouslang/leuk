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

# Channels whose sender is the local machine user — always exempt from the
# remote-user allowlist (refactor-plan §3.6).
_LOCAL_CHANNELS = frozenset({"repl", "pipe"})

# Type alias for the session factory callable.
# (channel_name: str, chat_id: str, channel: Channel) -> AgentSession
_SessionFactory = Callable[[str, str, Any], Awaitable[Any]]


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
        self._allowed_users: set[str] = set(getattr(config, "allowed_users", []))
        self._channels: dict[str, Channel] = {}
        # (channel_name, chat_id) -> AgentSession
        self._sessions: dict[tuple[str, str], Any] = {}
        # Completed tasks remove themselves (add_done_callback) so this never
        # grows unbounded across many chat sessions.
        self._forward_tasks: set[asyncio.Task[Any]] = set()

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
        tasks = list(self._forward_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
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
        from leuk.types import AgentState

        # Allowlist check — local channels (REPL/pipe) are always exempt.
        if (
            self._allowed_users
            and msg.channel not in _LOCAL_CHANNELS
            and msg.sender not in self._allowed_users
        ):
            logger.debug(
                "Dropping message from unlisted sender %r on channel %r",
                msg.sender,
                msg.channel,
            )
            return

        key = (msg.channel, msg.chat_id)
        session = self._sessions.get(key)

        if session is None:
            channel = self._channels.get(msg.channel)
            session = await self._session_factory(msg.channel, msg.chat_id, channel)
            session.start()
            self._sessions[key] = session
            task = asyncio.create_task(
                self._forward_events(msg.channel, msg.chat_id, session),
                name=f"channel-fwd-{msg.channel}-{msg.chat_id}",
            )
            self._forward_tasks.add(task)
            task.add_done_callback(self._forward_tasks.discard)
            was_idle = True
        else:
            # Only acknowledge if the agent wasn't already working — a burst of
            # messages to a busy agent gets a single ack, not one per message
            # (refactor-plan §4.5).
            was_idle = getattr(session, "state", AgentState.IDLE) == AgentState.IDLE

        session.push(msg.text)

        if was_idle:
            await self._acknowledge(msg.channel, msg.chat_id)

    async def _acknowledge(self, channel_name: str, chat_id: str) -> None:
        """Tell the user the agent is working — prefer a native typing
        indicator (auto-expiring, no message spam) and fall back to a one-off
        text ack for channels that don't support one (refactor-plan §4.5/§6.1)."""
        channel = self._channels.get(channel_name)
        if channel is None:
            return
        notify_typing = getattr(channel, "notify_typing", None)
        try:
            if callable(notify_typing):
                await notify_typing(chat_id)
            else:
                await channel.send(chat_id, "⏳ Working on it...")
        except Exception:
            pass

    async def _forward_events(
        self, channel_name: str, chat_id: str, session: Any
    ) -> None:
        """Consume events from an AgentSession and relay the reply back.

        Text deltas are accumulated as the turn streams. If the channel exposes
        an ``edit`` method (e.g. Telegram), the reply is sent once and then
        *edited in place* as more text arrives (debounced), so the user sees a
        single, growing message instead of an ack followed by a separate reply
        (refactor-plan §6.1/§6.2). Channels without ``edit`` get one message at
        ``TURN_COMPLETE`` — the original behaviour.
        """
        import time

        from leuk.types import AgentState, StreamEvent, StreamEventType

        channel = self._channels.get(channel_name)
        if channel is None:
            return

        editor: Any = getattr(channel, "edit", None)
        can_edit = callable(editor)

        response_parts: list[str] = []
        live_msg_id: Any = None  # message being edited in place (edit-capable channels)
        last_edit = 0.0
        debounce = 1.0  # seconds between in-place edits, to respect rate limits

        async def _flush_live() -> None:
            """Create or update the in-place reply with the text so far."""
            nonlocal live_msg_id, last_edit
            text = "".join(response_parts)
            if not text.strip():
                return
            try:
                if live_msg_id is None:
                    live_msg_id = await channel.send(chat_id, text)
                else:
                    await editor(chat_id, live_msg_id, text)
                last_edit = time.monotonic()
            except Exception:
                logger.debug("In-place edit failed on %r", channel_name, exc_info=True)

        while True:
            try:
                event = await session.event_queue.get()
            except asyncio.CancelledError:
                raise
            except Exception:
                break

            # Stop sentinel from AgentSession.stop() — only ever put after the
            # loop has stopped, so always break (checked by identity).
            if event is _STOP_SENTINEL:
                break

            if not isinstance(event, StreamEvent):
                continue

            if event.type == StreamEventType.TEXT_DELTA and event.content:
                response_parts.append(event.content)
                if can_edit and (time.monotonic() - last_edit) >= debounce:
                    await _flush_live()
            elif event.type == StreamEventType.TOOL_CALL_START and event.tool_call:
                # Notify the channel user that a tool is running.
                tc = event.tool_call
                try:
                    await channel.send(chat_id, f"🔧 Running `{tc.name}`...")
                except Exception:
                    pass
            elif event.type == StreamEventType.TURN_COMPLETE:
                if response_parts:
                    text = "".join(response_parts)
                    try:
                        if can_edit and live_msg_id is not None:
                            await editor(chat_id, live_msg_id, text)
                        else:
                            await channel.send(chat_id, text)
                    except Exception as exc:
                        logger.error(
                            "Failed to send response on channel %r chat %r: %s",
                            channel_name,
                            chat_id,
                            exc,
                        )
                response_parts.clear()
                live_msg_id = None
                last_edit = 0.0
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


# Sub-modules that are infrastructure, not channels — never auto-imported.
_NON_CHANNEL_MODULES = frozenset({"base", "markdown"})


def _import_channels() -> None:
    """Import every channel sub-module so they self-register.

    Discovered dynamically via :func:`pkgutil.iter_modules` over this package
    (refactor-plan §4.4) — adding ``channels/<new>.py`` that calls
    :func:`register_channel` is enough; no edit here is needed. Each import is
    wrapped in ``try/except ImportError`` so a missing optional dependency
    (aiogram, slack-bolt, discord.py) never blocks the other channels.
    """
    import importlib
    import pkgutil

    for mod in pkgutil.iter_modules(__path__):
        if mod.name in _NON_CHANNEL_MODULES or mod.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{mod.name}")
        except ImportError as exc:
            logger.debug("Channel module %r unavailable: %s", mod.name, exc)


__all__ = [
    "Channel",
    "ChannelMessage",
    "ChannelRegistry",
    "MessageCallback",
    "register_channel",
]
