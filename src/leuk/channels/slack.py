"""Slack channel via slack-bolt (Socket Mode).

Optional dependency: ``pip install slack-bolt``

Self-registers as ``"slack"`` at import time.  The factory returns
``None`` if ``slack-bolt`` is not installed or the required tokens
(``slack_bot_token`` + ``slack_app_token``) are absent from config.

Socket Mode requires a Slack app with Socket Mode enabled and an
App-Level Token (xapp-…) in addition to the Bot Token (xoxb-…).
"""

from __future__ import annotations

import logging
from typing import Any

from leuk.channels.base import ChannelMessage, MessageCallback
from leuk.channels import register_channel

logger = logging.getLogger(__name__)

_CHANNEL_NAME = "slack"

try:
    from slack_bolt.async_app import AsyncApp
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

    _SLACK_AVAILABLE = True
except ImportError:
    _SLACK_AVAILABLE = False


class SlackChannel:
    """Slack Socket Mode channel backed by slack-bolt.

    Each Slack channel/DM (identified by its channel_id) maps to its own
    AgentSession in the registry.
    """

    name = _CHANNEL_NAME

    def __init__(self, bot_token: str, app_token: str) -> None:
        self._bot_token = bot_token
        self._app_token = app_token
        self._callback: MessageCallback | None = None
        self._app: Any = None
        self._handler: Any = None

    # ── Channel protocol ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create the Bolt app, register event handlers, and start Socket Mode."""
        from slack_bolt.async_app import AsyncApp
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

        self._app = AsyncApp(token=self._bot_token)

        @self._app.event("message")
        async def _on_message(event: dict[str, Any], say: Any) -> None:
            # Ignore bot messages and subtypes (edits, deletes, etc.)
            if event.get("bot_id") or event.get("subtype"):
                return
            text: str = event.get("text", "").strip()
            if not text or self._callback is None:
                return
            msg = ChannelMessage(
                text=text,
                chat_id=event.get("channel", ""),
                sender=event.get("user", "unknown"),
                channel=_CHANNEL_NAME,
                metadata={
                    "ts": event.get("ts"),
                    "thread_ts": event.get("thread_ts"),
                    "team": event.get("team"),
                },
            )
            await self._callback(msg)

        self._handler = AsyncSocketModeHandler(self._app, self._app_token)
        await self._handler.start_async()
        logger.info("Slack Socket Mode connected")

    async def send(self, chat_id: str, text: str) -> None:
        """Post *text* to the Slack channel identified by *chat_id*."""
        if self._app is None:
            raise RuntimeError("SlackChannel not connected")
        # Slack has a ~4000-char text limit per block; split if needed.
        limit = 3000
        for i in range(0, len(text), limit):
            await self._app.client.chat_postMessage(
                channel=chat_id, text=text[i : i + limit]
            )

    def on_message(self, callback: MessageCallback) -> None:
        self._callback = callback

    async def disconnect(self) -> None:
        """Stop the Socket Mode handler."""
        if self._handler is not None:
            try:
                await self._handler.close_async()
            except Exception:
                pass
            self._handler = None
        self._app = None


# ── Self-registration ─────────────────────────────────────────────────────


def _make_slack(config: Any) -> SlackChannel | None:
    """Factory: return a SlackChannel if slack-bolt is installed and tokens are set."""
    if not _SLACK_AVAILABLE:
        return None
    bot_token: str = getattr(config, "slack_bot_token", "")
    app_token: str = getattr(config, "slack_app_token", "")
    if not bot_token or not app_token:
        return None
    return SlackChannel(bot_token, app_token)


register_channel(_CHANNEL_NAME, _make_slack)
