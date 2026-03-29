"""Slack channel via slack-bolt (Socket Mode).

Optional dependency: ``pip install slack-bolt``

Self-registers as ``"slack"`` at import time.  The factory returns
``None`` if ``slack-bolt`` is not installed or the required tokens
(``slack_bot_token`` + ``slack_app_token``) are absent from config.

Socket Mode requires a Slack app with Socket Mode enabled and an
App-Level Token (xapp-…) in addition to the Bot Token (xoxb-…).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from leuk.channels.base import ChannelMessage, MessageCallback
from leuk.channels import register_channel
from leuk.safety import ApprovalResult

logger = logging.getLogger(__name__)

_CHANNEL_NAME = "slack"

try:
    from slack_bolt.async_app import AsyncApp  # noqa: F401
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler  # noqa: F401

    _SLACK_AVAILABLE = True
except ImportError:
    _SLACK_AVAILABLE = False


class SlackChannel:
    """Slack Socket Mode channel backed by slack-bolt.

    Each Slack channel/DM (identified by its channel_id) maps to its own
    AgentSession in the registry.
    """

    name = _CHANNEL_NAME

    def __init__(
        self, bot_token: str, app_token: str, *, approval_timeout: int = 120
    ) -> None:
        self._bot_token = bot_token
        self._app_token = app_token
        self._approval_timeout = approval_timeout
        self._callback: MessageCallback | None = None
        self._app: Any = None
        self._handler: Any = None
        self._pending_approvals: dict[str, asyncio.Future[ApprovalResult]] = {}

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

        # Approval button handlers
        @self._app.action("leuk_allow")
        @self._app.action("leuk_deny")
        @self._app.action("leuk_always_allow")
        @self._app.action("leuk_always_deny")
        async def _on_approval_action(ack: Any, body: dict[str, Any]) -> None:
            await ack()
            await self._handle_approval_action(body)

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

    # ── Interactive approval ──────────────────────────────────────────────

    async def request_approval(
        self, chat_id: str, tool_name: str, tool_args: str, reason: str
    ) -> ApprovalResult:
        """Send a Block Kit message with approval buttons and wait for response."""
        if self._app is None:
            return ApprovalResult(approved=False)

        approval_id = uuid.uuid4().hex[:12]
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"🔐 *Permission required*\n\n"
                        f"`{tool_name}({tool_args})`\n\n"
                        f"_{reason}_"
                    ),
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Allow"},
                        "action_id": "leuk_allow",
                        "value": f"{approval_id}:allow",
                        "style": "primary",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "❌ Deny"},
                        "action_id": "leuk_deny",
                        "value": f"{approval_id}:deny",
                        "style": "danger",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "🔒 Always Allow"},
                        "action_id": "leuk_always_allow",
                        "value": f"{approval_id}:always_allow",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "🚫 Always Deny"},
                        "action_id": "leuk_always_deny",
                        "value": f"{approval_id}:always_deny",
                    },
                ],
            },
        ]

        future: asyncio.Future[ApprovalResult] = asyncio.get_event_loop().create_future()
        self._pending_approvals[approval_id] = future

        await self._app.client.chat_postMessage(
            channel=chat_id, text="Permission required", blocks=blocks
        )

        try:
            result = await asyncio.wait_for(future, timeout=self._approval_timeout)
        except asyncio.TimeoutError:
            result = ApprovalResult(approved=False)
            logger.info("Slack approval %s timed out, auto-denying", approval_id)
        finally:
            self._pending_approvals.pop(approval_id, None)

        return result

    async def _handle_approval_action(self, body: dict[str, Any]) -> None:
        """Resolve a pending approval future from a Slack button press."""
        actions = body.get("actions", [])
        if not actions:
            return
        value: str = actions[0].get("value", "")
        parts = value.split(":", 1)
        if len(parts) != 2:
            return
        approval_id, action = parts

        future = self._pending_approvals.get(approval_id)
        if future is None or future.done():
            return

        if action == "allow":
            future.set_result(ApprovalResult(approved=True))
        elif action == "deny":
            future.set_result(ApprovalResult(approved=False))
        elif action == "always_allow":
            future.set_result(ApprovalResult(approved=True, remember=True))
        elif action == "always_deny":
            future.set_result(ApprovalResult(approved=False, remember=True))

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
