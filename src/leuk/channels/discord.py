"""Discord channel via discord.py.

Optional dependency: ``pip install discord.py``

Self-registers as ``"discord"`` at import time.  The factory returns
``None`` if ``discord.py`` is not installed or ``discord_bot_token`` is
absent from config.

The bot requires the ``MESSAGE_CONTENT`` privileged intent to read message
text.  Enable it in the Discord Developer Portal under your application's
Bot settings.
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

_CHANNEL_NAME = "discord"

try:
    import discord

    _DISCORD_AVAILABLE = True
except ImportError:
    _DISCORD_AVAILABLE = False


class DiscordChannel:
    """Discord channel backed by discord.py.

    Each Discord channel (text channel or DM) identified by its channel_id
    maps to its own AgentSession in the registry.
    """

    name = _CHANNEL_NAME

    def __init__(self, token: str, *, approval_timeout: int = 120) -> None:
        self._token = token
        self._approval_timeout = approval_timeout
        self._callback: MessageCallback | None = None
        self._client: Any = None
        self._run_task: asyncio.Task[Any] | None = None
        self._pending_approvals: dict[str, asyncio.Future[ApprovalResult]] = {}

    # ── Channel protocol ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create the discord.py Client, register on_message, and start the bot."""
        import discord

        intents = discord.Intents.default()
        intents.message_content = True  # privileged intent

        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready() -> None:
            logger.info("Discord bot logged in as %s", self._client.user)

        @self._client.event
        async def on_message(message: discord.Message) -> None:
            # Ignore messages from the bot itself
            if message.author == self._client.user:
                return
            text = message.content.strip()
            if not text or self._callback is None:
                return
            msg = ChannelMessage(
                text=text,
                chat_id=str(message.channel.id),
                sender=str(message.author.id),
                channel=_CHANNEL_NAME,
                metadata={
                    "username": str(message.author),
                    "guild_id": str(message.guild.id) if message.guild else None,
                    "message_id": str(message.id),
                },
            )
            await self._callback(msg)

        # Start the bot in a background task (client.start() blocks until stopped)
        self._run_task = asyncio.create_task(
            self._client.start(self._token),
            name="discord-client",
        )

    async def send(self, chat_id: str, text: str) -> None:
        """Send *text* to the Discord channel identified by *chat_id*."""
        if self._client is None:
            raise RuntimeError("DiscordChannel not connected")
        channel = self._client.get_channel(int(chat_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(chat_id))
        # Discord has a 2000-char message limit; split if needed.
        limit = 1990
        for i in range(0, len(text), limit):
            await channel.send(text[i : i + limit])

    def on_message(self, callback: MessageCallback) -> None:
        self._callback = callback

    # ── Interactive approval ──────────────────────────────────────────────

    async def request_approval(
        self, chat_id: str, tool_name: str, tool_args: str, reason: str
    ) -> ApprovalResult:
        """Send a message with discord.ui buttons and wait for response."""
        if self._client is None:
            return ApprovalResult(approved=False)


        channel = self._client.get_channel(int(chat_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(chat_id))

        approval_id = uuid.uuid4().hex[:12]
        future: asyncio.Future[ApprovalResult] = asyncio.get_event_loop().create_future()
        self._pending_approvals[approval_id] = future

        text = (
            f"🔐 **Permission required**\n\n"
            f"`{tool_name}({tool_args})`\n\n"
            f"*{reason}*"
        )

        view = _ApprovalView(approval_id, self._pending_approvals, timeout=self._approval_timeout)
        await channel.send(text, view=view)

        try:
            result = await asyncio.wait_for(future, timeout=self._approval_timeout)
        except asyncio.TimeoutError:
            result = ApprovalResult(approved=False)
            logger.info("Discord approval %s timed out, auto-denying", approval_id)
        finally:
            self._pending_approvals.pop(approval_id, None)

        return result

    async def disconnect(self) -> None:
        """Close the Discord client."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except (asyncio.CancelledError, Exception):
                pass
        self._run_task = None


# ── Approval view (discord.ui) ────────────────────────────────────────────

if _DISCORD_AVAILABLE:
    import discord

    class _ApprovalView(discord.ui.View):
        """Four-button approval view for tool-use confirmation."""

        def __init__(
            self,
            approval_id: str,
            pending: dict[str, asyncio.Future[ApprovalResult]],
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self._approval_id = approval_id
            self._pending = pending

        def _resolve(self, result: ApprovalResult) -> None:
            future = self._pending.get(self._approval_id)
            if future and not future.done():
                future.set_result(result)

        @discord.ui.button(label="✅ Allow", style=discord.ButtonStyle.success)
        async def allow_btn(
            self, interaction: discord.Interaction, button: discord.ui.Button[Any]
        ) -> None:
            self._resolve(ApprovalResult(approved=True))
            await interaction.response.send_message("✅ Allowed", ephemeral=True)
            self.stop()

        @discord.ui.button(label="❌ Deny", style=discord.ButtonStyle.danger)
        async def deny_btn(
            self, interaction: discord.Interaction, button: discord.ui.Button[Any]
        ) -> None:
            self._resolve(ApprovalResult(approved=False))
            await interaction.response.send_message("❌ Denied", ephemeral=True)
            self.stop()

        @discord.ui.button(label="🔒 Always Allow", style=discord.ButtonStyle.secondary)
        async def always_allow_btn(
            self, interaction: discord.Interaction, button: discord.ui.Button[Any]
        ) -> None:
            self._resolve(ApprovalResult(approved=True, remember=True))
            await interaction.response.send_message("🔒 Always allowed (saved)", ephemeral=True)
            self.stop()

        @discord.ui.button(label="🚫 Always Deny", style=discord.ButtonStyle.secondary)
        async def always_deny_btn(
            self, interaction: discord.Interaction, button: discord.ui.Button[Any]
        ) -> None:
            self._resolve(ApprovalResult(approved=False, remember=True))
            await interaction.response.send_message("🚫 Always denied (saved)", ephemeral=True)
            self.stop()


# ── Self-registration ─────────────────────────────────────────────────────


def _make_discord(config: Any) -> DiscordChannel | None:
    """Factory: return a DiscordChannel if discord.py is installed and token is set."""
    if not _DISCORD_AVAILABLE:
        return None
    token: str = getattr(config, "discord_bot_token", "")
    if not token:
        return None
    return DiscordChannel(token)


register_channel(_CHANNEL_NAME, _make_discord)
