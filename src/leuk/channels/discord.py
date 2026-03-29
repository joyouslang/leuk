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
from typing import Any

from leuk.channels.base import ChannelMessage, MessageCallback
from leuk.channels import register_channel

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

    def __init__(self, token: str) -> None:
        self._token = token
        self._callback: MessageCallback | None = None
        self._client: Any = None
        self._run_task: asyncio.Task[Any] | None = None

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
