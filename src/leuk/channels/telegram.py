"""Telegram channel via aiogram (v3).

Optional dependency: ``pip install aiogram``

Self-registers as ``"telegram"`` at import time.  The factory returns
``None`` if ``aiogram`` is not installed or ``telegram_bot_token`` is
absent from the config.
"""

from __future__ import annotations

import logging
from typing import Any

from leuk.channels.base import ChannelMessage, MessageCallback
from leuk.channels import register_channel

logger = logging.getLogger(__name__)

_CHANNEL_NAME = "telegram"

try:
    from aiogram import Bot, Dispatcher
    from aiogram.client.default import DefaultBotProperties
    from aiogram.enums import ParseMode
    from aiogram.types import Message as TelegramMessage

    _AIOGRAM_AVAILABLE = True
except ImportError:
    _AIOGRAM_AVAILABLE = False


class TelegramChannel:
    """Telegram Bot API channel backed by aiogram v3.

    Each Telegram chat_id maps to its own AgentSession in the registry.
    """

    name = _CHANNEL_NAME

    def __init__(self, token: str) -> None:
        self._token = token
        self._callback: MessageCallback | None = None
        self._bot: Any = None
        self._dp: Any = None
        self._polling_task: Any = None

    # ── Channel protocol ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create Bot + Dispatcher, register handler, start long-polling."""
        from aiogram import Bot, Dispatcher
        from aiogram.client.default import DefaultBotProperties
        from aiogram.enums import ParseMode

        self._bot = Bot(
            token=self._token,
            default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
        )
        self._dp = Dispatcher()

        # Register message handler
        @self._dp.message()
        async def _on_message(message: TelegramMessage) -> None:
            if message.text and self._callback is not None:
                msg = ChannelMessage(
                    text=message.text,
                    chat_id=str(message.chat.id),
                    sender=str(message.from_user.id) if message.from_user else "unknown",
                    channel=_CHANNEL_NAME,
                    metadata={
                        "username": getattr(message.from_user, "username", None),
                        "first_name": getattr(message.from_user, "first_name", None),
                    },
                )
                await self._callback(msg)

        import asyncio

        self._polling_task = asyncio.create_task(
            self._dp.start_polling(self._bot, handle_signals=False),
            name="telegram-polling",
        )
        logger.info("Telegram polling started")

    async def send(self, chat_id: str, text: str) -> None:
        """Send *text* to the Telegram chat identified by *chat_id*."""
        if self._bot is None:
            raise RuntimeError("TelegramChannel not connected")
        # Telegram has a 4096-char message limit; split if needed.
        limit = 4096
        for i in range(0, len(text), limit):
            await self._bot.send_message(chat_id=int(chat_id), text=text[i : i + limit])

    def on_message(self, callback: MessageCallback) -> None:
        self._callback = callback

    async def disconnect(self) -> None:
        """Stop polling and close the Bot session."""
        import asyncio

        if self._polling_task and not self._polling_task.done():
            self._polling_task.cancel()
            try:
                await self._polling_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._dp is not None:
            await self._dp.stop_polling()
        if self._bot is not None:
            await self._bot.session.close()
            self._bot = None


# ── Self-registration ─────────────────────────────────────────────────────


def _make_telegram(config: Any) -> TelegramChannel | None:
    """Factory: return a TelegramChannel if aiogram is installed and token is set."""
    if not _AIOGRAM_AVAILABLE:
        return None
    token: str = getattr(config, "telegram_bot_token", "")
    if not token:
        return None
    return TelegramChannel(token)


register_channel(_CHANNEL_NAME, _make_telegram)
