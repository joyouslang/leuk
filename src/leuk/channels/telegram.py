"""Telegram channel via aiogram (v3).

Optional dependency: ``pip install aiogram``

Self-registers as ``"telegram"`` at import time.  The factory returns
``None`` if ``aiogram`` is not installed or ``telegram_bot_token`` is
absent from the config.
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

_CHANNEL_NAME = "telegram"
_APPROVAL_TIMEOUT = 120  # seconds; overridden via config at runtime

try:
    from aiogram import Bot, Dispatcher  # noqa: F401
    from aiogram.client.default import DefaultBotProperties  # noqa: F401
    from aiogram.enums import ParseMode  # noqa: F401
    from aiogram.types import (
        CallbackQuery,
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        Message as TelegramMessage,
    )

    _AIOGRAM_AVAILABLE = True
except ImportError:
    _AIOGRAM_AVAILABLE = False


class TelegramChannel:
    """Telegram Bot API channel backed by aiogram v3.

    Each Telegram chat_id maps to its own AgentSession in the registry.
    """

    name = _CHANNEL_NAME

    def __init__(self, token: str, *, approval_timeout: int = _APPROVAL_TIMEOUT) -> None:
        self._token = token
        self._approval_timeout = approval_timeout
        self._callback: MessageCallback | None = None
        self._bot: Any = None
        self._dp: Any = None
        self._polling_task: Any = None
        # Pending approval futures keyed by unique approval ID.
        self._pending_approvals: dict[str, asyncio.Future[ApprovalResult]] = {}

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

        # Register callback-query handler for approval buttons
        @self._dp.callback_query()
        async def _on_callback(query: CallbackQuery) -> None:
            await self._handle_approval_callback(query)

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

    # ── Interactive approval ──────────────────────────────────────────────

    async def request_approval(
        self, chat_id: str, tool_name: str, tool_args: str, reason: str
    ) -> ApprovalResult:
        """Send an approval request with inline buttons and wait for response."""
        if self._bot is None:
            return ApprovalResult(approved=False)

        approval_id = uuid.uuid4().hex[:12]
        text = (
            f"🔐 *Permission required*\n\n"
            f"`{tool_name}({tool_args})`\n\n"
            f"_{reason}_"
        )
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="✅ Allow", callback_data=f"approve:{approval_id}:allow"
                    ),
                    InlineKeyboardButton(
                        text="❌ Deny", callback_data=f"approve:{approval_id}:deny"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="🔒 Always Allow",
                        callback_data=f"approve:{approval_id}:always_allow",
                    ),
                    InlineKeyboardButton(
                        text="🚫 Always Deny",
                        callback_data=f"approve:{approval_id}:always_deny",
                    ),
                ],
            ]
        )

        future: asyncio.Future[ApprovalResult] = asyncio.get_event_loop().create_future()
        self._pending_approvals[approval_id] = future

        sent_msg = await self._bot.send_message(
            chat_id=int(chat_id), text=text, reply_markup=keyboard
        )

        try:
            result = await asyncio.wait_for(future, timeout=self._approval_timeout)
        except asyncio.TimeoutError:
            result = ApprovalResult(approved=False)
            logger.info("Approval %s timed out, auto-denying", approval_id)
        finally:
            self._pending_approvals.pop(approval_id, None)

        # Edit the original message to show the outcome
        outcome = "✅ Allowed" if result.approved else "❌ Denied"
        if result.remember:
            outcome += " (saved)"
        try:
            await self._bot.edit_message_text(
                chat_id=int(chat_id),
                message_id=sent_msg.message_id,
                text=f"{text}\n\n*{outcome}*",
            )
        except Exception:
            pass  # Best-effort edit

        return result

    async def _handle_approval_callback(self, query: CallbackQuery) -> None:
        """Resolve a pending approval future from an inline button press."""
        data = query.data or ""
        if not data.startswith("approve:"):
            return

        parts = data.split(":", 2)
        if len(parts) != 3:
            return

        _, approval_id, action = parts
        future = self._pending_approvals.get(approval_id)
        if future is None or future.done():
            await query.answer("This approval has expired.")
            return

        if action == "allow":
            future.set_result(ApprovalResult(approved=True))
        elif action == "deny":
            future.set_result(ApprovalResult(approved=False))
        elif action == "always_allow":
            future.set_result(ApprovalResult(approved=True, remember=True))
        elif action == "always_deny":
            future.set_result(ApprovalResult(approved=False, remember=True))
        else:
            await query.answer("Unknown action.")
            return

        await query.answer("✓")


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
