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
from html import escape as _esc
from typing import Any

from leuk.channels.base import ChannelMessage, MessageCallback
from leuk.channels import register_channel
from leuk.channels.markdown import markdown_to_telegram_html, split_for_telegram
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

        # HTML parse mode (not legacy Markdown): we render replies through
        # ``markdown_to_telegram_html`` which HTML-escapes everything first, so
        # special characters and unbalanced emphasis never break a message
        # (refactor-plan §6.3).
        self._bot = Bot(
            token=self._token,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
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

    async def send(self, chat_id: str, text: str) -> int | None:
        """Send *text* (Markdown) to *chat_id*; return the last message id.

        The text is split on line boundaries to respect Telegram's 4096-char
        limit and each chunk is converted to HTML independently (so a chunk is
        always self-contained, balanced markup). The returned message id lets
        the registry edit the reply in place (see :meth:`edit`).
        """
        if self._bot is None:
            raise RuntimeError("TelegramChannel not connected")
        last_id: int | None = None
        for chunk in split_for_telegram(text):
            sent = await self._bot.send_message(
                chat_id=int(chat_id), text=markdown_to_telegram_html(chunk)
            )
            last_id = sent.message_id
        return last_id

    async def edit(self, chat_id: str, message_id: int, text: str) -> None:
        """Edit a previously sent message in place (used for streaming replies).

        Only the first 4096 chars are editable in a single message; longer text
        is truncated for the live view and the full reply lands at turn end via
        the final edit/send path in the registry.
        """
        if self._bot is None:
            return
        chunk = split_for_telegram(text)[0]
        try:
            await self._bot.edit_message_text(
                chat_id=int(chat_id),
                message_id=message_id,
                text=markdown_to_telegram_html(chunk),
            )
        except Exception:
            # "message is not modified" and rate-limit errors are non-fatal.
            logger.debug("Telegram edit_message_text failed", exc_info=True)

    async def notify_typing(self, chat_id: str) -> None:
        """Show Telegram's native 'typing…' indicator (auto-expires ~5s)."""
        if self._bot is None:
            return
        try:
            await self._bot.send_chat_action(chat_id=int(chat_id), action="typing")
        except Exception:
            logger.debug("Telegram send_chat_action failed", exc_info=True)

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
            f"🔐 <b>Permission required</b>\n\n"
            f"<code>{_esc(tool_name)}({_esc(tool_args)})</code>\n\n"
            f"<i>{_esc(reason)}</i>"
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

        future: asyncio.Future[ApprovalResult] = asyncio.get_running_loop().create_future()
        self._pending_approvals[approval_id] = future

        # For desktop control, show a "before" screenshot so the remote user can
        # see the current desktop state they're approving an action against.
        if tool_name == "input_control":
            await self._send_desktop_screenshot(chat_id, caption="🖥 Before action")

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
                text=f"{text}\n\n<b>{_esc(outcome)}</b>",
            )
        except Exception:
            pass  # Best-effort edit

        # After an approved desktop action, show the resulting state.
        if tool_name == "input_control" and result.approved:
            await self._send_desktop_screenshot(chat_id, caption="🖥 After action")

        return result

    async def _send_desktop_screenshot(self, chat_id: str, *, caption: str) -> None:
        """Best-effort: capture the desktop and send it as a photo."""
        if self._bot is None:
            return
        try:
            from aiogram.types import BufferedInputFile

            from leuk.host import capture_png

            png, _reason = await asyncio.to_thread(capture_png)
            if not png:
                return
            await self._bot.send_photo(
                chat_id=int(chat_id),
                photo=BufferedInputFile(png, filename="desktop.png"),
                caption=caption,
            )
        except Exception:  # noqa: BLE001 — screenshots are best-effort
            logger.debug("Could not send desktop screenshot", exc_info=True)

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
