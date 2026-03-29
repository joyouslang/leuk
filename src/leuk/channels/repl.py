"""REPL channel: wraps stdin/stdout as a :class:`~leuk.channels.base.Channel`.

This allows the interactive terminal to participate in the same
channel + session routing as Telegram, Slack, etc.

The chat_id is always ``"default"`` (there is only one REPL conversation).
The sender is read from the ``USER`` environment variable or ``"user"``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from leuk.channels.base import ChannelMessage, MessageCallback
from leuk.channels import register_channel

_CHAT_ID = "default"
_CHANNEL_NAME = "repl"


class ReplChannel:
    """Reads text lines from stdin and writes responses to stdout.

    Parameters
    ----------
    prompt:
        The input prompt string shown before each user line.
    """

    name = _CHANNEL_NAME

    def __init__(self, prompt: str = "you> ") -> None:
        self._prompt = prompt
        self._callback: MessageCallback | None = None
        self._task: asyncio.Task[Any] | None = None

    # ── Channel protocol ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Start the background stdin-reading loop."""
        self._task = asyncio.create_task(
            self._read_loop(), name="repl-channel-read"
        )

    async def send(self, chat_id: str, text: str) -> None:
        """Write *text* to stdout (appending a trailing newline if absent)."""
        if not text.endswith("\n"):
            text += "\n"
        sys.stdout.write(text)
        sys.stdout.flush()

    def on_message(self, callback: MessageCallback) -> None:
        self._callback = callback

    async def disconnect(self) -> None:
        """Cancel the stdin loop."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    # ── Internal ──────────────────────────────────────────────────────────

    async def _read_loop(self) -> None:
        loop = asyncio.get_running_loop()
        sender = os.environ.get("USER", "user")

        while True:
            try:
                sys.stdout.write(self._prompt)
                sys.stdout.flush()
                line: str = await loop.run_in_executor(None, sys.stdin.readline)
            except (EOFError, OSError):
                break

            if not line:
                # EOF
                break

            text = line.rstrip("\n")
            if not text:
                continue

            if self._callback is not None:
                msg = ChannelMessage(
                    text=text,
                    chat_id=_CHAT_ID,
                    sender=sender,
                    channel=_CHANNEL_NAME,
                )
                await self._callback(msg)


# ── Self-registration ─────────────────────────────────────────────────────


def _make_repl(config: Any) -> ReplChannel | None:
    """Factory: return a ReplChannel if the REPL channel is enabled."""
    if not getattr(config, "repl_enabled", True):
        return None
    return ReplChannel()


register_channel(_CHANNEL_NAME, _make_repl)
