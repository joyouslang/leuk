"""Pipe channel: non-interactive stdin/stdout for piped / CI usage.

Replaces the old ``ReplChannel`` (refactor-plan §3.6). The interactive REPL
already owns stdin/stdout through ``prompt_toolkit``; a channel that also reads
stdin would race against it. So this channel activates **only when stdin is not
a TTY** — i.e. when leuk is driven by a pipe or a script::

    echo "summarise this repo" | leuk

In an interactive terminal the factory returns ``None`` and the channel never
starts, which removes the need for the REPL to defensively disable it.

The chat_id is always ``"default"`` (one piped conversation). The sender is the
local ``USER`` — the registry treats ``"pipe"`` as a local, allowlist-exempt
channel.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from leuk.channels.base import ChannelMessage, MessageCallback
from leuk.channels import register_channel

_CHAT_ID = "default"
_CHANNEL_NAME = "pipe"


class PipeChannel:
    """Reads lines from piped stdin and writes responses to stdout."""

    name = _CHANNEL_NAME

    def __init__(self) -> None:
        self._callback: MessageCallback | None = None
        self._task: asyncio.Task[Any] | None = None

    # ── Channel protocol ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Start the background stdin-reading loop."""
        self._task = asyncio.create_task(self._read_loop(), name="pipe-channel-read")

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
                line: str = await loop.run_in_executor(None, sys.stdin.readline)
            except (EOFError, OSError):
                break

            if not line:
                break  # EOF — the pipe is closed

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


def _make_pipe(config: Any) -> PipeChannel | None:
    """Factory: return a PipeChannel only for non-interactive stdin.

    Returns ``None`` when stdin is a TTY (the interactive REPL handles input)
    or when ``pipe_enabled`` is turned off in config.
    """
    if not getattr(config, "pipe_enabled", True):
        return None
    try:
        if sys.stdin.isatty():
            return None
    except (ValueError, OSError):
        # stdin may be detached (e.g. under some test harnesses) — treat as
        # non-interactive only if explicitly enabled, otherwise skip.
        return None
    return PipeChannel()


register_channel(_CHANNEL_NAME, _make_pipe)
