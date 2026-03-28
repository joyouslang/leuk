# Skill: /add-telegram

Add Telegram Bot API support to leuk, allowing the agent to receive and respond to
messages via a Telegram bot.

---

## Prerequisites

1. A Telegram bot token from [@BotFather](https://t.me/botfather) — create a bot and
   copy the token (format: `123456:ABCdef...`).
2. `aiogram` installed: `uv add aiogram`.
3. Phase 1.1 (sub-agent concurrency limits) is recommended but not required.

---

## Step 1 — Add dependency

Edit `pyproject.toml`: add a `[telegram]` optional dependency group.

```toml
[project.optional-dependencies]
telegram = [
    "aiogram>=3.0",
]
```

Then run: `uv sync --extra telegram`

---

## Step 2 — Add credential field to `LLMConfig`

File: `src/leuk/config.py`

In class `LLMConfig`, add after `zen_api_key`:

```python
telegram_bot_token: str = ""
```

Also add a `channels` section to `Settings`. After the `mcp_servers` field, add:

```python
telegram_enabled: bool = Field(
    default=False,
    description="Enable Telegram bot channel",
)
```

Or, more cleanly, add a nested `TelegramConfig` model following the same
`BaseSettings` / `SettingsConfigDict(env_prefix="LEUK_TELEGRAM_")` pattern
used by `LLMConfig`, `SQLiteConfig`, and `AgentConfig`, then add it to `Settings`.

---

## Step 3 — Create the channel module

Create `src/leuk/channels/__init__.py` (empty or with a `ChannelRegistry` stub per
section 3.1 of IMPLEMENTATION_PLAN.md).

Create `src/leuk/channels/telegram.py`:

```python
"""Telegram bot channel for leuk."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

from aiogram import Bot, Dispatcher
from aiogram.types import Message as TGMessage

logger = logging.getLogger(__name__)


class TelegramChannel:
    """Connects a Telegram bot to the leuk agent loop."""

    def __init__(self, token: str) -> None:
        self._bot = Bot(token=token)
        self._dp = Dispatcher()
        self._on_message: Callable[[str, str], Awaitable[None]] | None = None

    def on_message(self, callback: Callable[[str, str], Awaitable[None]]) -> None:
        """Register handler: callback(chat_id, text)."""
        self._on_message = callback

        @self._dp.message()
        async def _handler(msg: TGMessage) -> None:
            if msg.text and self._on_message:
                await self._on_message(str(msg.chat.id), msg.text)

    async def send(self, chat_id: str, text: str) -> None:
        """Send a reply to a Telegram chat."""
        # Telegram has a 4096-char limit per message.
        for chunk in [text[i:i+4096] for i in range(0, len(text), 4096)]:
            await self._bot.send_message(chat_id, chunk)

    async def connect(self) -> None:
        """Start polling (runs until disconnect() is called)."""
        logger.info("Telegram bot polling started")
        await self._dp.start_polling(self._bot)

    async def disconnect(self) -> None:
        await self._dp.stop_polling()
        await self._bot.session.close()
```

---

## Step 4 — Wire into the REPL

File: `src/leuk/cli/repl.py`

In the `run()` function (or wherever the main `asyncio` tasks are started), after
`AgentSession` and `ToolRegistry` are set up, add:

```python
from leuk.channels.telegram import TelegramChannel

if settings.llm.telegram_bot_token:
    tg = TelegramChannel(settings.llm.telegram_bot_token)

    async def _on_tg_message(chat_id: str, text: str) -> None:
        # Each chat gets its own session; for now reuse the main session.
        await agent_session.submit(text)
        # Send final assistant reply back to Telegram.
        # Listen to event_queue for the response or use a callback.

    tg.on_message(_on_tg_message)
    asyncio.create_task(tg.connect(), name="telegram")
```

The full routing (per-chat sessions, response forwarding from `event_queue`) mirrors
the REPL render loop — read `src/leuk/cli/repl.py` `_render_loop` for the pattern.

---

## Step 5 — Store the token

```bash
# Option A: environment variable
export LEUK_LLM_TELEGRAM_BOT_TOKEN="123456:ABCdef..."

# Option B: credentials file
python - <<'EOF'
from leuk.config import load_credentials, save_credentials
creds = load_credentials()
creds["telegram_bot_token"] = "123456:ABCdef..."
save_credentials(creds)
EOF
```

Then update `load_settings()` in `src/leuk/config.py` to overlay
`telegram_bot_token` from credentials, following the same pattern as
`anthropic_api_key`.

---

## Step 6 — Verification

```bash
# Start leuk — the bot should begin polling.
leuk

# In a separate terminal, send a message to your bot via Telegram.
# The agent should respond.

# Check logs for "Telegram bot polling started".
```

---

## Notes

- Each Telegram chat should ideally get its own `AgentSession` (keyed by `chat_id`)
  once the full channels abstraction (section 3.1) is implemented.
- Rate-limit outgoing messages to stay under Telegram's 30 msg/s per bot limit.
- For production use, consider webhook mode instead of polling (`aiogram` supports
  both).
