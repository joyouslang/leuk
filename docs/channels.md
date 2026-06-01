[Home](README.md) › Channels

# Channels

Channels let you talk to leuk from chat apps. The `ChannelRegistry`
(`src/leuk/channels/__init__.py`) routes incoming messages to per-chat
`AgentSession`s and forwards events back.

| Channel | Module | Extra |
|---------|--------|-------|
| Telegram | `telegram.py` | `channels-telegram` (aiogram) |
| Slack | `slack.py` | `channels-slack` (slack-bolt) |
| Discord | `discord.py` | `channels-discord` (discord.py) |
| REPL | `repl.py` | built-in (disabled when the interactive REPL runs) |

Each channel implements the `Channel` protocol (`channels/base.py`):
`connect`, `disconnect`, `send`, and `on_message`. Concrete channels also add a
`request_approval` method for channel-native tool-approval prompts (e.g.
`telegram.py`), but it is not part of the base protocol.

## Setup

Set a bot token (e.g. `LEUK_CHANNELS_TELEGRAM_BOT_TOKEN`, or via `/auth`) and an
**allowlist** of user ids:

```bash
LEUK_CHANNELS_ALLOWED_USERS='["123456789"]'
```

Only allowlisted senders are served (the local REPL user is always exempt).

## Approvals over channels

When a tool needs approval, the request is routed to the originating channel as
inline buttons (`request_approval`) — Allow / Deny / Always. Denial stops the
agent. For [desktop control](tools/input_control.md), Telegram approvals include a
**before** screenshot and an **after** screenshot. See [Safety](safety.md).

## See also

- [Safety & Approvals](safety.md) · [Configuration](configuration.md) · [Architecture Overview](architecture.md)
