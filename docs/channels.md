[Home](README.md) › Channels

# Channels

Channels let you talk to leuk from chat apps. The `ChannelRegistry`
(`src/leuk/channels/__init__.py`) routes incoming messages to per-chat
`AgentSession`s and forwards events back. Channel modules are **auto-discovered**
(`pkgutil.iter_modules` over the `channels/` package) — dropping in a new
`channels/<name>.py` that calls `register_channel(...)` is enough; missing
optional dependencies are skipped silently.

| Channel | Module | Extra |
|---------|--------|-------|
| Telegram | `telegram.py` | `channels-telegram` (aiogram) |
| Slack | `slack.py` | `channels-slack` (slack-bolt) |
| Discord | `discord.py` | `channels-discord` (discord.py) |
| Pipe | `pipe.py` | built-in (non-interactive stdin/stdout) |

Each channel implements the `Channel` protocol (`channels/base.py`):
`connect`, `disconnect`, `send`, and `on_message`. Channels may **optionally**
add `edit(chat_id, message_id, text)` (edit a reply in place — enables streaming)
and `notify_typing(chat_id)` (native typing indicator); the registry uses them
when present. Concrete channels also add a `request_approval` method for
channel-native tool-approval prompts (e.g. `telegram.py`), but it is not part of
the base protocol.

### Pipe channel (non-interactive)

The `pipe` channel wraps stdin/stdout for scripted/CI use and activates **only
when stdin is not a TTY** (so it never races the interactive REPL for input):

```bash
echo "summarise this repo" | leuk
```

Disable it with `channels.pipe_enabled = false`. The `pipe` channel is local, so
it is exempt from the allowlist.

## Replies, typing, and edit-in-place

- On Telegram, replies render through a **Markdown→HTML** converter
  (`channels/markdown.py`): text is HTML-escaped first, then a safe subset
  (bold/italic/strikethrough/code/links/headings) becomes Telegram HTML tags, so
  special characters never silently break a message.
- On Slack, the same subset converts to Slack's **mrkdwn** dialect
  (`*bold*`, `_italic_`, `~strike~`, `<url|label>` links; headings → bold) —
  Slack does not render standard Markdown. Discord renders Markdown natively,
  so its channel sends replies unconverted.
- The registry acknowledges a new turn only when the agent was **idle** — a
  burst of messages to a busy agent does not spam acks. When the channel supports
  it (Telegram), the ack is a native **typing indicator** rather than a text
  message.
- Edit-capable channels (Telegram) receive a single reply that is **edited in
  place** as the turn streams (debounced), instead of an ack followed by a
  separate reply.

## Setup

Set a bot token (e.g. `LEUK_CHANNELS_TELEGRAM_BOT_TOKEN`, or via `/auth`) and an
**allowlist** of user ids:

```bash
LEUK_CHANNELS_ALLOWED_USERS='["123456789"]'
```

Only allowlisted senders are served (local channels — `pipe` — are always exempt).

## Approvals over channels

When a tool needs approval, the request is routed to the originating channel as
inline buttons (`request_approval`) — Allow / Deny / Always. Denial stops the
agent. For [desktop control](tools/input_control.md), Telegram approvals include a
**before** screenshot and an **after** screenshot. See [Safety](safety.md).

## See also

- [Safety & Approvals](safety.md) · [Configuration](configuration.md) · [Architecture Overview](architecture.md)
