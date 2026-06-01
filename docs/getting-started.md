[Home](README.md) › Getting Started

# Getting Started

## Requirements

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- Optional system binaries for some features: `ydotool`/`grim` (desktop control),
  `playwright` browsers (browser tool).

## Install

```bash
git clone https://github.com/joyouslang/leuk.git
cd leuk

uv sync                      # core
uv sync --extra voice        # + speech-to-text / text-to-speech (pulls PyTorch)
uv sync --extra browser      # + Playwright browser tool
uv sync --extra input-control# + desktop keyboard/mouse (needs ydotool, see below)
uv sync --extra channels     # + Telegram/Slack/Discord
uv sync --group dev          # + pytest, ruff, mypy
```

### Optional dependency extras

| Extra | Enables | Notes |
|-------|---------|-------|
| `voice` | STT/TTS/VAD | PyTorch, transformers, sounddevice, omegaconf |
| `browser` | [Browser tool](tools/browser.md) | also run `playwright install chromium` |
| `input-control` | [Desktop control](tools/input_control.md) | needs system `ydotool` (+`grim` on Wayland) |
| `channels-telegram` / `-slack` / `-discord` / `channels` | [Channels](channels.md) | bot SDKs |

> Several features also need **system binaries** (ydotool, grim, Chromium).
> **Run [`leuk doctor`](reference/system-dependencies.md)** — it checks each
> optional feature for your distro/session and prints the exact setup commands
> and how to enable it. For desktop control, `bash scripts/setup-input-control.sh`
> does the whole setup in one shot. See [System Dependencies](reference/system-dependencies.md).

## First run

```bash
uv run leuk            # or: uv run python -m leuk
```

leuk starts with **no active session** and a one-line hint. A session is created
when you send your first message or pick one with `/sessions` (see
[Sessions](sessions-and-persistence.md)). On first launch you'll be warned that
no credentials are configured — run `/auth`.

The default provider is **OpenCode Zen** (`big-pickle`), which offers free access
with no API key.

## Authentication

Run `/auth` and choose **add**. See [Providers](providers.md) for per-provider
details.

### Claude Pro / Max (OAuth, no API key)

```
leuk> /auth → add → Anthropic → "Claude Pro/Max subscription (OAuth login)"
```

A browser window opens; log in, authorize, and paste the code back. The OAuth
token is stored in `~/.config/leuk/credentials.json` (mode `0600`) and refreshed
automatically on `401`.

### API key

```
leuk> /auth → add → <provider> → "Enter API key manually"
```

Keys can also be set via environment variables — see
[Environment Variables](reference/environment.md).

## Next steps

- Learn the prompt and commands: [REPL & Commands](repl-commands.md)
- Configure behavior: [Configuration](configuration.md)
- Understand how it works: [Architecture Overview](architecture.md)

## See also

- [Configuration](configuration.md) · [Providers](providers.md) · [Development](development.md)
