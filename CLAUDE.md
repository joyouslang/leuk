# leuk — Claude Code Guide

Persistent AI agent with sub-agent orchestration, multi-provider LLM support,
multimodal input, voice, desktop/browser control, and environment access
(shell, file I/O, web fetch, MCP servers).

## Documentation is the ground truth (keep it in sync)

The **[wiki](docs/README.md)** (`docs/`) is the canonical, authoritative
documentation. It must stay in sync with the code: **whenever a change touches
behavior, configuration, commands, tools, providers, or architecture, update the
matching wiki page in the same change.** If a page and the code disagree, fix the
page. This file is a quick agent-facing map; the root `README.md` is a short
landing page. Both defer to the wiki.

Key pages: [Architecture](docs/architecture.md) ·
[Configuration](docs/configuration.md) ·
[Env vars](docs/reference/environment.md) ·
[REPL & Commands](docs/repl-commands.md) ·
[Tools](docs/tools.md) · [Safety](docs/safety.md) ·
[Voice](docs/voice.md) · [Multimodal](docs/multimodal.md) ·
[Development](docs/development.md).

---

## Architecture

```
src/leuk/
├── agent/
│   ├── core.py          # Agent main loop (rounds, tool dispatch, context mgmt)
│   ├── context.py       # Tiered compaction: truncate → mask → summarize → drop
│   ├── session.py       # AgentSession (asyncio task + input/event queues)
│   └── sub_agent.py     # SubAgentManager — spawns and tracks sub-agents
├── cli/
│   ├── repl.py          # Interactive REPL loop + command dispatch; `leuk doctor` entry
│   ├── tui.py           # Persistent-input full-screen TUI (default): TuiRenderer sink + ReplTUI app
│   ├── blocks.py        # Shared scrollback block model + rich→ANSI bridge (TUI + history browser)
│   ├── auth.py          # /auth credential wizard
│   ├── render.py        # StreamRenderer — rich.Live streaming display (classic-prompt fallback)
│   ├── history_browser.py  # Interactive history view: navigate, expand blocks, inline media
│   ├── doctor.py        # `leuk doctor` / `/doctor` — optional-feature setup diagnostics
│   ├── extensions_manager.py  # /skills & /mcp manager (TUI) + `leuk skills`/`leuk mcp` CLI
│   └── settings_dialog.py  # Tabbed /settings UI
├── config.py            # All configuration (pydantic-settings; config.json + env vars)
├── safety.py            # SafetyGuard — rule-based tool permission checks
├── tools/
│   ├── base.py          # Tool protocol + ToolRegistry
│   ├── shell.py         # ShellTool (with optional Docker sandbox)
│   ├── file_read.py     # FileReadTool
│   ├── file_edit.py     # FileEditTool
│   ├── browser.py       # BrowserTool (Playwright; SPA/AJAX-aware, optional)
│   ├── input_control.py # InputControlTool (ydotool keyboard/mouse, optional)
│   ├── monitoring.py    # MonitoringTool — read-only host data (screenshot/geometry/system info)
│   ├── local_llm.py     # LocalLLMTool (Ollama, optional)
│   ├── memory_write.py  # MemoryWriteTool
│   ├── sub_agent.py     # SubAgentTool (tool-facing wrapper)
│   └── web_fetch.py     # WebFetchTool
├── host.py              # Read-only host observation: screen capture + HiDPI scaling + system info (shared)
├── media.py             # Multimodal: parse/serialize [screenshot/image/audio/video] tags, load/open media files
├── media_render.py      # Render media blocks for the history browser (metadata line / ANSI thumbnail)
├── skills/
│   ├── loader.py        # SkillLoader — discover SKILL.md bundles, trust/enable state, importers
│   └── tool.py          # SkillTool — progressive-disclosure skill index + read
├── providers/
│   ├── base.py          # LLMProvider protocol (generate/stream/model_info)
│   ├── model_info.py    # queried model metadata (context window, vision/audio)
│   ├── context_window.py # resolve max context (live query → override → unknown)
│   ├── catalog.py       # create_provider() factory
│   ├── anthropic.py     # Anthropic provider
│   ├── openai.py        # OpenAI provider
│   ├── google.py        # Google Gemini provider
│   ├── openrouter.py    # OpenRouter provider
│   └── zen.py           # Zen provider
├── mcp/
│   ├── client.py        # MCP server client (stdio + SSE)
│   ├── bridge.py        # Bridge MCP tools into ToolRegistry
│   └── registry.py      # Import connectors from registries (MCP registry / ClawHub / URL) → mcp_servers
├── channels/
│   ├── base.py          # Channel protocol + ChannelMessage
│   ├── telegram.py      # Telegram bot channel (aiogram)
│   ├── slack.py         # Slack channel (slack-bolt)
│   ├── discord.py       # Discord channel (discord.py)
│   ├── markdown.py      # Markdown → Telegram-HTML converter
│   └── pipe.py          # Non-interactive stdin/stdout channel (piped/CI use)
├── scheduler/
│   ├── task.py          # ScheduledTask dataclass
│   ├── store.py         # SchedulerStore — SQLite CRUD
│   └── runner.py        # TaskScheduler — background poll loop
├── sandbox/
│   ├── container.py     # ContainerRunner + ContainerSandbox (Docker)
│   └── mount_policy.py  # Bind-mount validation
├── memory/
│   └── loader.py        # Hierarchical memory file loading
├── persistence/
│   ├── base.py          # HotStore protocol
│   ├── memory.py        # In-memory HotStore
│   └── sqlite.py        # SQLiteStore — sessions + messages
├── voice/
│   ├── recorder.py      # Silero VAD + mic capture
│   ├── stt.py           # Whisper speech-to-text
│   └── tts.py           # Silero TTS (interruptible; half-duplex with the mic)
└── types.py             # Message, Role, Session, ToolSpec, StreamEvent
```

---

## Configuration

Settings are loaded by `src/leuk/config.py` `load_settings()` in this order
(highest priority first):

1. Environment variables (`LEUK_LLM_*`, `LEUK_SQLITE_*`, `LEUK_*`) — for CI/Docker/power users
2. `~/.config/leuk/config.json` — **the single config file**, written by `/settings`;
   holds flat keys (last provider/model, feature toggles, voice) **and** nested
   sub-model sections, e.g. `{"llm": {"temperature": 0.2}, "input_control": {"enabled": true}}`
3. `~/.config/leuk/credentials.json` (API keys only, mode 0600)
4. Compiled-in defaults

A legacy `config.env` is auto-migrated into `config.json` on first run
(`migrate_legacy_config_env`). Settings can be configured entirely from
`/settings` / `config.json` — no need to export env vars into your shell.

Key settings (env var ↔ the same `config.json` field):

| Env var | Default | Description |
|---------|---------|-------------|
| `LEUK_LLM_PROVIDER` | `zen` | LLM provider: `anthropic`, `openai`, `google`, `openrouter`, `local`, `zen` |
| `LEUK_LLM_MODEL` | `big-pickle` | Model identifier |
| `LEUK_LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `LEUK_LLM_MAX_TOKENS` | `16384` | Max tokens per LLM call |
| `LEUK_MAX_TOOL_ROUNDS` | `50` | Max consecutive tool-use rounds |
| `LEUK_MAX_CONTEXT_TOKENS` | *(auto)* | Compaction-budget override; default derives from the model's queried context window |
| `LEUK_SQLITE_PATH` | `~/.config/leuk/leuk.db` | SQLite database path |
| `LEUK_CHANNELS_TELEGRAM_BOT_TOKEN` | — | Telegram bot token |
| `LEUK_CHANNELS_ALLOWED_USERS` | `[]` | JSON list of allowed user IDs |
| `LEUK_SCHEDULER_ENABLED` | `false` | Enable background task scheduler |
| `LEUK_LOCAL_LLM_ENABLED` | `false` | Enable the local_llm tool (Ollama) |
| `LEUK_INPUT_CONTROL_ENABLED` | `false` | Enable the input_control tool (ydotool; X11+Wayland) |
| `LEUK_INPUT_CONTROL_AUTO_APPROVE` | `false` | Auto-approve desktop control (DANGEROUS) |

---

## Running

```bash
# Install
uv sync

# Start the REPL
leuk

# With a specific provider
LEUK_LLM_PROVIDER=anthropic LEUK_LLM_MODEL=claude-sonnet-4-5-20251022 leuk

# Run tests
python -m pytest tests/
```

---

## Adding tools

1. Create `src/leuk/tools/<name>.py` implementing the `Tool` protocol
   (`src/leuk/tools/base.py`): `spec` property returning a `ToolSpec`, and
   `async execute(arguments) -> str`.
2. Register in `src/leuk/tools/__init__.py` `create_default_registry()`.
3. Add a `ToolRule` in `src/leuk/config.py` `_default_safety_rules()` if the tool
   needs custom permissions.

---

## Testing

```bash
python -m pytest tests/
python -m pytest tests/ -v          # verbose
python -m pytest tests/test_foo.py  # single file
```

Tests use `pytest-asyncio` with `asyncio_mode = "auto"` (see `pyproject.toml`).

---

## Code style

- Python 3.13+, `from __future__ import annotations` in every file
- `ruff` for linting/formatting (`uv run ruff check .`, `uv run ruff format .`)
- `mypy` for type checking (`uv run mypy src/`)
- Line length: 100 characters
