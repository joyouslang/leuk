# leuk — Claude Code Guide

Persistent AI agent with sub-agent orchestration, multi-provider LLM support,
and environment access (shell, file I/O, web fetch, MCP servers).

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
│   ├── repl.py          # Interactive REPL (prompt_toolkit + rich)
│   ├── auth.py          # /auth credential wizard
│   ├── render.py        # StreamRenderer — live streaming display
│   └── settings_dialog.py  # Tabbed /settings UI
├── config.py            # All configuration (pydantic-settings, env vars, config.env)
├── safety.py            # SafetyGuard — rule-based tool permission checks
├── tools/
│   ├── base.py          # Tool protocol + ToolRegistry
│   ├── shell.py         # ShellTool (with optional Docker sandbox)
│   ├── file_read.py     # FileReadTool
│   ├── file_edit.py     # FileEditTool
│   ├── browser.py       # BrowserTool (Playwright, optional)
│   ├── local_llm.py     # LocalLLMTool (Ollama, optional)
│   ├── memory_write.py  # MemoryWriteTool
│   ├── sub_agent.py     # SubAgentTool (tool-facing wrapper)
│   └── web_fetch.py     # WebFetchTool
├── providers/
│   ├── base.py          # LLMProvider protocol
│   ├── catalog.py       # create_provider() factory
│   ├── anthropic.py     # Anthropic provider
│   ├── openai.py        # OpenAI provider
│   ├── google.py        # Google Gemini provider
│   ├── openrouter.py    # OpenRouter provider
│   └── zen.py           # Zen provider
├── mcp/
│   ├── client.py        # MCP server client (stdio + SSE)
│   └── bridge.py        # Bridge MCP tools into ToolRegistry
├── channels/
│   ├── base.py          # Channel protocol + ChannelMessage
│   ├── telegram.py      # Telegram bot channel (aiogram)
│   ├── slack.py         # Slack channel (slack-bolt)
│   ├── discord.py       # Discord channel (discord.py)
│   └── repl.py          # REPL stdin/stdout channel
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
│   ├── recorder.py      # Silero VAD + microphone capture
│   ├── stt.py           # Whisper speech-to-text
│   └── tts.py           # Silero TTS
└── types.py             # Message, Role, Session, ToolSpec, StreamEvent
```

---

## Configuration

Settings are loaded by `src/leuk/config.py` `load_settings()` in this order
(highest priority first):

1. Environment variables (`LEUK_LLM_*`, `LEUK_SQLITE_*`, `LEUK_*`)
2. `~/.config/leuk/config.env`
3. `~/.config/leuk/credentials.json` (API keys only, mode 0600)
4. `~/.config/leuk/config.json` (last-used provider/model, written by the REPL)
5. Compiled-in defaults

Key settings:

| Env var | Default | Description |
|---------|---------|-------------|
| `LEUK_LLM_PROVIDER` | `zen` | LLM provider: `anthropic`, `openai`, `google`, `openrouter`, `local`, `zen` |
| `LEUK_LLM_MODEL` | `big-pickle` | Model identifier |
| `LEUK_LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `LEUK_LLM_MAX_TOKENS` | `16384` | Max tokens per LLM call |
| `LEUK_MAX_TOOL_ROUNDS` | `50` | Max consecutive tool-use rounds |
| `LEUK_MAX_CONTEXT_TOKENS` | `100000` | Context window budget before truncation |
| `LEUK_SQLITE_PATH` | `~/.config/leuk/leuk.db` | SQLite database path |
| `LEUK_CHANNELS_TELEGRAM_BOT_TOKEN` | — | Telegram bot token |
| `LEUK_CHANNELS_ALLOWED_USERS` | `[]` | JSON list of allowed user IDs |
| `LEUK_SCHEDULER_ENABLED` | `false` | Enable background task scheduler |
| `LEUK_LOCAL_LLM_ENABLED` | `false` | Enable the local_llm tool (Ollama) |

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
