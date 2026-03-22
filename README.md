# leuk

Persistent AI agent with sub-agent orchestration and full environment access.

leuk is a terminal-based agentic coding assistant that connects to multiple LLM
providers, persists conversations across sessions, manages context windows
automatically, and gives the model access to your shell, filesystem, and the
web.  Sub-agents can be spawned to work on independent tasks in parallel.

## Features

- **Multi-provider** -- Anthropic (Claude), OpenAI, Google Gemini, OpenRouter,
  OpenCode Zen, and local models (Ollama / vLLM) through a unified interface.
- **Persistent sessions** -- Conversations are stored in SQLite and survive
  restarts.  In-memory hot state keeps context for instant resume.
- **Tool system** -- Shell execution, file read/edit, web fetch, and sub-agent
  delegation out of the box.
- **Safety guardrails** -- Dangerous-command detection, read-only sandbox mode,
  and configurable allow/deny/ask rules protect against destructive operations.
- **MCP support** -- Connect external tool servers via the Model Context
  Protocol (stdio and SSE transports).
- **Context management** -- Automatic sliding-window truncation or
  LLM-powered summarization to stay within token limits.
- **Sub-agents** -- Delegate tasks to independent child agents that run
  concurrently with their own sessions.
- **Streaming** -- Real-time token streaming with compact tool-call status
  tracking (spinners, elapsed time, truncated results).
- **Voice input** -- Hands-free speech-to-text via local Whisper
  (HuggingFace transformers) or OpenAI Whisper API, with neural Silero VAD
  for accurate speech detection.
- **Text-to-speech** -- Read agent responses aloud via local Silero TTS
  (multilingual, dual-model) or OpenAI TTS API with background playback.
- **OAuth login** -- Authenticate with a Claude Pro / Max subscription via
  browser-based OAuth PKCE -- no API key required.

## Requirements

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/leuk.git
cd leuk

# Install core dependencies
uv sync

# Install with voice support (STT + TTS, includes PyTorch ~180 MB)
uv sync --extra voice

# Install dev tools (pytest, ruff, mypy)
uv sync --group dev
```

### PyTorch index

By default, voice dependencies pull CPU-only PyTorch from the dedicated
index configured in `pyproject.toml`.  To use a different accelerator
(CUDA, ROCm), override the index:

```toml
# pyproject.toml
[tool.uv.sources]
torch = { index = "pytorch-rocm" }

[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm7.2"
explicit = true
```

Then re-run `uv sync --extra voice`.

## Quick start

```bash
# Run the REPL
uv run leuk

# Or via python -m
uv run python -m leuk
```

On first launch leuk will warn that no credentials are configured.  Run
`/auth` inside the REPL to set up a provider.  The default provider is
**OpenCode Zen** (`big-pickle` model) which offers free access with no API
key required.

### Authenticate with Claude Pro / Max (OAuth)

```
leuk> /auth
  1) Anthropic (Claude)  not configured
  ...
  a)dd  e)dit  d)elete  0) cancel

Choice: a
Provider: 1

  1) Claude Pro/Max subscription (OAuth login)
  2) Create a new API key (instructions)
  3) Enter API key manually

Method: 1
```

A browser window opens.  Log in with your Claude account, authorize access,
and paste the code back into the terminal.  leuk stores the OAuth token in
`~/.config/leuk/credentials.json` (mode 0600) and refreshes it automatically.

### Authenticate with an API key

For any provider you can paste an API key directly:

```
leuk> /auth
Choice: a
Provider: 2          # e.g. OpenAI
OpenAI API key: sk-...
```

## REPL commands

| Command      | Description                            |
| ------------ | -------------------------------------- |
| `/help`      | Show available commands                |
| `/models`    | Open model selector dialog             |
| `/new`       | Start a new session                    |
| `/sessions`  | List recent sessions                   |
| `/auth`      | Select provider / manage credentials   |
| `/sandbox`   | Toggle read-only sandbox mode          |
| `/safety`    | Show safety guardrail status           |
| `/verbose`   | Toggle verbose tool output             |
| `/voice`     | Toggle voice input (push-to-talk)      |
| `/speak`     | Toggle text-to-speech output           |
| `/quit`      | Exit leuk                              |

### `/models`

Opens an interactive popup dialog listing models grouped by provider.  Only
providers with configured credentials are shown.  Selecting a model from a
different provider switches the active provider automatically.

You can also set the model via the `LEUK_LLM_MODEL` environment variable or
`~/.config/leuk/config.env`.

### `/sandbox`

Toggles **read-only sandbox mode**.  When enabled, all write operations
(shell commands, file edits, sub-agent spawning) are blocked by the safety
guard.

### `/safety`

Shows the current safety configuration: sandbox state, project root, and
a summary of deny/ask/allow rules.

### `/voice`

Toggles **hands-free voice input**.  When enabled, a neural Silero VAD
continuously monitors the microphone and automatically detects speech.
When you stop talking, the audio is transcribed via Whisper and sent to
the agent as text.  No button press required.

Requires the `[voice]` optional dependencies.

### `/speak`

Toggles **text-to-speech output**.  When enabled, assistant responses are
spoken aloud after streaming completes.  Uses Silero TTS by default
(multilingual, loads separate models for English and the user's language).

Requires the `[voice]` optional dependencies.

## Built-in tools

The agent has access to the following tools by default:

| Tool         | Description                                             |
| ------------ | ------------------------------------------------------- |
| `shell`      | Execute shell commands (timeout, working directory)     |
| `file_read`  | Read files with line numbers and offset/limit           |
| `file_edit`  | Create files or edit by exact string replacement        |
| `web_fetch`  | Fetch URLs and extract text, with optional CSS selector |
| `sub_agent`  | Spawn a child agent for an independent task             |

## Safety guardrails

All tool calls pass through a `SafetyGuard` before execution, providing
three layers of protection:

1. **Dangerous-command detection** -- Regex patterns catch destructive shell
   commands (`rm -rf`, `sudo`, `curl | bash`, `git push --force`, `mkfs`,
   etc.) and escalate to user confirmation.

2. **Read-only sandbox** -- A master toggle (`/sandbox`) that blocks all
   write tools (shell, file_edit, sub_agent).

3. **Configurable rules** -- Per-tool allowlist/blocklist rules with
   deny > ask > allow priority.  Rules use glob patterns matched against the
   primary argument (command, path, URL).

Path containment ensures file writes stay within the project root.  Protected
paths (`~/.ssh`, `~/.gnupg`, etc.) are always blocked for writes.

## Architecture

```
                     +-----------+
                     |  CLI/REPL |
                     +-----+-----+
                           |
                    +------v------+
                    |    Agent    |
                    |    Loop     |
                    +--+--+--+---+
                       |  |  |
            +----------+  |  +-----------+
            |             |              |
      +-----v---+  +-----v-----+  +-----v------+
      | Provider |  |   Tools   |  |  Context   |
      | (LLM)   |  | Registry  |  |  Manager   |
      +---------+  +-----+-----+  +------------+
                         |
         +-------+-------+-------+-------+-------+
         |       |       |       |       |       |
       Shell  FileRead FileEdit SubAg  WebFetch MCP
                                               (bridge)
              +------+------+
              |      |      |
            Safety  Render  Voice
            Guard   Engine  (STT/TTS)

                +-------+-------+
                |               |
              SQLite          Memory
             (durable)      (hot state)
```

### Agent loop

1. The user message is appended to the session history.
2. The context window is prepared: tool results are truncated, then older
   messages are dropped (sliding window) or summarized.
3. The provider is called with the prepared context and available tools.
4. If the assistant responds with tool calls, each call passes through the
   **safety guard** (deny/ask/allow), then executes.  Results are appended.
   Steps 2-4 repeat for up to `max_tool_rounds` (default 50).
5. All messages are persisted to SQLite after each turn.

### Streaming and rendering

The `StreamRenderer` consumes the async stream from `Agent.run_stream()` and
manages strict phase separation between `rich.Live` (animated spinners
during tool execution) and `prompt_toolkit` (user input).

Tool calls display as compact status lines:
- `⟳ shell(command='ls')` -- in-flight with spinner
- `✓ shell  120ms` -- completed with elapsed time
- `✗ shell  45ms  [ERROR] command not found` -- failed with error preview

The `/verbose` toggle switches between truncated previews (~200 chars) and
full tool output.

### Sub-agents

The `sub_agent` tool creates an independent `Agent` instance with its own
`Session` (linked via `parent_session_id`).  The sub-agent runs as an
`asyncio.Task`, sharing the same provider and tool registry but maintaining
separate conversation history.

### MCP integration

External MCP servers are configured in `Settings.mcp_servers`.  On startup
the REPL connects to each server, discovers its tools via `list_tools()`, and
registers them in the tool registry with prefixed names
(`mcp_{server}_{tool}`).  Calls are proxied through `MCPClient.call_tool()`.

Two transports are supported:

- **stdio** -- launches the server as a subprocess.
- **SSE** -- connects to an HTTP Server-Sent Events endpoint.

## Voice

Voice features require the `[voice]` optional dependency group:

```bash
uv sync --extra voice
```

This installs additional packages: PyTorch, transformers (Whisper),
sounddevice, numpy, and omegaconf (for Silero TTS via torch.hub).

### Speech-to-text (STT)

Two backends are available:

| Backend | Engine | Speed | Offline |
| ------- | ------ | ----- | ------- |
| `local` (default) | HuggingFace Whisper (transformers) | GPU: ~10× realtime | Yes |
| `openai` | OpenAI Whisper API | Real-time | No |

The local backend uses the `turbo` model by default (large-v3-turbo).
Silero VAD filters silence before transcription to prevent hallucinations.

### Text-to-speech (TTS)

Two backends are available:

| Backend | Engine | Speed | Offline |
| ------- | ------ | ----- | ------- |
| `local` (default) | Silero TTS (multilingual) | 3–17× realtime on CPU | Yes |
| `openai` | OpenAI TTS API (tts-1) | Real-time | No |

The local backend loads two Silero models when the user's language is not
English — one for English text and one for the configured language.
Playback blocks until complete (VAD is paused during playback to prevent
feedback loops).

## Configuration

All configuration lives under `~/.config/leuk/`.

| File                | Purpose                                      |
| ------------------- | -------------------------------------------- |
| `config.env`        | Key-value env overrides (pydantic-settings)  |
| `credentials.json`  | API keys and OAuth tokens (mode 0600)        |
| `leuk.db`           | SQLite session and message storage           |
| `repl_history`      | REPL command history (prompt_toolkit)        |

Settings are loaded with the following precedence (highest first):

1. Environment variables
2. `~/.config/leuk/config.env`
3. `~/.config/leuk/credentials.json`
4. Built-in defaults

### Environment variables

#### LLM (`LEUK_LLM_` prefix)

| Variable                        | Default                      | Description                    |
| ------------------------------- | ---------------------------- | ------------------------------ |
| `LEUK_LLM_PROVIDER`            | `zen`                        | Active provider                |
| `LEUK_LLM_MODEL`               | `big-pickle`                 | Model identifier               |
| `LEUK_LLM_TEMPERATURE`         | `0.0`                        | Sampling temperature (0-2)     |
| `LEUK_LLM_MAX_TOKENS`          | `16384`                      | Max output tokens              |
| `LEUK_LLM_ANTHROPIC_API_KEY`   | --                           | Anthropic API key              |
| `LEUK_LLM_ANTHROPIC_AUTH_TOKEN`| --                           | Anthropic OAuth bearer token   |
| `LEUK_LLM_OPENAI_API_KEY`      | --                           | OpenAI API key                 |
| `LEUK_LLM_GOOGLE_API_KEY`      | --                           | Google Gemini API key          |
| `LEUK_LLM_OPENROUTER_API_KEY`  | --                           | OpenRouter API key             |
| `LEUK_LLM_ZEN_API_KEY`         | --                           | OpenCode Zen API key           |
| `LEUK_LLM_LOCAL_BASE_URL`      | `http://localhost:11434/v1`  | Local model endpoint           |
| `LEUK_LLM_LOCAL_API_KEY`       | `ollama`                     | Local model API key            |

#### Agent (`LEUK_` prefix)

| Variable                     | Default            | Description                              |
| ---------------------------- | ------------------ | ---------------------------------------- |
| `LEUK_MAX_TOOL_ROUNDS`      | `50`               | Max consecutive tool-use rounds          |
| `LEUK_MAX_CONTEXT_TOKENS`   | `100000`           | Max estimated context window tokens      |
| `LEUK_MAX_TOOL_RESULT_TOKENS`| `8000`            | Max tokens per tool result               |
| `LEUK_CONTEXT_STRATEGY`     | `sliding_window`   | `sliding_window` or `summarize`          |
| `LEUK_SYSTEM_PROMPT`        | *(built-in)*       | System prompt text                       |

#### Persistence

| Variable                | Default                      | Description            |
| ----------------------- | ---------------------------- | ---------------------- |
| `LEUK_SQLITE_PATH`     | `~/.config/leuk/leuk.db`    | SQLite database path   |

### Credentials JSON

`~/.config/leuk/credentials.json` stores provider credentials:

```json
{
  "anthropic_api_key": "sk-ant-...",
  "anthropic_auth_token": "sk-ant-oat01-...",
  "anthropic_refresh_token": "sk-ant-ort01-...",
  "openai_api_key": "sk-...",
  "google_api_key": "AIza...",
  "openrouter_api_key": "sk-or-...",
  "zen_api_key": "...",
  "local_api_key": "..."
}
```

The file is created with mode `0600` (owner read/write only).

## Providers

### OpenCode Zen (default)

Free access to curated models via an OpenAI-compatible gateway.  The default
model is `big-pickle`.  No API key required for basic usage.

### Anthropic

Supports two authentication methods:

- **API key** (`anthropic_api_key`) -- standard `X-Api-Key` header.
- **OAuth token** (`anthropic_auth_token`) -- sends `Authorization: Bearer`
  with the `anthropic-beta: oauth-2025-04-20` header.  Obtained via the
  `/auth` OAuth PKCE flow.  Tokens are refreshed automatically on `401`.

### OpenAI

Standard OpenAI client.  Also used as the backend for the `local` provider.

### Google Gemini

Uses the `google-genai` SDK.  Tool calls generate synthetic IDs
(`call_{name}_{index}`).

### OpenRouter

Extends the OpenAI provider with `base_url=https://openrouter.ai/api/v1` and
the OpenRouter API key.

### Local (Ollama / vLLM)

Uses the OpenAI provider pointed at a local endpoint (default
`http://localhost:11434/v1`).  No API key required for Ollama.

## Persistence

### SQLite (durable store)

All sessions and messages are stored in `~/.config/leuk/leuk.db`:

- **sessions** table -- `id`, `status`, `created_at`, `updated_at`,
  `system_prompt`, `metadata` (JSON), `parent_session_id`.
- **messages** table -- `session_id` (FK), `role`, `content`, `tool_calls`
  (JSON), `tool_result` (JSON), `timestamp`, `metadata` (JSON).

### In-memory hot store

An in-memory store keeps serialized context for fast session resume:

- Recent messages (last 100) are JSON-cached in memory.
- The active session ID is tracked for automatic resume on restart.

Sessions are always persisted to SQLite regardless.

## Context management

Two strategies are available (set via `LEUK_CONTEXT_STRATEGY`):

### `sliding_window` (default)

1. Tool results exceeding `max_tool_result_tokens` are truncated.
2. If total estimated tokens exceed `max_context_tokens`, the oldest
   non-system messages are dropped from the front.
3. A placeholder note is injected: *"[SYSTEM NOTE: N earlier messages were
   trimmed...]"*.

### `summarize`

1. Same tool result truncation.
2. If over budget, the message list is split in half.
3. The first half is sent to the LLM with a summarization prompt.
4. The summary replaces the first half as a system note.
5. Falls back to `sliding_window` on summarization failure.

## Model catalog

The `/models` dialog fetches available models dynamically from each provider
API at runtime.  Results are cached in-process for the session duration.

Any model identifier can also be set directly via `LEUK_LLM_MODEL`.

## Development

```bash
# Install all dependencies (core + voice + dev)
uv sync --extra voice --group dev

# Run tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/ tests/
```

### Project layout

```
leuk/
  pyproject.toml
  src/leuk/
    __init__.py              # Package root
    __main__.py              # python -m leuk entry point
    config.py                # Settings, credentials, safety config
    types.py                 # Core data types (Message, Session, ToolSpec, ...)
    safety.py                # SafetyGuard, dangerous-op detection, rules
    py.typed                 # PEP 561 marker
    agent/
      core.py                # Agent loop with tool dispatch and safety gate
      context.py             # Context window management
      sub_agent.py           # Sub-agent orchestration
    cli/
      repl.py                # Interactive REPL, slash commands, voice integration
      render.py              # StreamRenderer, ToolStatusTracker, /verbose
      auth.py                # OAuth PKCE and API key management
      models.py              # Model selection dialog
    mcp/
      client.py              # MCP client (stdio + SSE transports)
      bridge.py              # Bridge MCP tools into tool registry
    persistence/
      base.py                # DurableStore and HotStore protocols
      sqlite.py              # SQLite implementation
      memory.py              # In-memory hot state store
    providers/
      base.py                # LLMProvider protocol, NoCredentialsError
      anthropic.py           # Anthropic Claude provider
      openai.py              # OpenAI provider (also local backend)
      google.py              # Google Gemini provider
      openrouter.py          # OpenRouter provider
      zen.py                 # OpenCode Zen provider
      catalog.py             # Dynamic model catalog (runtime API fetching)
    tools/
      base.py                # Tool protocol and ToolRegistry
      shell.py               # Shell command execution
      file_read.py           # File reading with line numbers
      file_edit.py           # File creation and editing
      web_fetch.py           # URL fetching and HTML extraction
      sub_agent.py           # Sub-agent delegation tool
    voice/
      __init__.py            # Optional dep detection, VOICE_AVAILABLE flag
      recorder.py            # MicRecorder, AudioClip (sounddevice capture)
      stt.py                 # LocalWhisperSTT, OpenAIWhisperSTT backends
      tts.py                 # SileroTTS, OpenAITTS backends
  tests/
    conftest.py              # MockProvider, shared fixtures
    test_agent.py            # Agent loop tests
    test_auth.py             # OAuth and credential tests
    test_config.py           # Configuration tests
    test_context.py          # Context management tests
    test_models.py           # Model catalog and selector tests
    test_persistence.py      # SQLite and memory store tests
    test_render.py           # StreamRenderer and tool status tests
    test_safety.py           # Safety guardrail tests
    test_tts.py              # Text-to-speech backend tests
    test_tools.py            # Tool execution tests
    test_types.py            # Data type tests
    test_voice.py            # Voice input and STT tests
```

### Test suite

261 tests covering:

- Agent loop (conversation, tool dispatch, streaming, max rounds)
- OAuth PKCE flow (PKCE generation, URL building, token exchange, refresh)
- Configuration (defaults, env precedence, credential loading)
- Context management (token estimation, truncation, sliding window)
- Model catalog (availability, dialog behavior)
- Persistence (SQLite CRUD, memory store, session management)
- Safety guardrails (dangerous commands, path validation, rules, sandbox)
- Tool-call rendering (status tracking, truncation, verbose mode, streaming)
- Tools (shell, file read/edit, web fetch, sub-agent, registry)
- Types (roles, sessions, messages, tool calls, stream events)
- Voice input (recorder, AudioClip, STT backends, factory)
- Text-to-speech (TTS backends, synthesis, factory)

## License

[AGPL-3.0-or-later](LICENSE)
