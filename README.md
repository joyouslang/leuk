# leuk

Persistent AI agent with sub-agent orchestration and full environment access.

leuk is a terminal-based agentic coding assistant that connects to multiple LLM
providers, persists conversations across sessions, manages context windows
automatically, and gives the model access to your shell, filesystem, and the
web.  Sub-agents can be spawned to work on independent tasks in parallel.

## Features

- **Multi-provider** -- Anthropic (Claude), OpenAI, Google Gemini, OpenRouter,
  and local models (Ollama / vLLM) through a unified interface.
- **Persistent sessions** -- Conversations are stored in SQLite and survive
  restarts.  Redis (optional) keeps hot context for instant resume.
- **Tool system** -- Shell execution, file read/edit, web fetch, and sub-agent
  delegation out of the box.
- **MCP support** -- Connect external tool servers via the Model Context
  Protocol (stdio and SSE transports).
- **Context management** -- Automatic sliding-window truncation or
  LLM-powered summarization to stay within token limits.
- **Sub-agents** -- Delegate tasks to independent child agents that run
  concurrently with their own sessions.
- **Streaming** -- Real-time token streaming in the terminal.
- **OAuth login** -- Authenticate with a Claude Pro / Max subscription via
  browser-based OAuth PKCE -- no API key required.

## Requirements

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/leuk.git
cd leuk

# Install with uv (creates the virtualenv automatically)
uv sync
```

## Quick start

```bash
# Run the REPL
uv run leuk

# Or via python -m
uv run python -m leuk
```

On first launch leuk will warn that no credentials are configured.  Run
`/auth` inside the REPL to set up a provider.

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
| `/quit`      | Exit leuk                              |

### `/models`

Opens an interactive popup dialog listing models grouped by provider.  Only
providers with configured credentials are shown.  Selecting a model from a
different provider switches the active provider automatically.

You can also set the model via the `LEUK_LLM_MODEL` environment variable or
`~/.config/leuk/config.env` for models not in the built-in catalog.

## Built-in tools

The agent has access to the following tools by default:

| Tool         | Description                                             |
| ------------ | ------------------------------------------------------- |
| `shell`      | Execute shell commands (timeout, working directory)     |
| `file_read`  | Read files with line numbers and offset/limit           |
| `file_edit`  | Create files or edit by exact string replacement        |
| `web_fetch`  | Fetch URLs and extract text, with optional CSS selector |
| `sub_agent`  | Spawn a child agent for an independent task             |

## Architecture

```
                     +-----------+
                     |  CLI/REPL |
                     +-----+-----+
                           |
                     +-----v-----+
                     |   Agent   |
                     |   Loop    |
                     +--+--+--+--+
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

             +-----------+-----------+
             |           |           |
           SQLite      Redis      Memory
          (durable)    (hot)    (fallback)
```

### Agent loop

1. The user message is appended to the session history.
2. The context window is prepared: tool results are truncated, then older
   messages are dropped (sliding window) or summarized.
3. The provider is called with the prepared context and available tools.
4. If the assistant responds with tool calls, each tool is executed and the
   results are appended.  Steps 2-4 repeat for up to `max_tool_rounds`
   (default 50).
5. All messages are persisted to SQLite after each turn.

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
| `LEUK_LLM_PROVIDER`            | `anthropic`                  | Active provider                |
| `LEUK_LLM_MODEL`               | `claude-sonnet-4-20250514`   | Model identifier               |
| `LEUK_LLM_TEMPERATURE`         | `0.0`                        | Sampling temperature (0-2)     |
| `LEUK_LLM_MAX_TOKENS`          | `16384`                      | Max output tokens              |
| `LEUK_LLM_ANTHROPIC_API_KEY`   | --                           | Anthropic API key              |
| `LEUK_LLM_ANTHROPIC_AUTH_TOKEN`| --                           | Anthropic OAuth bearer token   |
| `LEUK_LLM_OPENAI_API_KEY`      | --                           | OpenAI API key                 |
| `LEUK_LLM_GOOGLE_API_KEY`      | --                           | Google Gemini API key          |
| `LEUK_LLM_OPENROUTER_API_KEY`  | --                           | OpenRouter API key             |
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
| `LEUK_REDIS_URL`        | `redis://localhost:6379/0`   | Redis connection URL   |
| `LEUK_REDIS_PREFIX`     | `leuk:`                      | Redis key prefix       |
| `LEUK_REDIS_TTL_SECONDS`| `86400`                     | Hot-state TTL (24h)    |

### Credentials JSON

`~/.config/leuk/credentials.json` stores provider credentials with the
following keys:

```json
{
  "anthropic_api_key": "sk-ant-...",
  "anthropic_auth_token": "sk-ant-oat01-...",
  "anthropic_refresh_token": "sk-ant-ort01-...",
  "openai_api_key": "sk-...",
  "google_api_key": "AIza...",
  "openrouter_api_key": "sk-or-...",
  "local_api_key": "..."
}
```

The file is created with mode `0600` (owner read/write only).

## Providers

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

### Redis (hot store)

Optional.  Stores serialized context for fast session resume:

- `leuk:ctx:{session_id}` -- JSON-serialized recent messages (TTL 24h).
- `leuk:active_session` -- ID of the last active session.

If Redis is unavailable, leuk falls back to an in-memory store automatically.
Sessions are still persisted to SQLite regardless.

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

The `/models` dialog includes a curated list of models per provider:

| Provider    | Models                                                       |
| ----------- | ------------------------------------------------------------ |
| Anthropic   | Sonnet 4, Opus 4, 3.7 Sonnet, 3.5 Haiku                    |
| OpenAI      | GPT-4.1, GPT-4.1 Mini, GPT-4.1 Nano, o3, o4 Mini           |
| Google      | Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.0 Flash          |
| OpenRouter  | Claude Sonnet/Opus 4, GPT-4.1, Gemini 2.5 Pro, DeepSeek R1/V3 |
| Local       | Llama 3.1 (8B/70B), Qwen 2.5 (7B/32B), DeepSeek R1 8B, Mistral 7B |

Any model identifier can also be set directly via `LEUK_LLM_MODEL`.

## Development

```bash
# Install dev dependencies
uv sync --dev

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
    __init__.py              # Package root, __version__
    __main__.py              # python -m leuk entry point
    config.py                # Settings, credentials, config directory
    types.py                 # Core data types (Message, Session, ToolSpec, ...)
    py.typed                 # PEP 561 marker
    agent/
      core.py                # Agent loop with tool dispatch
      context.py             # Context window management
      sub_agent.py           # Sub-agent orchestration
    cli/
      repl.py                # Interactive REPL and streaming output
      auth.py                # OAuth PKCE and API key management
      models.py              # Model selection dialog and catalog
    mcp/
      client.py              # MCP client (stdio + SSE transports)
      bridge.py              # Bridge MCP tools into tool registry
    persistence/
      base.py                # DurableStore and HotStore protocols
      sqlite.py              # SQLite implementation
      redis.py               # Redis implementation
      memory.py              # In-memory fallback
    providers/
      base.py                # LLMProvider protocol, NoCredentialsError
      anthropic.py           # Anthropic Claude provider
      openai.py              # OpenAI provider (also local backend)
      google.py              # Google Gemini provider
      openrouter.py          # OpenRouter provider
    tools/
      base.py                # Tool protocol and ToolRegistry
      shell.py               # Shell command execution
      file_read.py           # File reading with line numbers
      file_edit.py           # File creation and editing
      web_fetch.py           # URL fetching and HTML extraction
      sub_agent.py           # Sub-agent delegation tool
  tests/
    conftest.py              # MockProvider, shared fixtures
    test_agent.py            # Agent loop tests
    test_auth.py             # OAuth and credential tests
    test_config.py           # Configuration tests
    test_context.py          # Context management tests
    test_models.py           # Model catalog and selector tests
    test_persistence.py      # SQLite and memory store tests
    test_tools.py            # Tool execution tests
    test_types.py            # Data type tests
```

### Test suite

128 tests covering:

- Agent loop (conversation, tool dispatch, streaming, max rounds)
- OAuth PKCE flow (PKCE generation, URL building, token exchange, refresh)
- Configuration (defaults, env precedence, credential loading)
- Context management (token estimation, truncation, sliding window)
- Model catalog (completeness, availability, dialog behavior)
- Persistence (SQLite CRUD, memory store, session management)
- Tools (shell, file read/edit, web fetch, sub-agent, registry)
- Types (roles, sessions, messages, tool calls, stream events)

## License

[AGPL-3.0-or-later](LICENSE)
