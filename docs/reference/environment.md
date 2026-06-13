[Home](../README.md) › [Configuration](../configuration.md) › Environment Variables

# Environment Variables

> **Env vars are optional.** The primary, recommended way to configure leuk is
> [`config.json`](../configuration.md) (via `/settings` or by editing the file) —
> nothing needs to go into your shell. Env vars exist for CI/Docker/power users
> and **override** the matching `config.json` field. Each `LEUK_<SECTION>_<FIELD>`
> env var maps to a `config.json` field, e.g. `LEUK_LLM_TEMPERATURE=0.2` ≡
> `{"llm": {"temperature": 0.2}}`.

Every config block reads a prefixed set of env vars (pydantic-settings). Prefixes
are defined by `model_config = SettingsConfigDict(env_prefix=...)` in
`src/leuk/config.py`.

## LLM — `LEUK_LLM_`

| Variable | Default | Description |
|----------|---------|-------------|
| `LEUK_LLM_PROVIDER` | `zen` | `zen`, `anthropic`, `openai`, `google`, `openrouter`, `local` |
| `LEUK_LLM_MODEL` | `big-pickle` | Model identifier |
| `LEUK_LLM_TEMPERATURE` | `0.0` | Sampling temperature (0–2) |
| `LEUK_LLM_MAX_TOKENS` | `16384` | Max output tokens per call |
| `LEUK_LLM_CONTEXT_WINDOW` | — | Override the usage-gauge window (else queried) |
| `LEUK_LLM_ANTHROPIC_API_KEY` | — | Anthropic API key |
| `LEUK_LLM_ANTHROPIC_AUTH_TOKEN` | — | Anthropic OAuth bearer token |
| `LEUK_LLM_OPENAI_API_KEY` | — | OpenAI key |
| `LEUK_LLM_GOOGLE_API_KEY` | — | Google Gemini key |
| `LEUK_LLM_OPENROUTER_API_KEY` | — | OpenRouter key |
| `LEUK_LLM_ZEN_API_KEY` | — | OpenCode Zen key |
| `LEUK_LLM_LOCAL_BASE_URL` | `http://localhost:11434/v1` | Local OpenAI-compatible endpoint |
| `LEUK_LLM_LOCAL_API_KEY` | `ollama` | Local endpoint key |

## Agent — `LEUK_`

| Variable | Default | Description |
|----------|---------|-------------|
| `LEUK_MAX_TOOL_ROUNDS` | `50` | Max consecutive tool-use rounds per turn |
| `LEUK_MAX_CONTEXT_TOKENS` | *(auto)* | Override the compaction budget; unset → derived from the model's queried context window (reserving reply room) |
| `LEUK_MAX_TOOL_RESULT_TOKENS` | `8000` | Max tokens per tool result before truncation |
| `LEUK_SYSTEM_PROMPT` | *(built-in)* | System prompt text |

## Steering — `LEUK_STEERING_`

Steers weaker/local models to persist like frontier models. See
[Steering](../steering.md).

| Variable | Default | Description |
|----------|---------|-------------|
| `LEUK_STEERING_ENABLED` | `auto` | `auto` (on only for `provider=local`), `on`, or `off` |
| `LEUK_STEERING_EXTRA_INSTRUCTIONS` | — | Extra steering text appended after the built-in block |
| `LEUK_STEERING_REMINDER_INTERVAL` | `8` | Re-inject a reminder every N tool rounds (0 = never); also after a tool error |
| `LEUK_STEERING_MAX_CONTINUATIONS` | `3` | Max self-reflection "continue" nudges per turn (bounded by `LEUK_MAX_TOOL_ROUNDS`) |
| `LEUK_STEERING_REFLECTION_MAX_TOKENS` | `256` | Token budget for each self-reflection check call |
| `LEUK_STEERING_REFLECT_ONLY_AFTER_TOOL_USE` | `true` | Only run the reflection check when a tool ran this turn (skips casual chat) |
| `LEUK_STEERING_NUDGE_ON_TRUNCATION` | `true` | On `finish_reason=length`, inject `continue` without a reflection call |
| `LEUK_STEERING_ENRICH_TOOL_ERRORS` | `true` | Append a recovery hint to errored tool results |
| `LEUK_STEERING_LOOP_DETECTION` | `true` | Detect & break a spinning (repeated/cyclic) tool loop, then consolidate |
| `LEUK_STEERING_LOOP_MIN_ROUNDS` | `4` | Only check for circling after this many tool rounds (the "lengthy" gate) |
| `LEUK_STEERING_LOOP_MAX_INTERVENTIONS` | `2` | Redirect nudges before forcing a tools-off consolidation reply |
| `LEUK_STEERING_SALVAGE_TEXT_TOOL_CALLS` | `true` | Recover tool calls a model emitted as plain text into real calls |

## Persistence — `LEUK_SQLITE_`

| Variable | Default | Description |
|----------|---------|-------------|
| `LEUK_SQLITE_PATH` | `~/.config/leuk/leuk.db` | SQLite database path |

## Local-LLM tool — `LEUK_LOCAL_LLM_`

| Variable | Default | Description |
|----------|---------|-------------|
| `LEUK_LOCAL_LLM_ENABLED` | `false` | Enable the `local_llm` delegation tool |
| `LEUK_LOCAL_LLM_BASE_URL` | `http://localhost:11434` | Ollama base URL |
| `LEUK_LOCAL_LLM_DEFAULT_MODEL` | `llama3.2` | Default Ollama model |

## Monitoring — `LEUK_MONITORING_`

| Variable | Default | Description |
|----------|---------|-------------|
| `LEUK_MONITORING_ENABLED` | `false` | Enable the read-only [monitoring tool](../tools/monitoring.md) (screenshot/geometry/system info) |

## Input control — `LEUK_INPUT_CONTROL_`

| Variable | Default | Description |
|----------|---------|-------------|
| `LEUK_INPUT_CONTROL_ENABLED` | `false` | Enable the [desktop control](../tools/input_control.md) tool |
| `LEUK_INPUT_CONTROL_BACKEND` | `ydotool` | Injection backend |
| `LEUK_INPUT_CONTROL_YDOTOOL_SOCKET` | — | `YDOTOOL_SOCKET` path |
| `LEUK_INPUT_CONTROL_VERIFY` | `on_failure` | `on_failure` \| `each_action` \| `never` |
| `LEUK_INPUT_CONTROL_AUTO_APPROVE` | `false` | Auto-approve desktop actions (dangerous) |

## Channels — `LEUK_CHANNELS_`

| Variable | Default | Description |
|----------|---------|-------------|
| `LEUK_CHANNELS_TELEGRAM_BOT_TOKEN` | — | Telegram bot token |
| `LEUK_CHANNELS_ALLOWED_USERS` | `[]` | JSON list of allowed user IDs ([Channels](../channels.md)) |

## Scheduler — `LEUK_SCHEDULER_`

| Variable | Default | Description |
|----------|---------|-------------|
| `LEUK_SCHEDULER_ENABLED` | `false` | Enable the background [scheduler](../scheduler.md) |

## Skills — `LEUK_SKILLS_`

| Variable | Default | Description |
|----------|---------|-------------|
| `LEUK_SKILLS_ENABLED` | `false` | Enable the [agent skills](../skills.md) runtime + `skill` tool |
| `LEUK_SKILLS_DIRECTORY` | `~/.config/leuk/skills` | Where installed skill bundles live |
| `LEUK_SKILLS_MAX_INDEX_SKILLS` | `50` | Cap on skills listed in the tool index |

## MCP registry — `LEUK_MCP_REGISTRY_`

| Variable | Default | Description |
|----------|---------|-------------|
| `LEUK_MCP_REGISTRY_URL` | `https://registry.modelcontextprotocol.io` | MCP registry base URL for [`/mcp`](../mcp.md) imports |
| `LEUK_MCP_REGISTRY_DEFAULT_SOURCE` | `mcp` | Default import source: `mcp`, `clawhub`, or `url` |

## Terminal UI — `LEUK_UI_`

| Variable | Default | Description |
|----------|---------|-------------|
| `LEUK_UI_MEDIA_RENDER` | `metadata` | History-browser media mode: `metadata` or `inline` |

## See also

- [Configuration](../configuration.md) · [Providers](../providers.md)
