[Home](README.md) › Configuration

# Configuration

All configuration is assembled by `load_settings()` in `src/leuk/config.py` into a
nested `Settings` model (pydantic-settings).

## Sources & precedence (highest first)

1. Environment variables (`LEUK_*`) — for CI/Docker/power users; see [Environment Variables](reference/environment.md)
2. `~/.config/leuk/config.json` — **the single config file**, written by `/settings`
3. `~/.config/leuk/credentials.json` (API keys / OAuth tokens, mode `0600`)
4. Compiled-in defaults

You don't need to export anything into your shell — configure everything from
`/settings` (which writes `config.json`) or by editing `config.json` directly. A
legacy `config.env` is **auto-migrated** into `config.json` on first run.

## `config.json`

`config.json` holds two kinds of keys:

- **Flat app keys** written by `/settings`: `last_provider`, `last_model`,
  `theme`, `review_policy`, `browser_enabled`, `input_control_enabled`,
  `input_control_auto_approve`, and the voice keys (`stt_*`, `tts_*`, `vad_*`).
- **Nested sub-model sections** mirroring the `Settings` tree, for the deeper
  engine knobs:

```json
{
  "llm": {"temperature": 0.2, "max_tokens": 8000},
  "input_control": {"enabled": true, "verify": "on_failure"},
  "channels": {"telegram_bot_token": "123456:ABC-..."}
}
```

Any env var (`LEUK_<SECTION>_<FIELD>`) overrides the matching `config.json` field.

## Files under `~/.config/leuk/`

| File | Purpose |
|------|---------|
| `config.json` | **all settings** — provider/model, theme, toggles, voice, nested sub-model config |
| `credentials.json` | provider API keys / OAuth tokens (mode 0600) |
| `leuk.db` | SQLite sessions + messages (see [Persistence](sessions-and-persistence.md)) |
| `repl_history` | prompt_toolkit command history |

## Config models

Each subsystem has a config block on `Settings` (`src/leuk/config.py`):

| Field | Model | Notes |
|-------|-------|-------|
| `llm` | `LLMConfig` | provider, model, keys, `context_window` override |
| `agent` | `AgentConfig` | `max_tool_rounds`, `max_context_tokens`, system prompt |
| `steering` | `SteeringConfig` | [steer weak/local models](steering.md): persistence + recovery, `auto`/`on`/`off` |
| `safety` | `SafetyConfig` | review policy, rules, approval timeout |
| `sandbox` | `SandboxConfig` | `none` or `container` (Docker) |
| `browser` | `BrowserConfig` | enable + headless (visible by default) |
| `monitoring` | `MonitoringConfig` | read-only host data (screenshot/geometry/system info) |
| `input_control` | `InputControlConfig` | desktop control + auto-approve + verify |
| `local_llm` | `LocalLLMConfig` | the optional `local_llm` tool |
| `scheduler` | `SchedulerConfig` | background task scheduler |
| `channels` | `ChannelsConfig` | bot tokens + allowlist |
| `archive` | `ArchiveConfig` | context archiving to disk |
| `mcp_servers` / `mcp_server` | MCP config | connect to (per-entry `enabled`) / expose servers |
| `mcp_registry` | `McpRegistryConfig` | registry URL + default source for [`/mcp`](mcp.md) imports |
| `skills` | `SkillsConfig` | [agent skills](skills.md): enable, directory, trusted/disabled slugs |
| `ui` | `UIConfig` | terminal UI prefs (`media_render`: `metadata` \| `inline`) |

## Feature toggles via `/settings` (config.json)

Some toggles are persisted to `config.json` and applied by `load_settings()` when
not already forced via env vars:

| `config.json` key | Effect |
|-------------------|--------|
| `theme` | active [colour theme](cli-and-ui.md) |
| `review_policy` | default [review policy](safety.md) |
| `browser_enabled` | enable the [Browser tool](tools/browser.md) |
| `monitoring_enabled` | enable the read-only [monitoring tool](tools/monitoring.md) |
| `input_control_enabled` | enable [desktop control](tools/input_control.md) |
| `input_control_auto_approve` | desktop-control auto-approval (also `/desktop-auto`) |
| `skills_enabled` | enable the [agent skills](skills.md) runtime |
| `media_render` | history-browser media mode: `metadata` or `inline` |
| `steering.enabled` | [model steering](steering.md): `auto` / `on` / `off` (also `/steering`) |
| STT/TTS/VAD keys | `stt_*`, `tts_*`, `vad_*` (see [Voice](voice.md)) |

## See also

- [Environment Variables](reference/environment.md) · [Providers](providers.md) · [Safety](safety.md)
