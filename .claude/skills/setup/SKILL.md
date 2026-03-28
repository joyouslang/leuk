# Skill: /setup

First-time configuration wizard for leuk.  Walk through provider selection,
API key storage, and a smoke-test to confirm everything works.

---

## Overview

This skill guides you through:
1. Choosing an LLM provider
2. Setting credentials securely
3. Verifying the configuration
4. Optional: voice mode setup

Run this skill interactively in the leuk REPL or follow the steps manually.

---

## Step 1 — Install leuk

```bash
# From source (recommended for development)
git clone <repo>
cd leuk
uv sync

# Or install the package
pip install leuk
```

---

## Step 2 — Choose a provider

leuk supports the following providers (see `src/leuk/config.py` `LLMConfig.provider`):

| Provider     | Env var                  | Notes                                      |
|--------------|--------------------------|--------------------------------------------|
| `anthropic`  | `LEUK_LLM_ANTHROPIC_API_KEY` | Claude models; recommended for best performance |
| `openai`     | `LEUK_LLM_OPENAI_API_KEY`    | GPT-4o and derivatives                    |
| `google`     | `LEUK_LLM_GOOGLE_API_KEY`    | Gemini models                              |
| `openrouter` | `LEUK_LLM_OPENROUTER_API_KEY`| Access to many models via one key         |
| `local`      | *(none)*                     | Ollama / vLLM via `LEUK_LLM_LOCAL_BASE_URL` |
| `zen`        | `LEUK_LLM_ZEN_API_KEY`       | Internal/custom endpoint                  |

---

## Step 3 — Store credentials

### Option A: Interactive (recommended)

Start leuk — it will prompt for credentials on first run:

```bash
leuk
# Follow the /auth prompt
```

The `/auth` command in the REPL (`src/leuk/cli/auth.py`) saves credentials to
`~/.config/leuk/credentials.json` with mode 0600.

### Option B: Environment variables

```bash
export LEUK_LLM_PROVIDER=anthropic
export LEUK_LLM_MODEL=claude-sonnet-4-5-20251022
export LEUK_LLM_ANTHROPIC_API_KEY=sk-ant-...
leuk
```

### Option C: Config file

```bash
mkdir -p ~/.config/leuk
cat > ~/.config/leuk/config.env <<EOF
LEUK_LLM_PROVIDER=anthropic
LEUK_LLM_MODEL=claude-sonnet-4-5-20251022
EOF

# Store API key separately (credentials file is more secure than config.env)
python - <<'PYEOF'
from leuk.config import load_credentials, save_credentials
creds = load_credentials()
creds["anthropic_api_key"] = input("Anthropic API key: ").strip()
save_credentials(creds)
print("Saved to ~/.config/leuk/credentials.json")
PYEOF
```

---

## Step 4 — Verify configuration

```bash
leuk
# Type: "say hello" or "what model are you?"
# Expected: a response from your chosen provider
```

Check the settings dialog with `/settings` in the REPL
(`src/leuk/cli/settings_dialog.py`).

Check which provider/model is active:

```bash
python - <<'EOF'
from leuk.config import load_settings
s = load_settings()
print(f"Provider: {s.llm.provider}")
print(f"Model:    {s.llm.model}")
print(f"DB path:  {s.sqlite.path}")
EOF
```

---

## Step 5 — Optional: Voice mode

Voice mode requires additional dependencies (see `pyproject.toml` `[voice]` group):

```bash
uv sync --extra voice
leuk
# /voice-settings to configure microphone, STT model, TTS voice
```

Voice components:
- STT: `src/leuk/voice/stt.py` (Whisper via `transformers`)
- TTS: `src/leuk/voice/tts.py` (Silero TTS)
- VAD: `src/leuk/voice/recorder.py` (Silero VAD)

---

## Step 6 — Optional: MCP servers

To connect leuk to external MCP servers (e.g. filesystem, databases), edit
`~/.config/leuk/config.env`:

```bash
# Not directly supported in config.env yet — use the Settings API:
python - <<'EOF'
import json
from leuk.config import config_dir

mcp_config = {
    "mcp_servers": [
        {
            "name": "filesystem",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        }
    ]
}

# Merge into persistent config
path = config_dir() / "settings.json"
existing = json.loads(path.read_text()) if path.exists() else {}
existing.update(mcp_config)
path.write_text(json.dumps(existing, indent=2))
print("MCP server configured.")
EOF
```

MCP client code lives in `src/leuk/mcp/client.py` and `src/leuk/mcp/bridge.py`.

---

## Troubleshooting

- `NoCredentialsError`: run `/auth` in the REPL or set the relevant env var.
- Provider not found: check `LEUK_LLM_PROVIDER` spelling against `src/leuk/providers/catalog.py`.
- SQLite errors: check permissions on `~/.config/leuk/leuk.db`.
- Run `/debug` skill for a full diagnostic check.
