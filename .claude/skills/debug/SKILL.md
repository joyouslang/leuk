# Skill: /debug

Diagnostic workflow for common leuk issues.  Run each check in order and stop
at the first failure — fix it before continuing.

---

## Check 1 — Python and dependencies

```bash
python --version
# Expected: Python 3.13+

uv run python -c "import leuk; print(leuk.__version__)"
# Expected: version string (no ImportError)

uv run python -c "import pydantic, pydantic_settings, aiosqlite, mcp, httpx, rich, prompt_toolkit"
# Expected: no output (all imports succeed)
```

If imports fail: `uv sync` to reinstall dependencies.

---

## Check 2 — Configuration loading

```bash
python - <<'EOF'
from leuk.config import load_settings
try:
    s = load_settings()
    print(f"OK — provider={s.llm.provider!r} model={s.llm.model!r}")
    print(f"     db={s.sqlite.path!r}")
    print(f"     max_tool_rounds={s.agent.max_tool_rounds}")
except Exception as e:
    print(f"FAIL — {type(e).__name__}: {e}")
EOF
```

Config files checked in order (see `src/leuk/config.py` `load_settings()`):
1. `LEUK_LLM_*` environment variables
2. `~/.config/leuk/config.env`
3. `~/.config/leuk/credentials.json`
4. `~/.config/leuk/config.json` (last-used provider/model)

---

## Check 3 — Credentials

```bash
python - <<'EOF'
from leuk.config import load_credentials, load_settings
creds = load_credentials()
s = load_settings()

providers = {
    "anthropic": s.llm.anthropic_api_key,
    "openai": s.llm.openai_api_key,
    "google": s.llm.google_api_key,
    "openrouter": s.llm.openrouter_api_key,
    "zen": s.llm.zen_api_key,
}

active = s.llm.provider
key = providers.get(active, "")
if key:
    masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "***"
    print(f"OK — {active} key present: {masked}")
elif active == "local":
    print(f"OK — local provider (no API key needed), base_url={s.llm.local_base_url!r}")
else:
    print(f"MISSING — no credentials for provider {active!r}")
    print("Run /auth in the REPL or set the appropriate LEUK_LLM_*_API_KEY env var.")
EOF
```

---

## Check 4 — SQLite store

```bash
python - <<'EOF'
import asyncio
from leuk.persistence.sqlite import SQLiteStore

async def check():
    store = SQLiteStore()
    try:
        await store.setup()
        sessions = await store.list_sessions(limit=5)
        print(f"OK — SQLite store connected, {len(sessions)} recent session(s)")
        await store.close()
    except Exception as e:
        print(f"FAIL — {type(e).__name__}: {e}")

asyncio.run(check())
EOF
```

If this fails: check that `~/.config/leuk/` is writable and disk is not full.

---

## Check 5 — LLM provider connectivity

```bash
python - <<'EOF'
import asyncio
from leuk.config import load_settings
from leuk.providers import create_provider
from leuk.types import Message, Role

async def check():
    s = load_settings()
    try:
        provider = create_provider(s.llm)
        msg = Message(role=Role.USER, content="Say 'ok' in one word.")
        reply = await provider.generate([msg], max_tokens=10)
        print(f"OK — {s.llm.provider} responded: {reply.content!r}")
    except Exception as e:
        print(f"FAIL — {type(e).__name__}: {e}")

asyncio.run(check())
EOF
```

Common failures:
- `NoCredentialsError` → run `/auth` or set env var (see Check 3)
- `httpx.ConnectError` → check internet / VPN / firewall
- `401 Unauthorized` → API key is wrong or expired
- `429 Too Many Requests` → rate limited; wait and retry

---

## Check 6 — Tool registry

```bash
python - <<'EOF'
from leuk.tools import create_default_registry
registry = create_default_registry()
for spec in registry.specs():
    print(f"  {spec.name}")
print(f"OK — {len(registry)} tools registered")
EOF
```

Expected tools: `shell`, `file_read`, `file_edit`, `sub_agent`, `web_fetch`.
Additional tools if enabled: `browser`, `local_llm`.

---

## Check 7 — Safety guard

```bash
python - <<'EOF'
from leuk.safety import SafetyGuard
from leuk.config import load_settings

s = load_settings()
guard = SafetyGuard(s.safety)

# Should be ALLOW
result = guard.check("file_read", "src/leuk/config.py")
print(f"file_read src/leuk/config.py → {result}")

# Should be DENY (matches .env rule)
result = guard.check("file_read", ".env")
print(f"file_read .env → {result}")
EOF
```

If safety rules are misbehaving, inspect `src/leuk/config.py`
`_default_safety_rules()` and `src/leuk/safety.py`.

---

## Check 8 — Voice mode (optional)

Only if you use `/voice`:

```bash
python - <<'EOF'
try:
    import sounddevice, torch, transformers
    print("OK — voice dependencies present")
    devices = sounddevice.query_devices()
    inputs = [d for d in devices if d["max_input_channels"] > 0]
    print(f"     {len(inputs)} input device(s) found")
except ImportError as e:
    print(f"SKIP — {e} (install with: uv sync --extra voice)")
EOF
```

---

## Check 9 — MCP server connections

```bash
python - <<'EOF'
from leuk.config import load_settings
s = load_settings()
if s.mcp_servers:
    for srv in s.mcp_servers:
        print(f"  configured: {srv.name!r} transport={srv.transport!r}")
else:
    print("No MCP servers configured.")
EOF
```

If an MCP server fails to connect, check `src/leuk/mcp/client.py` and verify the
command/URL in your config.

---

## Common fixes

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: leuk` | Run `uv sync` or `pip install -e .` |
| `NoCredentialsError` | `/auth` in REPL or `export LEUK_LLM_<PROVIDER>_API_KEY=...` |
| Agent loops forever | Check `LEUK_MAX_TOOL_ROUNDS` (default 50); interrupt with `Ctrl-C` |
| Context overflow | Reduce `LEUK_MAX_CONTEXT_TOKENS` or switch to `summarize` strategy |
| SQLite locked | Another leuk process is running; check with `pgrep leuk` |
| Voice not working | `uv sync --extra voice` then `playwright install` if using browser too |
