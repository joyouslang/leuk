# Skill: /add-ollama-tool

Add a `local_llm` tool that lets the leuk agent delegate cheap subtasks
(summarization, classification, extraction) to a local Ollama model, without
consuming tokens from the primary LLM provider.

This is different from using Ollama *as the provider* (`LEUK_LLM_PROVIDER=local`) —
the tool runs a secondary model alongside whatever primary provider is configured.

---

## Prerequisites

1. [Ollama](https://ollama.com) installed and running: `ollama serve`
2. At least one model pulled: `ollama pull llama3.2`
3. `httpx` is already in `pyproject.toml` dependencies — no new package needed.

---

## Step 1 — Add config section

File: `src/leuk/config.py`

Add after `MCPServerConfig`:

```python
class LocalLLMConfig(BaseSettings):
    """Local LLM (Ollama) tool configuration."""

    model_config = SettingsConfigDict(env_prefix="LEUK_LOCAL_LLM_", extra="ignore")

    enabled: bool = Field(default=False, description="Enable the local_llm tool")
    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL",
    )
    default_model: str = Field(
        default="llama3.2",
        description="Default model name for local_llm tool calls",
    )
```

Add to `Settings`:

```python
local_llm: LocalLLMConfig = Field(default_factory=LocalLLMConfig)
```

---

## Step 2 — Create the tool

Create `src/leuk/tools/local_llm.py`:

```python
"""Local LLM tool — delegates to Ollama via HTTP."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from leuk.types import ToolSpec

logger = logging.getLogger(__name__)

_TIMEOUT = 60.0  # seconds; local models can be slow


class LocalLLMTool:
    """Run a prompt against a local Ollama model."""

    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "llama3.2") -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="local_llm",
            description=(
                "Run a prompt against a local Ollama model. "
                "Use for cheap subtasks: summarization, classification, entity extraction. "
                "Returns the model's raw text response."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to send to the local model",
                    },
                    "model": {
                        "type": "string",
                        "description": f"Ollama model name (default: {self._default_model!r})",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate (default: 1000)",
                    },
                },
                "required": ["prompt"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        prompt = arguments["prompt"]
        model = arguments.get("model", self._default_model)
        max_tokens = int(arguments.get("max_tokens", 1000))

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            try:
                resp = await client.post(f"{self._base_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")
            except httpx.ConnectError:
                return f"[local_llm] Cannot connect to Ollama at {self._base_url}. Is 'ollama serve' running?"
            except httpx.HTTPStatusError as e:
                return f"[local_llm] HTTP error {e.response.status_code}: {e.response.text}"
```

---

## Step 3 — Register the tool

File: `src/leuk/tools/__init__.py`

In `create_default_registry()`, add conditional registration:

```python
from leuk.config import load_settings

def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ShellTool())
    registry.register(FileReadTool())
    registry.register(FileEditTool())
    registry.register(SubAgentTool())
    registry.register(WebFetchTool())

    settings = load_settings()
    if settings.local_llm.enabled:
        from leuk.tools.local_llm import LocalLLMTool
        registry.register(LocalLLMTool(
            base_url=settings.local_llm.base_url,
            default_model=settings.local_llm.default_model,
        ))

    return registry
```

---

## Step 4 — Enable the tool

```bash
# Option A: environment variable (per session)
export LEUK_LOCAL_LLM_ENABLED=true

# Option B: persist in config.env
echo "LEUK_LOCAL_LLM_ENABLED=true" >> ~/.config/leuk/config.env

# Override default model
echo "LEUK_LOCAL_LLM_DEFAULT_MODEL=mistral" >> ~/.config/leuk/config.env
```

---

## Verification

```bash
# Confirm Ollama is running and model is available
curl http://localhost:11434/api/tags | python -m json.tool

# Start leuk and test the tool
leuk
# > use local_llm to summarize: "The quick brown fox jumps over the lazy dog"
```

The agent should call `local_llm(prompt="Summarize: ...")` and receive a short
text response from the local model.

---

## Notes

- The tool is intentionally simple: no streaming, no tool recursion. The primary
  agent decides how to use the result.
- Supports any model available in `ollama list`. Check with `ollama list`.
- To use other local backends (llama.cpp server, vLLM), change `base_url` to the
  appropriate endpoint — the `/api/generate` path is Ollama-specific; vLLM uses
  the OpenAI-compatible `/v1/completions`.
- For a vLLM backend, the existing `providers/openai.py` with `local_base_url`
  may be a better fit than this tool.
