"""Local LLM tool — delegate subtasks to a local Ollama model."""

from __future__ import annotations

from typing import Any

import httpx

from leuk.types import ToolSpec

_TIMEOUT = 60  # seconds


class LocalLLMTool:
    """Run a prompt against a local Ollama model and return the response text."""

    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "llama3.2") -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="local_llm",
            description=(
                "Send a prompt to a local Ollama model and return the response. "
                "Use this for cheap subtasks like summarization, classification, or "
                "text extraction that don't need the main provider."
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
                        "description": f"Ollama model name (default: {self._default_model})",
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
        prompt: str = arguments["prompt"]
        model: str = arguments.get("model", self._default_model)
        max_tokens: int = int(arguments.get("max_tokens", 1000))

        url = f"{self._base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except httpx.TimeoutException:
            return f"[ERROR] Ollama request timed out after {_TIMEOUT}s"
        except httpx.ConnectError:
            return f"[ERROR] Could not connect to Ollama at {self._base_url}"
        except httpx.HTTPStatusError as exc:
            return f"[ERROR] Ollama returned HTTP {exc.response.status_code}"
        except httpx.RequestError as exc:
            return f"[ERROR] Request failed: {exc}"

        data = response.json()
        return data.get("response", "")
