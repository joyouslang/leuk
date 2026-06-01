"""Resolve a model's context-window size (in tokens) for the usage gauge.

Resolution order (highest priority first) — no hard-coded per-model values:

1. A live query against the provider (``provider.model_info().context_window``).
   Most accurate; OpenAI-compatible gateways (OpenRouter, vLLM, Zen), Ollama,
   and Google all report it.
2. The user's ``LLMConfig.context_window`` override.
3. ``None`` if the provider doesn't expose it and the user hasn't set it — the
   gauge then shows the raw token estimate instead of a made-up percentage.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from leuk.config import LLMConfig
from leuk.providers.model_info import ModelInfo

logger = logging.getLogger(__name__)


@runtime_checkable
class _SupportsModelInfo(Protocol):
    async def model_info(self) -> ModelInfo: ...


async def resolve_context_window(
    config: LLMConfig,
    provider: object | None = None,
) -> int | None:
    """Resolve the context window for the configured model (see module docstring).

    Never raises — query failures are logged at debug level and fall through.
    """
    if provider is not None and isinstance(provider, _SupportsModelInfo):
        try:
            info = await provider.model_info()
            if info.context_window and info.context_window > 0:
                return int(info.context_window)
        except Exception as exc:  # noqa: BLE001 — query is best-effort
            logger.debug("model_info query failed for %s: %s", config.model, exc)

    if config.context_window:
        return config.context_window

    return None
