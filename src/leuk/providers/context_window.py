"""Resolve a model's context-window size (in tokens).

Used for the REPL's context-usage gauge. We rely solely on what the provider
reports, with the user's explicit config as a last resort — no hardcoded
per-model values (those go stale as models change).

Resolution order:

1. A live query against the provider (``provider.context_window()`` if
   implemented — e.g. OpenAI-compatible ``models.list`` exposes
   ``context_length`` on gateways like OpenRouter / Zen).
2. The user's ``LLMConfig.context_window`` override (last resort).
3. ``None`` if genuinely unknown — callers should then show a raw token
   count rather than inventing a percentage.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from leuk.config import LLMConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class _SupportsContextWindow(Protocol):
    async def context_window(self) -> int | None: ...


async def resolve_context_window(
    config: LLMConfig,
    provider: object | None = None,
) -> int | None:
    """Resolve the context window for the configured model.

    See the module docstring for the resolution order. Never raises — query
    failures are logged at debug level and fall through to the user override.
    """
    # 1. Best-effort live query from the provider.
    if provider is not None and isinstance(provider, _SupportsContextWindow):
        try:
            window = await provider.context_window()
            if window and window > 0:
                return int(window)
        except Exception as exc:  # noqa: BLE001 — query is best-effort
            logger.debug("context_window query failed for %s: %s", config.model, exc)

    # 2. User override (last resort).
    if config.context_window:
        return config.context_window

    # 3. Unknown.
    return None
