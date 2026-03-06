"""OpenCode Zen provider (OpenAI-compatible gateway with curated models)."""

from __future__ import annotations

from leuk.config import LLMConfig
from leuk.providers.openai import OpenAIProvider


class ZenProvider(OpenAIProvider):
    """LLM provider backed by the OpenCode Zen API.

    OpenCode Zen is an AI gateway that offers a curated set of models
    (including free ones) tested for coding agent workloads.  It exposes
    an OpenAI-compatible ``/chat/completions`` endpoint, so we reuse the
    :class:`OpenAIProvider` with a different base URL and API key.
    """

    ZEN_BASE_URL = "https://opencode.ai/zen/v1"

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(
            config,
            base_url=self.ZEN_BASE_URL,
            api_key=config.zen_api_key,
        )
