"""OpenRouter provider (OpenAI-compatible with custom base URL and headers)."""

from __future__ import annotations

from leuk.config import LLMConfig
from leuk.providers.openai import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    """LLM provider backed by the OpenRouter API.

    OpenRouter is OpenAI-compatible, so we reuse the OpenAI provider
    with a different base URL and API key.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(
            config,
            base_url=self.OPENROUTER_BASE_URL,
            api_key=config.openrouter_api_key,
        )
