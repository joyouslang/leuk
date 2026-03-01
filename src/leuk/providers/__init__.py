"""LLM provider abstraction layer."""

from __future__ import annotations

from leuk.config import LLMConfig
from leuk.providers.base import LLMProvider, NoCredentialsError


def _check_credentials(config: LLMConfig) -> None:
    """Raise NoCredentialsError if the active provider has no credentials.

    For Anthropic, either ``api_key`` (for console API keys) or
    ``auth_token`` (for OAuth with ``user:inference`` scope) is valid.
    """
    match config.provider:
        case "anthropic":
            if not config.anthropic_api_key and not config.anthropic_auth_token:
                raise NoCredentialsError("anthropic")
        case "openai":
            if not config.openai_api_key:
                raise NoCredentialsError("openai")
        case "google":
            if not config.google_api_key:
                raise NoCredentialsError("google")
        case "openrouter":
            if not config.openrouter_api_key:
                raise NoCredentialsError("openrouter")
        case "local":
            pass  # local (ollama) often needs no key


def create_provider(config: LLMConfig) -> LLMProvider:
    """Factory: instantiate the configured LLM provider.

    Raises NoCredentialsError if the provider requires credentials that
    are not set.
    """
    _check_credentials(config)

    match config.provider:
        case "anthropic":
            from leuk.providers.anthropic import AnthropicProvider

            return AnthropicProvider(config)
        case "openai":
            from leuk.providers.openai import OpenAIProvider

            return OpenAIProvider(config)
        case "google":
            from leuk.providers.google import GoogleProvider

            return GoogleProvider(config)
        case "openrouter":
            from leuk.providers.openrouter import OpenRouterProvider

            return OpenRouterProvider(config)
        case "local":
            from leuk.providers.openai import OpenAIProvider

            return OpenAIProvider(config, base_url=config.local_base_url, api_key=config.local_api_key)
        case _:
            raise ValueError(f"Unknown LLM provider: {config.provider!r}")


__all__ = ["LLMProvider", "NoCredentialsError", "create_provider"]
