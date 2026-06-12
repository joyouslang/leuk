"""Dynamic model catalog -- fetches available models from provider APIs.

Instead of maintaining a hard-coded curated list, this module queries each
provider's model endpoint at runtime.  Results are cached in-process so the
model selector dialog stays snappy after the first fetch.
"""

from __future__ import annotations

import logging

import httpx

from leuk.billing import CC_USER_AGENT
from leuk.config import LLMConfig

logger = logging.getLogger(__name__)

# ── Provider metadata ──────────────────────────────────────────────

PROVIDER_NAMES: dict[str, str] = {
    "zen": "OpenCode Zen",
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "openrouter": "OpenRouter",
    "local": "Local",
}

# Credential keys used by each provider (for the has_credentials check).
_CRED_KEYS: dict[str, list[str]] = {
    "anthropic": ["anthropic_api_key", "anthropic_auth_token"],
    "openai": ["openai_api_key"],
    "google": ["google_api_key"],
    "openrouter": ["openrouter_api_key"],
    "zen": ["zen_api_key"],
    "local": [],  # always available
}

# Prefixes that indicate non-chat models (embeddings, TTS, etc.)
_SKIP_PREFIXES = (
    "text-embedding",
    "text-moderation",
    "dall-e",
    "whisper",
    "tts-",
    "davinci",
    "babbage",
    "curie",
    "ada",
)

_TIMEOUT = 15  # seconds for HTTP requests

# ── In-process cache ───────────────────────────────────────────────

_cache: dict[str, list[tuple[str, str]]] = {}


def invalidate_cache(provider: str | None = None) -> None:
    """Clear the cached model list(s)."""
    if provider:
        _cache.pop(provider, None)
    else:
        _cache.clear()


# ── Public helpers ─────────────────────────────────────────────────


def has_credentials(provider: str, creds: dict[str, str]) -> bool:
    """Check whether *creds* contains at least one key for *provider*."""
    if provider == "local":
        return True
    keys = _CRED_KEYS.get(provider, [f"{provider}_api_key"])
    return any(bool(creds.get(k)) for k in keys)


async def fetch_models(provider: str, config: LLMConfig) -> list[tuple[str, str]]:
    """Return ``[(model_id, display_name), ...]`` for *provider*.

    Uses a cached result when available; falls back to an empty list on
    network errors.
    """
    if provider in _cache:
        return _cache[provider]

    try:
        models = await _fetch_from_provider(provider, config)
    except Exception as exc:
        logger.warning("Failed to fetch models for %s: %s", provider, exc)
        models = []

    _cache[provider] = models
    return models


async def fetch_all_available(
    config: LLMConfig, creds: dict[str, str]
) -> dict[str, list[tuple[str, str]]]:
    """Fetch models for every provider that has credentials.

    Returns ``{provider: [(model_id, display_name), ...]}``, omitting
    providers whose fetch returned an empty list.
    """
    result: dict[str, list[tuple[str, str]]] = {}
    for prov_key in PROVIDER_NAMES:
        if not has_credentials(prov_key, creds):
            continue
        models = await fetch_models(prov_key, config)
        if models:
            result[prov_key] = models
    return result


# ── Per-provider fetchers ──────────────────────────────────────────


async def _fetch_from_provider(provider: str, config: LLMConfig) -> list[tuple[str, str]]:
    match provider:
        case "anthropic":
            return await _fetch_anthropic(config)
        case "openai":
            return await _fetch_openai(config)
        case "google":
            return await _fetch_google(config)
        case "openrouter":
            return await _fetch_openrouter(config)
        case "zen":
            return await _fetch_zen(config)
        case "local":
            return await _fetch_local(config)
        case _:
            return []


def _anthropic_headers(config: LLMConfig) -> dict[str, str] | None:
    """Auth headers for the Anthropic model listing, or None if no credentials.

    Mirrors the generation client: an API key uses ``x-api-key``; a Claude
    subscription uses the OAuth bearer token + the oauth beta header.
    """
    headers: dict[str, str] = {
        "anthropic-version": "2023-06-01",
        "User-Agent": CC_USER_AGENT,
    }
    if config.anthropic_api_key:
        headers["x-api-key"] = config.anthropic_api_key
    elif config.anthropic_auth_token:
        headers["authorization"] = f"Bearer {config.anthropic_auth_token}"
        headers["anthropic-beta"] = "files-api-2025-04-14,oauth-2025-04-20"
    else:
        return None
    return headers


async def _fetch_anthropic(config: LLMConfig) -> list[tuple[str, str]]:
    headers = _anthropic_headers(config)
    if headers is None:
        return []

    async def _get(hdrs: dict[str, str]) -> dict:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get("https://api.anthropic.com/v1/models", headers=hdrs)
            resp.raise_for_status()
            return resp.json()

    try:
        data = await _get(headers)
    except httpx.HTTPStatusError as exc:
        # A 401 under a Claude subscription means the OAuth token expired. Refresh
        # it and retry once — the same flow the generation path uses
        # (AnthropicProvider._try_refresh_token). No fallback: we use live data.
        if (
            exc.response.status_code == 401
            and not config.anthropic_api_key
            and config.anthropic_auth_token
        ):
            from leuk.cli.auth import refresh_anthropic_token

            new_token = refresh_anthropic_token()
            if not new_token:
                raise
            config.anthropic_auth_token = new_token
            refreshed = _anthropic_headers(config)
            assert refreshed is not None
            data = await _get(refreshed)
        else:
            raise

    models: list[tuple[str, str]] = []
    for m in data.get("data", []):
        model_id = m.get("id", "")
        display = m.get("display_name", model_id)
        if model_id:
            models.append((model_id, display))
    models.sort(key=lambda x: x[1])
    return models


async def _fetch_openai(config: LLMConfig) -> list[tuple[str, str]]:
    if not config.openai_api_key:
        return []

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {config.openai_api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()

    models: list[tuple[str, str]] = []
    for m in data.get("data", []):
        model_id = m.get("id", "")
        if not model_id:
            continue
        if any(model_id.startswith(p) for p in _SKIP_PREFIXES):
            continue
        models.append((model_id, model_id))

    models.sort(key=lambda x: x[0])
    return models


async def _fetch_google(config: LLMConfig) -> list[tuple[str, str]]:
    if not config.google_api_key:
        return []

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": config.google_api_key},
        )
        resp.raise_for_status()
        data = resp.json()

    models: list[tuple[str, str]] = []
    for m in data.get("models", []):
        model_id = m.get("name", "").removeprefix("models/")
        display = m.get("displayName", model_id)
        supported = m.get("supportedGenerationMethods", [])
        if "generateContent" in supported and model_id:
            models.append((model_id, display))

    models.sort(key=lambda x: x[1])
    return models


async def _fetch_openrouter(config: LLMConfig) -> list[tuple[str, str]]:
    if not config.openrouter_api_key:
        return []

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {config.openrouter_api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()

    models: list[tuple[str, str]] = []
    for m in data.get("data", []):
        model_id = m.get("id", "")
        display = m.get("name", model_id)
        if model_id:
            models.append((model_id, display))

    models.sort(key=lambda x: x[1])
    return models


async def _fetch_zen(config: LLMConfig) -> list[tuple[str, str]]:
    headers: dict[str, str] = {}
    if config.zen_api_key:
        headers["Authorization"] = f"Bearer {config.zen_api_key}"

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get("https://opencode.ai/zen/v1/models", headers=headers)
        resp.raise_for_status()
        data = resp.json()

    models: list[tuple[str, str]] = []
    for m in data.get("data", []):
        model_id = m.get("id", "")
        display = m.get("name", model_id)
        if model_id:
            models.append((model_id, display))

    models.sort(key=lambda x: x[1])
    return models


async def _fetch_local(config: LLMConfig) -> list[tuple[str, str]]:
    """List models from any OpenAI-compatible local endpoint.

    The standard ``GET {base}/models`` works for llama.cpp's llama-server,
    vLLM, and Ollama alike. Older Ollama builds without it fall back to the
    native ``/api/tags``.
    """
    v1 = config.local_base_url.rstrip("/")

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(f"{v1}/models")
            resp.raise_for_status()
            data = resp.json()
            entries = data.get("data", []) if isinstance(data, dict) else []
            ids = [m.get("id", "") for m in entries]
        except (httpx.HTTPError, ValueError):
            # Fall back to Ollama's native listing at the base address
            # (without the /v1 suffix).
            base = v1[:-3] if v1.endswith("/v1") else v1
            resp = await client.get(f"{base}/api/tags")
            resp.raise_for_status()
            ids = [m.get("name", "") for m in resp.json().get("models", [])]

    models: list[tuple[str, str]] = []
    for model_id in ids:
        if not model_id:
            continue
        display = model_id.removesuffix(":latest")
        # llama-server reports the GGUF path as the id — show just the file name.
        if "/" in display:
            display = display.rsplit("/", 1)[-1]
        models.append((model_id, display))

    models.sort(key=lambda x: x[1])
    return models
