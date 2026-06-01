"""Queried model metadata — capabilities and context window.

We do **not** hard-code or guess model capabilities from their names (that goes
stale and is wrong for fast-moving local stacks like Ollama/vLLM, which now do
vision/audio). Instead each provider implements ``model_info()`` to **query its
API** for the active model and report what it actually exposes. Anything a
provider's API doesn't report stays ``None`` ("unknown"), and the caller treats
unknown vision as "send natively and let the API tell us" rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelInfo:
    """What a provider reports about the active model.

    Fields are ``None`` when the provider's API doesn't expose them.
    """

    context_window: int | None = None
    supports_vision: bool | None = None
    supports_audio: bool | None = None


# Field names different gateways use for the context length.
_CONTEXT_KEYS = (
    "context_length",
    "context_window",
    "max_context_length",
    "max_input_tokens",
    "max_model_len",
)


def context_window_from_obj(obj: object) -> int | None:
    """Pull a context-window size out of a model object / dict, trying the
    field names used across OpenAI-compatible gateways (OpenRouter, vLLM, …)."""
    def _get(source: object, key: str) -> object:
        if isinstance(source, dict):
            return source.get(key)
        return getattr(source, key, None)

    for key in _CONTEXT_KEYS:
        v = _get(obj, key)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    extra = _get(obj, "model_extra")
    if isinstance(extra, dict):
        for key in _CONTEXT_KEYS:
            v = extra.get(key)
            if isinstance(v, (int, float)) and v > 0:
                return int(v)
    return None


def modalities_from_obj(obj: object) -> tuple[bool | None, bool | None]:
    """Return ``(supports_vision, supports_audio)`` from a model object, reading
    the modality metadata OpenRouter-style gateways expose. ``(None, None)`` when
    no modality info is present."""
    arch = obj.get("architecture") if isinstance(obj, dict) else getattr(obj, "architecture", None)
    if arch is None:
        extra = obj.get("model_extra") if isinstance(obj, dict) else getattr(obj, "model_extra", None)
        if isinstance(extra, dict):
            arch = extra.get("architecture")
    if arch is None:
        return None, None

    def _field(name: str) -> object:
        if isinstance(arch, dict):
            return arch.get(name)
        return getattr(arch, name, None)

    mods = _field("input_modalities")
    if isinstance(mods, (list, tuple)):
        low = [str(x).lower() for x in mods]
        return ("image" in low), ("audio" in low)
    modality = _field("modality")  # e.g. "text+image->text"
    if isinstance(modality, str):
        inp = modality.split("->", 1)[0].lower()
        return ("image" in inp), ("audio" in inp)
    return None, None
