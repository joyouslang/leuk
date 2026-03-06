"""Voice input/output for leuk.

Requires optional ``[voice]`` dependencies::

    uv pip install leuk[voice]
"""

from __future__ import annotations

VOICE_AVAILABLE = False
_MISSING_REASON = ""

try:
    import sounddevice  # noqa: F401
    import numpy  # noqa: F401

    VOICE_AVAILABLE = True
except ImportError as exc:
    _MISSING_REASON = (
        f"Voice dependencies not installed ({exc}). Install with: uv pip install leuk[voice]"
    )


def require_voice() -> None:
    """Raise ImportError if voice dependencies are missing."""
    if not VOICE_AVAILABLE:
        raise ImportError(_MISSING_REASON)
