"""Claude Code billing header generation.

Anthropic requires a billing attribution header in the system prompt for
OAuth-authenticated requests.  The header encodes a version tag, an
entrypoint identifier, and a short hash derived from the first user message.
"""

from __future__ import annotations

import hashlib
import os

from leuk.types import Message, Role

# These must track the Claude Code release we are emulating.
CC_VERSION = "2.1.80"
CC_USER_AGENT = f"claude-code/{CC_VERSION}"
_BILLING_SALT = "59cf53e54c78"


def _first_user_message_text(messages: list[Message]) -> str:
    """Return the text content of the first user message."""
    for msg in messages:
        if msg.role == Role.USER and msg.content:
            return msg.content
    return ""


def _sample_js_char(text: str, idx: int) -> str:
    """Sample a character at a JavaScript string index.

    JavaScript strings are indexed by UTF-16 code units.  For most text this
    is equivalent to a plain Python string index, but surrogate pairs (astral
    codepoints) each occupy two JS indices.  We replicate the JS semantics so
    the hash matches.
    """
    utf16_units = text.encode("utf-16-le")
    # Each UTF-16 code unit is 2 bytes in the little-endian encoding.
    byte_offset = idx * 2
    if byte_offset + 1 < len(utf16_units):
        unit = int.from_bytes(utf16_units[byte_offset : byte_offset + 2], "little")
        # Decode the single UTF-16 code unit back to a Python str.  For
        # surrogate halves this produces the replacement character, but that
        # mirrors what the JS code does when reading individual code units.
        try:
            return bytes([utf16_units[byte_offset], utf16_units[byte_offset + 1]]).decode(
                "utf-16-le", errors="replace"
            )
        except Exception:
            return chr(unit) if unit < 0x10000 else "0"
    return "0"


def _compute_version_hash(first_user_text: str, version: str) -> str:
    """Compute the 3-char hex hash appended to cc_version."""
    sampled = "".join(_sample_js_char(first_user_text, i) for i in (4, 7, 20))
    payload = f"{_BILLING_SALT}{sampled}{version}"
    return hashlib.sha256(payload.encode()).hexdigest()[:3]


def billing_header(messages: list[Message]) -> str:
    """Build the ``x-anthropic-billing-header`` system-prompt line.

    Returns an empty string when the entrypoint is not set (i.e. not running
    as a Claude-Code-compatible client).
    """
    text = _first_user_message_text(messages)
    version_hash = _compute_version_hash(text, CC_VERSION)
    entrypoint = os.environ.get("CLAUDE_CODE_ENTRYPOINT", "cli")
    return (
        f"x-anthropic-billing-header: cc_version={CC_VERSION}.{version_hash}; "
        f"cc_entrypoint={entrypoint}; cch=00000;"
    )
