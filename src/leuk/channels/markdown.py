"""Markdown → Telegram-HTML conversion.

Telegram's legacy ``Markdown`` parse mode fails *silently* on many real
inputs — unbalanced ``_``/``*``, code containing special characters, nested
emphasis — and the whole message either drops its formatting or errors out
(refactor-plan §6.3). Telegram's HTML mode is far more forgiving: we
HTML-escape everything first (so ``<``, ``>``, ``&`` are always safe), then
translate a small, well-defined subset of Markdown into the
`Telegram-supported HTML tags
<https://core.telegram.org/bots/api#html-style>`_.

Supported: fenced code blocks, inline code, bold, italic, strikethrough,
links, and headings (rendered as bold). Anything else passes through as
escaped plain text — never as a parse error.

This module has **no third-party dependencies** (no aiogram, no markdown-it)
so it can be unit-tested in isolation.
"""

from __future__ import annotations

import html
import re

# Code fences: ```lang\n...```  (lang optional). DOTALL so the body spans lines.
_FENCE_RE = re.compile(r"```([\w+-]*)\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_BOLD_STAR_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_BOLD_USCORE_RE = re.compile(r"(?<!_)__(.+?)__(?!_)", re.DOTALL)
_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*(?!\s)([^*\n]+?)\*(?!\*)")
_ITALIC_USCORE_RE = re.compile(r"(?<![\w_])_(?!\s)([^_\n]+?)_(?![\w_])")
_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*$", re.MULTILINE)

# Sentinel used to stash already-rendered code spans so later passes never
# touch their contents (NUL never appears in Telegram message text).
_STASH = "\x00{}\x00"


def markdown_to_telegram_html(text: str) -> str:
    """Convert a Markdown *text* to Telegram-flavoured HTML.

    The output is safe to send with ``parse_mode=HTML``: all literal text is
    HTML-escaped and only the recognised constructs become tags.
    """
    stash: list[str] = []

    def _stash(rendered: str) -> str:
        token = _STASH.format(len(stash))
        stash.append(rendered)
        return token

    # 1. Fenced code blocks (escape body, stash so nothing else rewrites it).
    def _fence(m: re.Match[str]) -> str:
        lang = m.group(1).strip()
        body = html.escape(m.group(2))
        if lang:
            return _stash(f'<pre><code class="language-{lang}">{body}</code></pre>')
        return _stash(f"<pre>{body}</pre>")

    text = _FENCE_RE.sub(_fence, text)

    # 2. Inline code.
    text = _INLINE_CODE_RE.sub(lambda m: _stash(f"<code>{html.escape(m.group(1))}</code>"), text)

    # 3. Escape everything that remains (code is already stashed).
    text = html.escape(text)

    # 4. Links — escape the visible label, keep the URL (escaped for the attr).
    text = _LINK_RE.sub(
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>',
        text,
    )

    # 5. Emphasis. Bold before italic so ``**x**`` is not eaten by the italic rule.
    text = _BOLD_STAR_RE.sub(r"<b>\1</b>", text)
    text = _BOLD_USCORE_RE.sub(r"<b>\1</b>", text)
    text = _STRIKE_RE.sub(r"<s>\1</s>", text)
    text = _ITALIC_STAR_RE.sub(r"<i>\1</i>", text)
    text = _ITALIC_USCORE_RE.sub(r"<i>\1</i>", text)

    # 6. Headings → bold lines.
    text = _HEADING_RE.sub(r"<b>\1</b>", text)

    # 7. Restore stashed code spans.
    for i, rendered in enumerate(stash):
        text = text.replace(_STASH.format(i), rendered)

    return text


def split_for_telegram(text: str, limit: int = 4096) -> list[str]:
    """Split *text* into chunks no longer than *limit* characters.

    Splits on line boundaries where possible so Markdown constructs (and the
    HTML they convert to) stay within a single chunk; only a single line that
    is itself longer than *limit* is hard-split.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            # A single oversized line: emit what we have, then hard-split.
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks
