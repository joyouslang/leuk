"""History tool — lets the agent navigate the full conversation at will.

Compaction (``agent/context.py``) keeps the in-context view small: older
messages are replaced by an incrementally-merged structured summary and
archived. The **complete** conversation, however, is always in SQLite — this
read-only tool exposes it to the model, so compaction never makes information
unreachable: the summary stays in context, and anything older can be searched
and re-read on demand.

Indices are stable positions in the full stored history (0-based), so a
``search`` hit can be expanded with ``read`` around its index.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from leuk.media import extract_media
from leuk.types import Message, ToolSpec

# Async callable returning the session's complete stored message list.
HistorySource = Callable[[], Awaitable[list[Message]]]

_SNIPPET = 160  # chars of context around a search match
_READ_CHARS = 2000  # max chars shown per message in read mode
_MAX_RESULTS = 20


def _message_text(msg: Message) -> str:
    """All searchable/displayable text of a message (media stripped)."""
    parts: list[str] = []
    if msg.content:
        clean, media = extract_media(msg.content)
        parts.append(clean)
        if media:
            parts.append(f"[{len(media)} media attachment(s)]")
    for tc in msg.tool_calls or []:
        parts.append(f"tool_call {tc.name}({tc.arguments})")
    if msg.tool_result:
        clean, media = extract_media(msg.tool_result.content or "")
        parts.append(f"tool_result {msg.tool_result.name}: {clean}")
        if media:
            parts.append(f"[{len(media)} media attachment(s)]")
    return "\n".join(p for p in parts if p)


def _header(i: int, msg: Message) -> str:
    ts = msg.timestamp.strftime("%Y-%m-%d %H:%M")
    return f"#{i} [{msg.role.value} · {ts}]"


class HistoryTool:
    """Read-only access to the session's full stored conversation."""

    def __init__(self) -> None:
        self._source: HistorySource | None = None

    def set_source(self, source: HistorySource) -> None:
        """Wire the active session's message fetcher (set by the Agent)."""
        self._source = source

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="history",
            description=(
                "Navigate the FULL conversation history of this session — "
                "including everything summarized away by context compaction. "
                "Use action='search' with a query to find earlier messages "
                "(returns indices + snippets), then action='read' with "
                "start/count to re-read the originals around an index."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "read"],
                        "description": "search: find messages by text; read: fetch a range",
                    },
                    "query": {
                        "type": "string",
                        "description": "Text to search for (case-insensitive; for action=search)",
                    },
                    "start": {
                        "type": "integer",
                        "description": "First message index to read (for action=read)",
                    },
                    "count": {
                        "type": "integer",
                        "description": "How many messages to read (default 5, max 20)",
                    },
                },
                "required": ["action"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        if self._source is None:
            return "[ERROR] History is not available in this context."
        messages = await self._source()
        action = arguments.get("action", "")
        if action == "search":
            return self._search(messages, str(arguments.get("query", "")))
        if action == "read":
            start = int(arguments.get("start", 0))
            count = min(int(arguments.get("count", 5) or 5), _MAX_RESULTS)
            return self._read(messages, start, count)
        return "[ERROR] Unknown action — use 'search' or 'read'."

    @staticmethod
    def _search(messages: list[Message], query: str) -> str:
        if not query.strip():
            return "[ERROR] action=search requires a non-empty 'query'."
        q = query.lower()
        hits: list[str] = []
        for i, msg in enumerate(messages):
            text = _message_text(msg)
            pos = text.lower().find(q)
            if pos < 0:
                continue
            lo = max(0, pos - _SNIPPET // 2)
            snippet = text[lo : lo + _SNIPPET].replace("\n", " ").strip()
            hits.append(f"{_header(i, msg)} …{snippet}…")
            if len(hits) >= _MAX_RESULTS:
                hits.append("(more matches exist — refine the query)")
                break
        if not hits:
            return f"No messages match {query!r} (searched {len(messages)})."
        return (
            f"{len(hits)} match(es) in {len(messages)} stored messages "
            f"(action='read' with start=<index> for full text):\n" + "\n".join(hits)
        )

    @staticmethod
    def _read(messages: list[Message], start: int, count: int) -> str:
        n = len(messages)
        if n == 0:
            return "The stored history is empty."
        start = max(0, min(start, n - 1))
        out: list[str] = [f"Messages {start}–{min(start + count, n) - 1} of {n}:"]
        for i in range(start, min(start + count, n)):
            text = _message_text(messages[i]) or "(empty)"
            if len(text) > _READ_CHARS:
                text = text[:_READ_CHARS] + f"… [+{len(text) - _READ_CHARS} chars]"
            out.append(f"{_header(i, messages[i])}\n{text}")
        return "\n\n".join(out)
