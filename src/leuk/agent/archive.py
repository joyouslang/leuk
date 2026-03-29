"""Conversation archiving: write dropped messages to markdown files."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from leuk.types import Message, Role

logger = logging.getLogger(__name__)


def _format_messages(messages: list[Message]) -> str:
    """Render a list of messages as markdown."""
    parts: list[str] = []
    for msg in messages:
        if msg.role == Role.SYSTEM:
            parts.append(f"## System\n\n{msg.content or ''}\n")
        elif msg.role == Role.USER:
            parts.append(f"## User\n\n{msg.content or ''}\n")
        elif msg.role == Role.ASSISTANT:
            header = "## Assistant\n\n"
            body = msg.content or ""
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    body += f"\n### Tool Call: {tc.name}\n\n```json\n{tc.arguments}\n```\n"
            parts.append(header + body + "\n")
        elif msg.role == Role.TOOL:
            if msg.tool_result:
                name = msg.tool_result.name or "tool"
                parts.append(f"### Tool Result: {name}\n\n{msg.tool_result.content}\n")
        else:
            parts.append(f"## {msg.role.value.title()}\n\n{msg.content or ''}\n")
    return "\n".join(parts)


def _archive_path(archive_dir: Path, session_id: str) -> Path:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return archive_dir / f"{session_id[:8]}_{timestamp}.md"


def _write_archive(path: Path, messages: list[Message]) -> None:
    """Synchronous file write, run via asyncio.to_thread."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _format_messages(messages)
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        if mode == "a":
            f.write("\n---\n\n")
        f.write(content)


async def archive_conversation(
    session_id: str,
    messages: list[Message],
    output_dir: str | Path,
) -> None:
    """Append dropped messages to a markdown archive file.

    Args:
        session_id: The session identifier (first 8 chars used in filename).
        messages: Messages to archive.
        output_dir: Directory to write archive files into (will be created).
    """
    if not messages:
        return

    archive_dir = Path(output_dir).expanduser()
    path = _archive_path(archive_dir, session_id)

    try:
        await asyncio.to_thread(_write_archive, path, messages)
        logger.debug("Archived %d messages to %s", len(messages), path)
    except Exception:
        logger.warning("Failed to archive %d messages to %s", len(messages), path, exc_info=True)
