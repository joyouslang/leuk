"""File reading tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from leuk.types import ToolSpec

_MAX_SIZE = 512_000  # 512 KB


class FileReadTool:
    """Read file contents from the local filesystem."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_read",
            description=(
                "Read the contents of a file. Returns the file text with line numbers. "
                "Optionally read a specific line range."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Starting line number (0-based, optional)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to return (optional)",
                    },
                },
                "required": ["path"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        path = Path(arguments["path"]).expanduser()
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit")

        if not path.exists():
            return f"[ERROR] File not found: {path}"
        if not path.is_file():
            return f"[ERROR] Not a file: {path}"
        if path.stat().st_size > _MAX_SIZE:
            return f"[ERROR] File too large ({path.stat().st_size} bytes, max {_MAX_SIZE})"

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"[ERROR] Cannot read file: {exc}"

        lines = text.splitlines(keepends=True)
        total = len(lines)

        # Apply offset and limit
        end = total if limit is None else min(offset + limit, total)
        selected = lines[offset:end]

        # Format with line numbers
        numbered = []
        for i, line in enumerate(selected, start=offset + 1):
            numbered.append(f"{i:6d}\t{line.rstrip()}")

        header = f"[{path}] ({total} lines total)"
        if offset > 0 or limit is not None:
            header += f" showing lines {offset + 1}-{offset + len(selected)}"

        return header + "\n" + "\n".join(numbered)
