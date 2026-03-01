"""File editing tool using exact string replacement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from leuk.types import ToolSpec


class FileEditTool:
    """Edit files by replacing exact string matches, or create new files."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_edit",
            description=(
                "Edit a file by replacing an exact string match with new text, "
                "or create a new file by providing only 'new_string'. "
                "To delete text, provide an empty 'new_string'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit or create",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to find and replace (omit to create a new file)",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default false)",
                    },
                },
                "required": ["path", "new_string"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        path = Path(arguments["path"]).expanduser()
        new_string: str = arguments["new_string"]
        old_string: str | None = arguments.get("old_string")
        replace_all: bool = arguments.get("replace_all", False)

        # Create mode: no old_string means write a new file
        if old_string is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_string, encoding="utf-8")
            return f"Created {path} ({len(new_string)} chars)"

        # Edit mode: replace old_string with new_string
        if not path.exists():
            return f"[ERROR] File not found: {path}"

        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"[ERROR] Cannot read file: {exc}"

        count = content.count(old_string)
        if count == 0:
            return "[ERROR] old_string not found in file"
        if count > 1 and not replace_all:
            return (
                f"[ERROR] old_string found {count} times. "
                "Provide more context to make it unique, or set replace_all=true."
            )

        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)

        path.write_text(new_content, encoding="utf-8")
        replacements = count if replace_all else 1
        return f"Edited {path}: {replacements} replacement(s) made"
