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
                "Change a file with a targeted **patch**: replace an exact "
                "'old_string' match with 'new_string' (set replace_all=true to "
                "change every occurrence; an empty 'new_string' deletes the text). "
                "Make the smallest change needed — never rewrite a whole existing "
                "file. Omitting 'old_string' is **create mode**, for NEW files only. "
                "To replace an existing file's entire contents (rare, last resort) "
                "you must pass overwrite=true, which requires user approval."
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
                        "description": (
                            "Exact text to find and replace. Omit ONLY to create a new "
                            "file; to change an existing file always provide this so the "
                            "edit is a patch, not a full rewrite."
                        ),
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default false)",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": (
                            "Replace an existing file's ENTIRE contents (default false). "
                            "Discouraged — prefer a patch (old_string/new_string). "
                            "Requires user approval."
                        ),
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
        overwrite: bool = arguments.get("overwrite", False)

        # Create mode: no old_string means write the file from scratch.
        if old_string is None:
            if path.exists() and not overwrite:
                # Don't clobber an existing file with a full rewrite. Require an
                # explicit patch, or an explicit (approved) overwrite=true.
                return (
                    f"[ERROR] {path} already exists. Don't rewrite the whole file — "
                    "change only what's needed with a patch (provide 'old_string' "
                    "and 'new_string'). To replace the entire file anyway (rarely "
                    "needed), pass overwrite=true; this requires user approval."
                )
            existed = path.exists()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_string, encoding="utf-8")
            verb = "Overwrote" if existed else "Created"
            return f"{verb} {path} ({len(new_string)} chars)"

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
