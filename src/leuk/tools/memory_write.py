"""memory_write tool: appends content to global or project-scoped memory files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from leuk.types import ToolSpec


class MemoryWriteTool:
    """Append content to persistent memory files (global or per-project).

    Memory files are plain markdown and are loaded at the start of each new
    session via :class:`leuk.memory.MemoryLoader`.
    """

    def __init__(
        self,
        memory_dir: str = "~/.config/leuk/memory",
        project_name: str = "",
    ) -> None:
        self.memory_dir = Path(memory_dir).expanduser()
        self.project_name = project_name

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="memory_write",
            description=(
                "Append content to a persistent memory file that is loaded at the start "
                "of every new session. Use 'global' scope for information relevant across "
                "all projects, or 'project' scope for project-specific context."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["global", "project"],
                        "description": (
                            "'global' writes to ~/.config/leuk/memory/GLOBAL.md; "
                            "'project' writes to the current project's MEMORY.md"
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "Markdown content to append to the memory file",
                    },
                },
                "required": ["scope", "content"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        scope = arguments["scope"]
        content = arguments["content"]

        if scope == "global":
            target = self.memory_dir / "GLOBAL.md"
        elif scope == "project":
            if not self.project_name:
                return "[ERROR] No project name is configured; cannot write project-scoped memory"
            target = self.memory_dir / "projects" / self.project_name / "MEMORY.md"
        else:
            return f"[ERROR] Unknown scope '{scope}'. Use 'global' or 'project'"

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as f:
                f.write(f"\n{content}\n")
        except OSError as exc:
            return f"[ERROR] Failed to write memory: {exc}"

        return f"Appended to {scope} memory ({target})"
