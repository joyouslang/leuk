"""MemoryLoader: reads hierarchical memory files and returns combined context."""

from __future__ import annotations

from pathlib import Path


class MemoryLoader:
    """Discovers and reads memory files from the hierarchy.

    Memory files:
      - ``{memory_dir}/GLOBAL.md``                        — global, always loaded
      - ``{memory_dir}/projects/{project_name}/MEMORY.md`` — per-project

    Token budget enforcement truncates from the top of global memory so that
    project-specific context is always preserved.
    """

    def __init__(
        self,
        memory_dir: str = "~/.config/leuk/memory",
        project_name: str = "",
        token_budget: int = 4000,
    ) -> None:
        self.memory_dir = Path(memory_dir).expanduser()
        self._project_name = project_name
        self.token_budget = token_budget

    # ------------------------------------------------------------------
    # Project detection
    # ------------------------------------------------------------------

    def detect_project_name(self) -> str:
        """Return the effective project name.

        Resolution order:
        1. Explicit ``project_name`` passed at construction.
        2. Nearest ``.git`` directory ancestor's folder name.
        3. Current working directory name.
        """
        if self._project_name:
            return self._project_name

        current = Path.cwd()
        while current != current.parent:
            if (current / ".git").exists():
                return current.name
            current = current.parent

        return Path.cwd().name

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: 1 token ≈ 4 characters."""
        return len(text) // 4

    def load(self) -> str:
        """Return combined memory context string, or empty string if none exists.

        The returned string is ready to be prepended to the system prompt.
        """
        project_name = self.detect_project_name()
        global_file = self.memory_dir / "GLOBAL.md"
        project_file = self.memory_dir / "projects" / project_name / "MEMORY.md"

        global_content = global_file.read_text(encoding="utf-8") if global_file.exists() else ""
        project_content = project_file.read_text(encoding="utf-8") if project_file.exists() else ""

        if not global_content and not project_content:
            return ""

        project_section = (
            f"## Project Memory ({project_name})\n\n{project_content.strip()}\n\n"
            if project_content
            else ""
        )
        global_section = (
            f"## Global Memory\n\n{global_content.strip()}\n\n"
            if global_content
            else ""
        )

        # Apply token budget: truncate from top of global memory to stay within budget
        budget_chars = self.token_budget * 4
        project_chars = len(project_section)
        available_for_global = max(0, budget_chars - project_chars)

        if available_for_global < len(global_section):
            if available_for_global == 0:
                global_section = ""
            else:
                # Keep the tail (most recent content) of global memory
                global_section = global_section[-available_for_global:]
                # Trim to the next newline so we don't start mid-line
                newline_idx = global_section.find("\n")
                if newline_idx > 0:
                    global_section = global_section[newline_idx + 1 :]

        combined = global_section + project_section
        return combined.strip()
