"""SkillTool: exposes installed SKILL.md skills to the model (progressive disclosure).

The tool's ``spec.description`` lists every usable skill (``name — description``)
so the model knows what is available without spending tokens on full bodies; it
calls ``read`` to pull a skill's full instructions on demand, then carries them
out with the existing shell/file tools. leuk never auto-runs skill scripts.
"""

from __future__ import annotations

from typing import Any

from leuk.skills.loader import SkillLoader
from leuk.types import ToolSpec

name = "skill"


class SkillTool:
    """A single tool that lists and reads installed agent skills."""

    name = "skill"

    def __init__(self, loader: SkillLoader) -> None:
        self._loader = loader

    @property
    def spec(self) -> ToolSpec:
        usable = self._loader.usable()
        if usable:
            index = "\n".join(f"- {m.name}: {m.description}" for m in usable)
            available = f"\n\nAvailable skills:\n{index}"
        else:
            available = "\n\n(No skills are currently installed and trusted.)"
        return ToolSpec(
            name=self.name,
            description=(
                "Access installed Agent Skills — reusable, self-contained playbooks "
                "(SKILL.md). Call action='read' with a skill's name to load its full "
                "instructions when a task matches it, then follow them using your other "
                "tools (shell, file edit, etc.). action='list' returns the same index."
                + available
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "read"]},
                    "name": {
                        "type": "string",
                        "description": "Skill name (or slug) to read; required for action='read'.",
                    },
                },
                "required": ["action"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        action = arguments.get("action")
        if action == "list":
            usable = self._loader.usable()
            if not usable:
                return "No skills installed and trusted. Install with `/skills` or `leuk skills add`."
            return "\n".join(f"- {m.name}: {m.description}" for m in usable)
        if action == "read":
            target = str(arguments.get("name", "")).strip()
            if not target:
                return "[ERROR] action='read' requires a 'name'."
            body = self._loader.read(target)
            if body is None:
                return (
                    f"[ERROR] No usable skill named {target!r} "
                    "(it may be uninstalled, untrusted, or disabled)."
                )
            return body
        return f"[ERROR] Unknown action {action!r} (use 'list' or 'read')."
