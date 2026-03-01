"""Sub-agent delegation tool -- lets the LLM spawn child agents for tasks."""

from __future__ import annotations

from typing import Any

from leuk.types import Role, ToolSpec


class SubAgentTool:
    """Tool that allows the agent to delegate tasks to sub-agents.

    Must be initialised with a reference to a SubAgentManager instance.
    """

    def __init__(self) -> None:
        self._manager: Any = None  # Set after construction via set_manager()

    def set_manager(self, manager: Any) -> None:
        """Inject the SubAgentManager after construction (avoids circular deps)."""
        self._manager = manager

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="sub_agent",
            description=(
                "Delegate a task to an independent sub-agent. The sub-agent gets its own "
                "conversation and can use all available tools. Use this for complex tasks "
                "that can be worked on independently, or when you need to explore multiple "
                "approaches in parallel. Returns the sub-agent's final response."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "A detailed description of the task for the sub-agent to perform",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": (
                            "Optional custom system prompt for the sub-agent. "
                            "If omitted, the default agent system prompt is used."
                        ),
                    },
                },
                "required": ["task"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        if self._manager is None:
            return "[ERROR] Sub-agent manager not initialised"

        task = arguments["task"]
        system_prompt = arguments.get("system_prompt")

        try:
            session_id = await self._manager.spawn(
                task,
                system_prompt=system_prompt,
                parent_session_id=getattr(self._manager, "_parent_session_id", None),
            )
            # Wait for the sub-agent to complete
            messages = await self._manager.wait(session_id)

            # Extract the final assistant response
            for msg in reversed(messages):
                if msg.role == Role.ASSISTANT and msg.content:
                    return f"[Sub-agent {session_id[:8]}] {msg.content}"

            return f"[Sub-agent {session_id[:8]}] Task completed but produced no text response."
        except Exception as exc:
            return f"[ERROR] Sub-agent failed: {exc}"
