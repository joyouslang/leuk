"""Sub-agent delegation tool -- lets the LLM spawn child agents for tasks."""

from __future__ import annotations

import logging
from typing import Any

from leuk.types import Role, ToolSpec

logger = logging.getLogger(__name__)


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
                    "role": {
                        "type": "string",
                        "description": (
                            "Optional role name to use for the sub-agent. "
                            "Roles define a custom system prompt, tool subset, and optional provider. "
                            "Built-in roles: 'researcher' (web_fetch), 'coder' (shell + file ops), "
                            "'reviewer' (file_read only). Custom roles can be defined via AgentTeam."
                        ),
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": (
                            "Optional custom system prompt for the sub-agent. "
                            "If omitted and a role is specified, the role's system prompt is used. "
                            "If omitted entirely, the default agent system prompt is used."
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
        role = arguments.get("role")
        system_prompt = arguments.get("system_prompt")

        queued_warning = ""
        if self._manager.active_count >= self._manager._max_concurrent:
            logger.warning(
                "Sub-agent concurrency limit reached (%d/%d active); new request will queue",
                self._manager.active_count,
                self._manager._max_concurrent,
            )
            queued_warning = (
                f"[WARNING: concurrency limit reached ({self._manager.active_count}/"
                f"{self._manager._max_concurrent} active); this sub-agent is queued] "
            )

        try:
            session_id = await self._manager.spawn(
                task,
                role=role,
                system_prompt=system_prompt,
                parent_session_id=getattr(self._manager, "_parent_session_id", None),
            )
            # Wait for the sub-agent to complete
            messages = await self._manager.wait(session_id)

            # Extract the final assistant response
            for msg in reversed(messages):
                if msg.role == Role.ASSISTANT and msg.content:
                    return f"{queued_warning}[Sub-agent {session_id[:8]}] {msg.content}"

            return f"{queued_warning}[Sub-agent {session_id[:8]}] Task completed but produced no text response."
        except Exception as exc:
            return f"[ERROR] Sub-agent failed: {exc}"
