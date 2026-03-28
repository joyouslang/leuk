"""Agent team orchestration: role-based sub-agent groups."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from leuk.config import RoleDefinition
from leuk.types import Message

if TYPE_CHECKING:
    from leuk.agent.sub_agent import SubAgentManager

logger = logging.getLogger(__name__)


class AgentTeam:
    """Manages a team of agents with distinct roles.

    Roles can be defined at runtime via :meth:`define_role` or loaded from
    ``settings.agent_teams.roles``.  Each role specifies a system prompt,
    an allowed tool subset, and an optional provider override.

    Typical usage::

        team = AgentTeam(manager)
        team.define_role("analyst", system_prompt="...", tools=["file_read"])
        sid1 = await team.spawn("analyst", "Summarise the logs in /var/log/app.log")
        sid2 = await team.spawn("coder", "Write a script to parse those logs")
        results = await team.collect()
    """

    def __init__(self, manager: SubAgentManager) -> None:
        self._manager = manager

    # ------------------------------------------------------------------
    # Role registration
    # ------------------------------------------------------------------

    def define_role(
        self,
        name: str,
        system_prompt: str,
        tools: list[str],
        provider: str = "",
        max_rounds: int = 0,
    ) -> None:
        """Register (or overwrite) a runtime role definition.

        Runtime definitions take precedence over roles defined in config.
        """
        self._manager.define_role(
            name,
            RoleDefinition(
                system_prompt=system_prompt,
                tools=tools,
                provider=provider,
                max_rounds=max_rounds,
            ),
        )
        logger.debug("Defined role %r with tools %s", name, tools)

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------

    async def spawn(self, role: str, task: str) -> str:
        """Spawn a sub-agent configured for *role* to handle *task*.

        Returns the sub-agent's session ID.  Raises ``ValueError`` if the
        role is not defined.
        """
        role_def = self._manager._resolve_role(role)
        if role_def is None:
            raise ValueError(
                f"Unknown role {role!r}. "
                "Define it with AgentTeam.define_role() or add it to settings.agent_teams.roles."
            )
        return await self._manager.spawn(task, role=role)

    # ------------------------------------------------------------------
    # Broadcast / collect
    # ------------------------------------------------------------------

    async def broadcast(self, message: str) -> dict[str, str]:
        """Spawn *message* as a task across all defined roles.

        Returns a mapping of ``{session_id: role_name}`` for the spawned
        agents.  Use :meth:`collect` afterwards to gather results.
        """
        all_roles = {
            **self._manager._settings.agent_teams.roles,
            **self._manager._runtime_roles,
        }
        spawned: dict[str, str] = {}
        for role_name in all_roles:
            sid = await self._manager.spawn(message, role=role_name)
            spawned[sid] = role_name
            logger.info("Broadcast to role %r → session %s", role_name, sid[:8])
        return spawned

    async def collect(self) -> dict[str, list[Message]]:
        """Wait for all active team members to complete and return their messages.

        Returns a mapping of ``{session_id: [messages]}``.
        """
        return await self._manager.wait_all()
