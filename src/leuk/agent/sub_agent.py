"""Sub-agent orchestration: spawn child agents as async tasks."""

from __future__ import annotations

import asyncio
import logging

from leuk.config import RoleDefinition, Settings
from leuk.persistence.base import HotStore
from leuk.persistence.sqlite import SQLiteStore
from leuk.providers.base import LLMProvider
from leuk.tools.base import ToolRegistry
from leuk.types import Message, Session

logger = logging.getLogger(__name__)


class SubAgentManager:
    """Manages in-process sub-agents for parallel task execution.

    Sub-agents share the same provider, persistence backends, and tool registry
    but each gets its own session and message history.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        provider: LLMProvider,
        tool_registry: ToolRegistry,
        sqlite: SQLiteStore,
        hot_store: HotStore,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._tools = tool_registry
        self._sqlite = sqlite
        self._hot_store = hot_store
        self._active_tasks: dict[str, asyncio.Task[list[Message]]] = {}
        self._max_concurrent = settings.agent.max_concurrent_sub_agents
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._queued_count = 0
        self._runtime_roles: dict[str, RoleDefinition] = {}

    # ------------------------------------------------------------------
    # Role management
    # ------------------------------------------------------------------

    def define_role(self, name: str, role: RoleDefinition) -> None:
        """Register a runtime role definition (overrides config-defined roles)."""
        self._runtime_roles[name] = role

    def _resolve_role(self, name: str) -> RoleDefinition | None:
        """Look up a role by name; runtime definitions take precedence over config."""
        if name in self._runtime_roles:
            return self._runtime_roles[name]
        return self._settings.agent_teams.roles.get(name)

    def _make_role_registry(self, role: RoleDefinition) -> ToolRegistry:
        """Return a ToolRegistry filtered to the role's allowed tool names.

        If the role's ``tools`` list is empty, the full registry is returned.
        """
        if not role.tools:
            return self._tools
        filtered = ToolRegistry()
        for tool_name in role.tools:
            tool = self._tools.get(tool_name)
            if tool is not None:
                filtered.register(tool)
            else:
                logger.debug("Role tool %r not found in registry, skipping", tool_name)
        return filtered

    # ------------------------------------------------------------------
    # Spawn / wait
    # ------------------------------------------------------------------

    async def spawn(
        self,
        task_description: str,
        *,
        system_prompt: str | None = None,
        role: str | None = None,
        parent_session_id: str | None = None,
    ) -> str:
        """Spawn a sub-agent to handle a task.

        If *role* is provided, the role's ``system_prompt``, ``tools`` allowlist,
        and optional ``provider`` override are applied.  An explicit
        *system_prompt* argument takes precedence over the role's system prompt.

        Returns the sub-agent's session ID.
        """
        from leuk.agent.core import Agent

        provider = self._provider
        tool_registry = self._tools
        max_rounds_override: int | None = None

        if role is not None:
            role_def = self._resolve_role(role)
            if role_def is None:
                logger.warning("Unknown role %r, spawning without role config", role)
            else:
                # System prompt: explicit arg > role definition > default
                if system_prompt is None and role_def.system_prompt:
                    system_prompt = role_def.system_prompt

                # Tool subset
                tool_registry = self._make_role_registry(role_def)

                # Provider override
                if role_def.provider and role_def.provider != self._settings.llm.provider:
                    from leuk.providers import create_provider

                    override_config = self._settings.llm.model_copy(
                        update={"provider": role_def.provider}
                    )
                    try:
                        provider = create_provider(override_config)
                        logger.info(
                            "Role %r using provider override %r", role, role_def.provider
                        )
                    except Exception as exc:
                        logger.warning(
                            "Could not create provider %r for role %r: %s — using default",
                            role_def.provider,
                            role,
                            exc,
                        )

                # max_rounds override
                if role_def.max_rounds > 0:
                    max_rounds_override = role_def.max_rounds

        # Build a settings copy with the role's max_rounds if needed
        effective_settings = self._settings
        if max_rounds_override is not None:
            effective_settings = self._settings.model_copy(
                update={"agent": self._settings.agent.model_copy(
                    update={"max_rounds": max_rounds_override}
                )}
            )

        session = Session(
            system_prompt=system_prompt or self._settings.agent.system_prompt,
            parent_session_id=parent_session_id,
            metadata={"task": task_description, **({"role": role} if role else {})},
        )

        agent = Agent(
            settings=effective_settings,
            provider=provider,
            tool_registry=tool_registry,
            sqlite=self._sqlite,
            hot_store=self._hot_store,
            session=session,
        )
        await agent.init()

        async def _run_sub_agent() -> list[Message]:
            results: list[Message] = []
            self._queued_count += 1
            try:
                await self._semaphore.acquire()
            except BaseException:
                self._queued_count -= 1
                raise
            self._queued_count -= 1
            try:
                async for msg in agent.run(task_description):
                    results.append(msg)
                await agent.shutdown()
            finally:
                self._semaphore.release()
            return results

        task = asyncio.create_task(_run_sub_agent(), name=f"sub-agent-{session.id[:8]}")
        self._active_tasks[session.id] = task
        logger.info(
            "Spawned sub-agent %s%s: %s",
            session.id[:8],
            f" (role={role!r})" if role else "",
            task_description[:80],
        )
        return session.id

    async def wait(self, session_id: str) -> list[Message]:
        """Wait for a sub-agent to complete and return its messages."""
        task = self._active_tasks.get(session_id)
        if task is None:
            raise ValueError(f"No active sub-agent with session {session_id}")
        try:
            return await task
        finally:
            self._active_tasks.pop(session_id, None)

    async def wait_all(self) -> dict[str, list[Message]]:
        """Wait for all active sub-agents to complete."""
        results: dict[str, list[Message]] = {}
        for sid, task in list(self._active_tasks.items()):
            try:
                results[sid] = await task
            except Exception:
                logger.exception("Sub-agent %s failed", sid[:8])
                results[sid] = []
            finally:
                self._active_tasks.pop(sid, None)
        return results

    @property
    def active_count(self) -> int:
        return len(self._active_tasks)

    @property
    def queued_count(self) -> int:
        """Number of sub-agents waiting for a concurrency slot."""
        return self._queued_count
