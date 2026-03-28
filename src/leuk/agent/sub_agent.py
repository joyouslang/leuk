"""Sub-agent orchestration: spawn child agents as async tasks."""

from __future__ import annotations

import asyncio
import logging

from leuk.config import Settings
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

    async def spawn(
        self,
        task_description: str,
        *,
        system_prompt: str | None = None,
        parent_session_id: str | None = None,
    ) -> str:
        """Spawn a sub-agent to handle a task.

        Returns the sub-agent's session ID.
        """
        from leuk.agent.core import Agent

        session = Session(
            system_prompt=system_prompt or self._settings.agent.system_prompt,
            parent_session_id=parent_session_id,
            metadata={"task": task_description},
        )

        agent = Agent(
            settings=self._settings,
            provider=self._provider,
            tool_registry=self._tools,
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
        logger.info("Spawned sub-agent %s: %s", session.id[:8], task_description[:80])
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
