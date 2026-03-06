"""In-memory hot state store for active session context."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class MemoryStore:
    """In-memory implementation of the HotStore protocol.

    Data does not survive process restarts, but sessions are still fully
    persisted via SQLite.
    """

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        logger.info("Using in-memory hot store")

    async def set_context(self, session_id: str, messages_json: str) -> None:
        self._data[f"ctx:{session_id}"] = messages_json

    async def get_context(self, session_id: str) -> str | None:
        return self._data.get(f"ctx:{session_id}")

    async def delete_context(self, session_id: str) -> None:
        self._data.pop(f"ctx:{session_id}", None)

    async def set_active_session(self, session_id: str) -> None:
        self._data["active_session"] = session_id

    async def get_active_session(self) -> str | None:
        return self._data.get("active_session")

    async def close(self) -> None:
        self._data.clear()
