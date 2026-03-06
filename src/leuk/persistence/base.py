"""Abstract persistence protocols."""

from __future__ import annotations

from typing import Protocol

from leuk.types import Message, Session


class DurableStore(Protocol):
    """Long-term storage for sessions and message history (SQLite)."""

    async def init(self) -> None:
        """Run migrations / create tables."""
        ...

    async def create_session(self, session: Session) -> None: ...

    async def get_session(self, session_id: str) -> Session | None: ...

    async def update_session(self, session: Session) -> None: ...

    async def list_sessions(self, *, limit: int = 20) -> list[Session]: ...

    async def append_message(self, session_id: str, message: Message) -> None: ...

    async def get_messages(self, session_id: str) -> list[Message]: ...

    async def close(self) -> None: ...


class HotStore(Protocol):
    """Fast ephemeral state for active sessions (in-memory)."""

    async def set_context(self, session_id: str, messages_json: str) -> None:
        """Cache the current conversation context."""
        ...

    async def get_context(self, session_id: str) -> str | None:
        """Retrieve cached context, or None if expired/missing."""
        ...

    async def delete_context(self, session_id: str) -> None: ...

    async def set_active_session(self, session_id: str) -> None:
        """Mark a session as the currently active one."""
        ...

    async def get_active_session(self) -> str | None: ...

    async def close(self) -> None: ...
