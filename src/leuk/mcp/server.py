"""MCP server that exposes leuk sessions and messaging as tools.

Start the server in stdio mode (another MCP client launches this as a subprocess)::

    leuk --mcp-server

Or call ``LeukMCPServer.run_stdio()`` from within an async context.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP

from leuk.persistence.base import DurableStore, HotStore
from leuk.types import Session, SessionStatus

logger = logging.getLogger(__name__)


class LeukMCPServer:
    """MCP server exposing leuk session control and history retrieval.

    Parameters
    ----------
    sqlite:
        Durable store for sessions and message history.
    hot_store:
        Ephemeral store for active-session context.
    sessions:
        Registry of running AgentSession objects keyed by session ID.
        The REPL (or any other host) keeps this dict updated as sessions
        are created and destroyed.
    """

    def __init__(
        self,
        sqlite: DurableStore,
        hot_store: HotStore,
        sessions: dict[str, Any],  # dict[str, AgentSession] — avoid circular import
    ) -> None:
        self._sqlite = sqlite
        self._hot_store = hot_store
        self._sessions = sessions
        self._mcp = self._build()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run_stdio(self) -> None:
        """Run the server using stdio transport (blocking until EOF)."""
        await self._mcp.run_stdio_async()

    @property
    def mcp(self) -> FastMCP:
        """Expose the underlying FastMCP instance (e.g. for testing)."""
        return self._mcp

    # ------------------------------------------------------------------
    # Internal: tool registration
    # ------------------------------------------------------------------

    def _build(self) -> FastMCP:
        mcp = FastMCP("leuk")

        # Capture self in closures
        sqlite = self._sqlite
        sessions = self._sessions

        # ── list_sessions ──────────────────────────────────────────────

        @mcp.tool()
        async def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
            """Return active and recent sessions with metadata.

            Parameters
            ----------
            limit:
                Maximum number of sessions to return (most recently updated first).
            """
            rows = await sqlite.list_sessions(limit=limit)
            result = []
            for s in rows:
                result.append(
                    {
                        "session_id": s.id,
                        "status": s.status.value,
                        "created_at": s.created_at.isoformat(),
                        "updated_at": s.updated_at.isoformat(),
                        "system_prompt": s.system_prompt[:200] if s.system_prompt else "",
                        "is_running": s.id in sessions,
                        "metadata": s.metadata,
                    }
                )
            return result

        # ── get_history ────────────────────────────────────────────────

        @mcp.tool()
        async def get_history(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
            """Fetch recent messages from a session.

            Parameters
            ----------
            session_id:
                The session to query.
            limit:
                Maximum number of messages to return (most recent first).
            """
            session = await sqlite.get_session(session_id)
            if session is None:
                return [{"error": f"Session {session_id!r} not found"}]

            messages = await sqlite.get_messages(session_id)
            # Return the *last* `limit` messages, most recent last
            tail = messages[-limit:] if len(messages) > limit else messages
            result = []
            for m in tail:
                entry: dict[str, Any] = {
                    "role": m.role.value,
                    "timestamp": m.timestamp.isoformat(),
                }
                if m.content:
                    entry["content"] = m.content
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in m.tool_calls
                    ]
                if m.tool_result:
                    entry["tool_result"] = {
                        "name": m.tool_result.name,
                        "content": m.tool_result.content,
                        "is_error": m.tool_result.is_error,
                    }
                result.append(entry)
            return result

        # ── send_message ───────────────────────────────────────────────

        @mcp.tool()
        async def send_message(session_id: str, message: str) -> dict[str, Any]:
            """Inject a user message into an active agent session.

            The message is enqueued; the session's background loop picks it up
            asynchronously.  If the session is not currently running, the
            message cannot be delivered.

            Parameters
            ----------
            session_id:
                Target session ID (must be in the active sessions registry).
            message:
                The text to inject as a user message.
            """
            agent_session = sessions.get(session_id)
            if agent_session is None:
                return {
                    "success": False,
                    "error": (
                        f"Session {session_id!r} is not currently running. "
                        "Use list_sessions() to find active sessions."
                    ),
                }

            agent_session.push(message)
            logger.info("MCP server: injected message into session %s", session_id)
            return {"success": True, "session_id": session_id}

        # ── create_session ─────────────────────────────────────────────

        @mcp.tool()
        async def create_session(system_prompt: str = "") -> dict[str, Any]:
            """Create a new agent session record.

            The session is persisted in SQLite and returned with its ID.
            To send messages to it, a running agent loop must be attached
            (e.g. by the REPL or ``--mcp-server`` mode).

            Parameters
            ----------
            system_prompt:
                Optional system prompt override for this session.
            """
            now = datetime.now(timezone.utc)
            session = Session(
                id=uuid.uuid4().hex,
                status=SessionStatus.ACTIVE,
                created_at=now,
                updated_at=now,
                system_prompt=system_prompt,
            )
            await sqlite.create_session(session)
            logger.info("MCP server: created session %s", session.id)
            return {
                "session_id": session.id,
                "status": session.status.value,
                "created_at": session.created_at.isoformat(),
                "system_prompt": system_prompt,
            }

        return mcp
