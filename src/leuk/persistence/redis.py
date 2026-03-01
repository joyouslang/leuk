"""Redis-backed hot state store."""

from __future__ import annotations

import redis.asyncio as aioredis

from leuk.config import RedisConfig


class RedisStore:
    """Ephemeral hot-state using Redis for active conversation context."""

    def __init__(self, config: RedisConfig) -> None:
        self._config = config
        self._client: aioredis.Redis = aioredis.from_url(
            config.url, decode_responses=True
        )
        self._prefix = config.prefix
        self._ttl = config.ttl_seconds

    def _key(self, *parts: str) -> str:
        return self._prefix + ":".join(parts)

    # ------------------------------------------------------------------
    # Context cache
    # ------------------------------------------------------------------

    async def set_context(self, session_id: str, messages_json: str) -> None:
        await self._client.set(
            self._key("ctx", session_id), messages_json, ex=self._ttl
        )

    async def get_context(self, session_id: str) -> str | None:
        return await self._client.get(self._key("ctx", session_id))

    async def delete_context(self, session_id: str) -> None:
        await self._client.delete(self._key("ctx", session_id))

    # ------------------------------------------------------------------
    # Active session tracking
    # ------------------------------------------------------------------

    async def set_active_session(self, session_id: str) -> None:
        await self._client.set(self._key("active_session"), session_id)

    async def get_active_session(self) -> str | None:
        return await self._client.get(self._key("active_session"))

    async def close(self) -> None:
        await self._client.aclose()
