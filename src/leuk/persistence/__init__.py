"""Persistence layer: SQLite for durable storage, Redis for hot state."""

from __future__ import annotations

import logging

from leuk.config import RedisConfig
from leuk.persistence.base import DurableStore, HotStore

logger = logging.getLogger(__name__)


async def create_hot_store(config: RedisConfig) -> HotStore:
    """Create a Redis hot store, falling back to in-memory if Redis is unavailable."""
    try:
        from leuk.persistence.redis import RedisStore

        store = RedisStore(config)
        # Probe the connection to fail fast
        await store.get_active_session()
        logger.info("Connected to Redis at %s", config.url)
        return store
    except Exception as exc:
        logger.warning("Redis unavailable (%s), using in-memory hot store", exc)
        from leuk.persistence.memory import MemoryStore

        return MemoryStore()


__all__ = ["DurableStore", "HotStore", "create_hot_store"]
