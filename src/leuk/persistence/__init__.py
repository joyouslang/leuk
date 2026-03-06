"""Persistence layer: SQLite for durable storage, MemoryStore for hot state."""

from __future__ import annotations

from leuk.persistence.base import DurableStore, HotStore
from leuk.persistence.memory import MemoryStore


def create_hot_store() -> MemoryStore:
    """Create an in-memory hot store for session context caching."""
    return MemoryStore()


__all__ = ["DurableStore", "HotStore", "MemoryStore", "create_hot_store"]
