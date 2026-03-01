"""Abstract tool protocol and registry."""

from __future__ import annotations

from typing import Any, Protocol

from leuk.types import ToolSpec


class Tool(Protocol):
    """Interface for agent tools."""

    @property
    def spec(self) -> ToolSpec:
        """Return the JSON-schema tool specification."""
        ...

    async def execute(self, arguments: dict[str, Any]) -> str:
        """Run the tool with the given arguments and return a string result."""
        ...


class ToolRegistry:
    """Registry mapping tool names to implementations."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
