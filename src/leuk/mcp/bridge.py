"""Bridge MCP server tools into the leuk tool registry."""

from __future__ import annotations

import logging
from typing import Any

from leuk.mcp.client import MCPClient
from leuk.tools.base import ToolRegistry
from leuk.types import ToolSpec

logger = logging.getLogger(__name__)


class MCPToolBridge:
    """Wraps an MCPClient's tools as standard leuk Tools.

    Each MCP tool is registered in the ToolRegistry so the agent can
    call it like any built-in tool.
    """

    def __init__(self, client: MCPClient) -> None:
        self._client = client
        self._tool_specs: list[ToolSpec] = []

    def register_tools(self, registry: ToolRegistry) -> None:
        """Discover and register all MCP server tools into the registry."""
        self._tool_specs = self._client.tool_specs()
        for spec in self._tool_specs:
            proxy = _MCPToolProxy(client=self._client, tool_spec=spec)
            registry.register(proxy)
        logger.info(
            "Registered %d MCP tools from [%s]",
            len(self._tool_specs),
            self._client.name,
        )


class _MCPToolProxy:
    """A proxy Tool that forwards execution to an MCP server."""

    def __init__(self, *, client: MCPClient, tool_spec: ToolSpec) -> None:
        self._client = client
        self._spec = tool_spec

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, arguments: dict[str, Any]) -> str:
        return await self._client.call_tool(self._spec.name, arguments)
