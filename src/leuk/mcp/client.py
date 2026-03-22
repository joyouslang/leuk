"""MCP client: connects to external MCP servers and bridges their tools."""

from __future__ import annotations

import logging
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client

from leuk.types import ToolSpec

logger = logging.getLogger(__name__)


class MCPClient:
    """Client that connects to an MCP server and exposes its tools.

    Supports two transports:
    - stdio: launches a subprocess (e.g. `npx @modelcontextprotocol/server-filesystem /tmp`)
    - sse: connects to an HTTP SSE endpoint

    Usage:
        client = MCPClient.stdio("npx", ["-y", "@mcp/server-filesystem", "/tmp"])
        await client.connect()
        tools = client.tool_specs()
        result = await client.call_tool("read_file", {"path": "/tmp/foo.txt"})
        await client.close()
    """

    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._context_manager: Any = None  # The read/write streams context manager
        self._streams_cm: Any = None
        self._tools: list[dict[str, Any]] = []
        self._transport: str = ""
        self._stdio_params: StdioServerParameters | None = None
        self._sse_url: str = ""
        self._name: str = ""

    @classmethod
    def stdio(cls, command: str, args: list[str] | None = None, *, name: str = "") -> MCPClient:
        """Create an MCP client that uses stdio transport."""
        client = cls()
        client._transport = "stdio"
        client._stdio_params = StdioServerParameters(command=command, args=args or [])
        client._name = name or command
        return client

    @classmethod
    def sse(cls, url: str, *, name: str = "") -> MCPClient:
        """Create an MCP client that uses SSE transport."""
        client = cls()
        client._transport = "sse"
        client._sse_url = url
        client._name = name or url
        return client

    async def connect(self) -> None:
        """Establish connection to the MCP server and discover tools."""
        if self._transport == "stdio":
            assert self._stdio_params is not None
            self._streams_cm = stdio_client(self._stdio_params)
            read_stream, write_stream = await self._streams_cm.__aenter__()
        elif self._transport == "sse":
            self._streams_cm = sse_client(self._sse_url)
            read_stream, write_stream = await self._streams_cm.__aenter__()
        else:
            raise ValueError(f"Unknown transport: {self._transport}")

        self._session = ClientSession(read_stream, write_stream)
        self._context_manager = self._session
        await self._session.__aenter__()
        await self._session.initialize()

        # Discover tools
        tools_result = await self._session.list_tools()
        self._tools = []
        for tool in tools_result.tools:
            self._tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
                }
            )

        logger.info(
            "MCP [%s]: connected, discovered %d tools: %s",
            self._name,
            len(self._tools),
            [t["name"] for t in self._tools],
        )

    def tool_specs(self) -> list[ToolSpec]:
        """Return ToolSpec objects for all tools exposed by this MCP server."""
        specs = []
        for t in self._tools:
            # Prefix tool names with server name to avoid collisions
            name = f"mcp_{self._name}_{t['name']}" if self._name else t["name"]
            # Sanitize name: only alphanumeric and underscores
            name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
            specs.append(
                ToolSpec(
                    name=name,
                    description=f"[MCP: {self._name}] {t['description']}",
                    parameters=t.get("input_schema", {"type": "object", "properties": {}}),
                )
            )
        return specs

    def _original_tool_name(self, prefixed_name: str) -> str | None:
        """Map a prefixed tool name back to the original MCP tool name."""
        for t in self._tools:
            expected = f"mcp_{self._name}_{t['name']}"
            expected = "".join(c if c.isalnum() or c == "_" else "_" for c in expected)
            if expected == prefixed_name:
                return t["name"]
        return None

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server.

        `tool_name` can be either the prefixed name or the original name.
        """
        if self._session is None:
            return "[ERROR] MCP client not connected"

        # Try to resolve prefixed name to original
        original = self._original_tool_name(tool_name)
        if original is None:
            # Maybe it's already the original name
            original = tool_name

        try:
            result = await self._session.call_tool(original, arguments)
            # Extract text from the result
            parts: list[str] = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
                else:
                    parts.append(str(content))
            return "\n".join(parts) if parts else "(no output)"
        except Exception as exc:
            return f"[ERROR] MCP tool call failed: {exc}"

    async def close(self) -> None:
        """Disconnect from the MCP server."""
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None
        if self._streams_cm is not None:
            try:
                await self._streams_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._streams_cm = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_connected(self) -> bool:
        return self._session is not None
