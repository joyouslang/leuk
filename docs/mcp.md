[Home](README.md) › MCP

# MCP (Model Context Protocol)

leuk can both **consume** external MCP tool servers and **expose** itself as one.

## Connecting to servers — `src/leuk/mcp/`

Configure servers in `Settings.mcp_servers`. On startup the REPL connects to each
(`MCPClient`), discovers its tools via `list_tools()`, and registers them in the
tool registry with prefixed names `mcp_{server}_{tool}` (`MCPToolBridge`). Calls
proxy through `MCPClient.call_tool()`.

Transports (`mcp/client.py`):

- **stdio** — launch the server as a subprocess.
- **SSE** — connect to an HTTP Server-Sent Events endpoint.

These bridged tools behave like any other [tool](tools.md) and pass through the
[SafetyGuard](safety.md).

## Exposing leuk as a server — `src/leuk/mcp/server.py`

Configured via `Settings.mcp_server` (`MCPExposureConfig`), leuk can serve its own
capabilities (e.g. `list_sessions`) to other MCP clients.

## See also

- [Tools](tools.md) · [Configuration](configuration.md)
