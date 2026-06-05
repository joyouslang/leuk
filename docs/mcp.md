[Home](README.md) › MCP

# MCP (Model Context Protocol)

leuk can both **consume** external MCP tool servers and **expose** itself as one.

## Connecting to servers — `src/leuk/mcp/`

Configure servers in `Settings.mcp_servers`. On startup the REPL connects to each
(`MCPClient`), discovers its tools via `list_tools()`, and registers them in the
tool registry with prefixed names `mcp_{server}_{tool}` (`MCPToolBridge`). Calls
proxy through `MCPClient.call_tool()`.

Connection happens **in the background, concurrently** — startup never blocks on a
slow or hung server (each connect is also bounded by a timeout). Tools appear as
soon as their server is ready (the agent reads tool specs each turn). `/mcp` shows
each server's live status (`connecting` / `connected` / `failed: …`). A per-server
`enabled` flag keeps an entry without connecting; stdio servers may also carry an
`env` map (merged onto the process environment for the subprocess).

Transports (`mcp/client.py`):

- **stdio** — launch the server as a subprocess.
- **SSE** — connect to an HTTP Server-Sent Events endpoint.

These bridged tools behave like any other [tool](tools.md) and pass through the
[SafetyGuard](safety.md).

## Importing connectors/plugins — `/mcp` · `src/leuk/mcp/registry.py`

Rather than hand-editing `mcp_servers`, you can **discover and import** MCP
connectors. A connector *is* an MCP server, so importing just resolves a registry
entry into an `MCPServerConfig`, saves it to `config.json`, and the loop above
connects it on the next start.

Open the manager with **`/mcp`** (a menu UI in your theme), or use the CLI:

```bash
leuk mcp search <query>                  # search the official MCP registry
leuk mcp add <id|url> [--source mcp|url] [--name N]
leuk mcp list                            # saved connectors + on/off state
leuk mcp enable <name> | disable <name> | remove <name>
```

Sources (`registry.py`):

- **`mcp`** — the official MCP registry REST API (`/v0/servers`); npm/pypi
  packages map to a stdio `npx`/`uvx` launch, remotes to an SSE URL. Search
  results carry their resolved config, so adding one needs no second network call.
- **`url`** — wrap any remote MCP server URL (e.g. one copied from a connector
  directory) into an SSE connector.

### Per-server arguments & secrets (filled at import)

Different servers need different flags/secrets (e.g. the filesystem server needs
`--allowed-directories`). The registry **documents** each server's required
arguments and environment variables (name, description, format, default), so leuk
captures them as `InputSpec`s and, at import, **prompts once for only the required
fields that have no default** — guided by their descriptions. Defaults are filled
automatically. Values are applied with `registry.apply_inputs()` into the saved
connector's `args`/`env`. This is the minimum-effort path: you fill one or two
fields and the connector is ready.

(ClawHub hosts *skills*, not MCP servers, so it is **not** a connector source —
see [Skills](skills.md).)

Each saved connector has an `enabled` flag — toggle it off to keep the entry but
skip connecting. Imported tools are still gated by the [SafetyGuard](safety.md)
when the model calls them; review third-party connectors like any dependency.

## Exposing leuk as a server — `src/leuk/mcp/server.py`

Configured via `Settings.mcp_server` (`MCPExposureConfig`), leuk can serve its own
capabilities (e.g. `list_sessions`) to other MCP clients.

## See also

- [Tools](tools.md) · [Configuration](configuration.md)
