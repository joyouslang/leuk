[Home](README.md) › Tools

# Tools

Tools implement the `Tool` protocol (`src/leuk/tools/base.py`): a `spec` property
returning a `ToolSpec` (name, description, JSON-schema parameters) and
`async execute(arguments) -> str`. They're registered in
`create_default_registry()` (`src/leuk/tools/__init__.py`) and gated by config.

## Built-in tools

| Tool | Module | Default | Notes |
|------|--------|---------|-------|
| `shell` | `shell.py` | on | Run commands (timeout, workdir; optional Docker sandbox) |
| `file_read` | `file_read.py` | on | Read with line numbers + offset/limit |
| `file_edit` | `file_edit.py` | on | Create new files; change existing ones with **targeted patches** (exact-string replace). Returns a summary + **unified diff** (shown with green/red highlighting). Never rewrites a whole existing file — `overwrite=true` (approval-gated) is required for that |
| `web_search` | `web_search.py` | on | Search the web (keyless DuckDuckGo) → ranked results; then `web_fetch` a result URL |
| `web_fetch` | `web_fetch.py` | on | Fetch a **URL**, extract text (optional CSS selector); rejects non-URL queries |
| `sub_agent` | `sub_agent.py` | on | Spawn a child agent ([Architecture](architecture.md)) |
| `memory_write` | `memory_write.py` | on | Write hierarchical memory |
| `browser` | `browser.py` | **opt-in** | [SPA/AJAX browser automation](tools/browser.md) |
| `monitoring` | `monitoring.py` | **opt-in** | Read-only host data: screenshot, screen geometry, system info |
| `input_control` | `input_control.py` | **opt-in** | [Desktop keyboard/mouse](tools/input_control.md) |
| `local_llm` | `local_llm.py` | **opt-in** | Delegate subtasks to a local Ollama model |
| `skill` | `skills/tool.py` | **opt-in** | List/read installed [agent skills](skills.md) (SKILL.md, instructions-only) |
| `mcp_*` | via `mcp/bridge.py` | per server | [External MCP tools](mcp.md) (import via [`/mcp`](mcp.md)) |

Optional tools are enabled via [config toggles](configuration.md) (e.g.
`/settings → General`, or `LEUK_*_ENABLED`).

## Result conventions

- Plain string on success; `"[ERROR] …"` on failure.
- Images use the inline tag `[screenshot:image/png;base64,…]` (also `[image:…]`,
  `[audio:…]`), which providers turn into native media blocks the model sees —
  see [Multimodal](multimodal.md).

## Adding a tool

See [Development → Adding tools](development.md#adding-a-tool). New high-risk tools
should also add a [SafetyGuard](safety.md) rule.

## See also

- [Browser](tools/browser.md) · [Input Control](tools/input_control.md) · [Safety](safety.md) · [MCP](mcp.md)
