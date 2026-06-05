"""Discover and import MCP connectors/plugins from registries.

A "connector" is just an MCP server, which leuk already runs and bridges. So
importing = resolve a registry entry into an :class:`~leuk.config.MCPServerConfig`
and persist it to ``mcp_servers`` in config.json; the existing startup loop +
``MCPClient``/``MCPToolBridge`` do the rest.

Sources:
  * ``mcp`` — the official MCP registry REST API (``/v0/servers``).
  * ``url`` — wrap any remote MCP server URL (e.g. one copied from a directory).

(ClawHub hosts *skills*, not MCP servers, so it is not a connector source — see
``leuk.skills`` for ClawHub skill import.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

from leuk.config import MCPServerConfig, load_persistent_config, save_persistent_config

_DEFAULT_REGISTRY = "https://registry.modelcontextprotocol.io"
_TIMEOUT = 12.0
_LIMIT = "10"
_RETRIES = 3  # the registry's TLS endpoint intermittently drops connections
_HEADERS = {
    "User-Agent": "leuk (+https://github.com/joyouslang/leuk)",
    "Accept": "application/json",
}

# registryType → the launcher command used for a stdio server.
_RUNTIME_FOR_REGISTRY = {"npm": "npx", "pypi": "uvx", "oci": "docker"}
# Node-style runners that take `-y` to auto-install.
_NPX_LIKE = {"npx", "bunx", "pnpx", "dlx"}


def _g(d: dict, *keys: str, default: object = None) -> object:
    """Tolerant getter: the registry uses camelCase, older payloads snake_case."""
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


@dataclass(slots=True)
class InputSpec:
    """A value the user must supply for a connector (a required arg or env var).

    Registry entries document each server's required arguments/env vars (with
    descriptions, formats, defaults), so we capture them here and prompt once at
    import — the answer to "every server needs different flags" with minimal effort.
    """

    id: str  # placeholder slug (for args) or env-var name (for env)
    target: str  # "arg" | "env"
    description: str = ""
    default: str = ""
    secret: bool = False
    flag: str = ""  # display: the "--flag" (named arg) or the env-var name
    fmt: str = ""  # format hint: string/number/filepath/boolean


@dataclass(slots=True)
class ConnectorHit:
    """One search result from a registry (carrying its resolved config)."""

    id: str  # registry identifier used to resolve (server name / slug)
    name: str
    description: str
    source: str
    resolved: "ResolvedConnector | None" = None  # precomputed → no 2nd network call


@dataclass(slots=True)
class ResolvedConnector:
    """A registry entry mapped to a leuk MCP server config, ready to persist.

    ``inputs`` lists required values with no default — fill them with
    :func:`apply_inputs` before saving.
    """

    config: MCPServerConfig
    inputs: list[InputSpec] = field(default_factory=list)
    summary: str = ""


def _sanitize_name(raw: str) -> str:
    """A short, filesystem/tool-safe connector name from a registry id."""
    base = raw.rsplit("/", 1)[-1].split("@", 1)[0]
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in base).strip("_-")
    return cleaned or "connector"


# ── official MCP registry (REST) ───────────────────────────────────


def _items(payload: dict) -> list[dict]:
    """Pull the server list out of a registry response, tolerating shape drift."""
    raw = payload.get("servers") or payload.get("data") or []
    out: list[dict] = []
    for it in raw:
        # Some API versions nest the server under {"server": {...}, "_meta": ...}.
        out.append(it.get("server", it) if isinstance(it, dict) else {})
    return out


def _server_id(entry: dict) -> str:
    return str(_g(entry, "name", "id", default="") or "")


def _slug(s: str) -> str:
    """A placeholder-token slug from an argument name/hint."""
    cleaned = "".join(c if c.isalnum() else "-" for c in s.lstrip("-")).strip("-").lower()
    return cleaned or "value"


def _flag(name: str) -> str:
    """Render a named-argument flag, adding `--` when the registry omits it."""
    return name if name.startswith("-") else f"--{name}"


def _consume_args(arg_list: object, inputs: list[InputSpec]) -> list[str]:
    """Build CLI tokens from registry runtime/package args, recording required
    values (no default) as ``${slug}`` placeholders + :class:`InputSpec` entries."""
    tokens: list[str] = []
    if not isinstance(arg_list, list):
        return tokens
    for a in arg_list:
        if not isinstance(a, dict):
            continue
        value = a.get("value") or a.get("default")
        desc = str(a.get("description", ""))
        fmt = str(a.get("format", ""))
        required = bool(a.get("isRequired"))
        if a.get("type") == "named":
            flag = _flag(str(a.get("name", "")))
            if flag == "--":
                continue
            if value:
                tokens += [flag, str(value)]
            elif required:
                sid = _slug(flag)
                tokens += [flag, f"${{{sid}}}"]
                inputs.append(InputSpec(sid, "arg", desc or flag, "", False, flag, fmt))
        else:  # positional
            if value:
                tokens.append(str(value))
            elif required:
                sid = _slug(str(a.get("valueHint") or desc or "arg"))
                tokens.append(f"${{{sid}}}")
                inputs.append(InputSpec(sid, "arg", desc, "", False, sid, fmt))
    return tokens


def _consume_env(env_list: object, inputs: list[InputSpec]) -> dict[str, str]:
    """Build the env dict (with defaults), recording required-without-default vars."""
    env: dict[str, str] = {}
    if not isinstance(env_list, list):
        return env
    for e in env_list:
        if not isinstance(e, dict) or not e.get("name"):
            continue
        name = str(e["name"])
        value = e.get("value") or e.get("default")
        if value is not None and value != "":
            env[name] = str(value)
        elif e.get("isRequired"):
            inputs.append(InputSpec(
                name, "env", str(e.get("description", "")), "",
                bool(e.get("isSecret")), name, str(e.get("format", "")),
            ))
    return env


def _map_registry_entry(entry: dict, *, name: str | None = None) -> ResolvedConnector:
    """Map one MCP-registry server object → ResolvedConnector (camelCase schema)."""
    server_id = _server_id(entry)
    conn_name = _sanitize_name(name or server_id)

    remotes = entry.get("remotes") or []
    if remotes:
        url = str(_g(remotes[0], "url", default="") or "")
        cfg = MCPServerConfig(name=conn_name, transport="sse", url=url)
        return ResolvedConnector(cfg, [], f"remote MCP server {server_id} ({url})")

    packages = entry.get("packages") or []
    if not packages:
        raise ValueError(f"registry entry {server_id!r} has no packages or remotes")
    pkg = packages[0]
    reg_type = str(_g(pkg, "registryType", "registry_type", default="") or "").lower()
    command = str(
        _g(pkg, "runtimeHint", "runtime_hint", default="")
        or _RUNTIME_FOR_REGISTRY.get(reg_type, "npx")
    )
    identifier = str(_g(pkg, "identifier", default="") or "")
    version = str(_g(pkg, "version", default="") or "")
    spec = f"{identifier}@{version}" if version and reg_type == "npm" else identifier

    inputs: list[InputSpec] = []
    runtime_tokens = _consume_args(_g(pkg, "runtimeArguments", "runtime_arguments"), inputs)
    package_tokens = _consume_args(_g(pkg, "packageArguments", "package_arguments"), inputs)
    if command in _NPX_LIKE and "-y" not in runtime_tokens:
        runtime_tokens = ["-y", *runtime_tokens]
    args = [*runtime_tokens, spec, *package_tokens] if spec else [*runtime_tokens, *package_tokens]
    env = _consume_env(_g(pkg, "environmentVariables", "environment_variables"), inputs)

    cfg = MCPServerConfig(name=conn_name, transport="stdio", command=command, args=args, env=env)
    return ResolvedConnector(cfg, inputs, f"{reg_type or 'stdio'} package {identifier}")


def _is_placeholder(token: str) -> bool:
    return token.startswith("${") and token.endswith("}")


def apply_inputs(resolved: ResolvedConnector, values: dict[str, str]) -> None:
    """Fill user-provided *values* into a resolved connector's args/env in place."""
    cfg = resolved.config
    cfg.args = [
        values.get(a[2:-1], a) if _is_placeholder(a) else a for a in cfg.args
    ]
    for spec in resolved.inputs:
        val = values.get(spec.id, "")
        if spec.target == "env" and val:
            cfg.env[spec.id] = val


def _registry_get(query: str, registry_url: str) -> list[dict]:
    """One fast GET against /v0/servers (search or list).

    Synchronous on purpose: the callers (the `/mcp` manager in a worker thread,
    and the `leuk mcp` CLI) are sync contexts. An async client run via
    ``asyncio.run`` inside ``asyncio.to_thread`` raised ConnectError.
    """
    url = f"{registry_url.rstrip('/')}/v0/servers"
    params = {"search": query, "limit": _LIMIT} if query else {"limit": _LIMIT}
    last_exc: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            with httpx.Client(
                timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True
            ) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                return _items(resp.json())
        except httpx.TransportError as exc:  # connect/read/protocol/timeout — retry
            last_exc = exc
            if attempt < _RETRIES - 1:
                time.sleep(0.5 * (attempt + 1))
    raise ValueError(
        f"could not reach the MCP registry after {_RETRIES} tries: {last_exc}"
    )


# ── add-by-URL ─────────────────────────────────────────────────────


def _url_resolve(url: str, name: str | None) -> ResolvedConnector:
    if "://" not in url:
        raise ValueError(f"{url!r} is not a valid URL")
    conn_name = _sanitize_name(name or url.rstrip("/").rsplit("/", 1)[-1] or "remote")
    cfg = MCPServerConfig(name=conn_name, transport="sse", url=url)
    return ResolvedConnector(cfg, [], f"remote MCP server {url}")


# ── public API ─────────────────────────────────────────────────────


def search(
    query: str, source: str = "mcp", *, registry_url: str = _DEFAULT_REGISTRY
) -> list[ConnectorHit]:
    """Search the MCP registry. Each hit carries its resolved config (no 2nd call)."""
    if source != "mcp":
        raise ValueError("only the 'mcp' source is searchable; use add-by-URL otherwise")
    hits: list[ConnectorHit] = []
    for entry in _registry_get(query, registry_url):
        sid = _server_id(entry)
        if not sid:
            continue
        try:
            resolved: ResolvedConnector | None = _map_registry_entry(entry)
        except ValueError:
            resolved = None
        title = str(_g(entry, "title", default="") or sid)
        hits.append(ConnectorHit(sid, title, str(entry.get("description", "")), "mcp", resolved))
    return hits


def resolve(
    identifier: str,
    source: str = "mcp",
    *,
    name: str | None = None,
    registry_url: str = _DEFAULT_REGISTRY,
) -> ResolvedConnector:
    """Resolve a registry id / URL into a persistable MCP server config."""
    if source == "url":
        return _url_resolve(identifier, name)
    if source != "mcp":
        raise ValueError(f"unknown connector source {source!r}")
    entries = _registry_get(identifier, registry_url)
    match = next((e for e in entries if _server_id(e) == identifier), None) or (
        entries[0] if entries else None
    )
    if match is None:
        raise ValueError(f"no MCP-registry server matching {identifier!r}")
    return _map_registry_entry(match, name=name)


# ── persistence (config.json ``mcp_servers``) ──────────────────────


def list_connectors() -> list[MCPServerConfig]:
    """The MCP servers currently saved in config.json."""
    raw = load_persistent_config().get("mcp_servers", [])
    return [MCPServerConfig.model_validate(s) for s in raw if isinstance(s, dict)]


def _save_connectors(servers: list[MCPServerConfig]) -> None:
    save_persistent_config({"mcp_servers": [s.model_dump() for s in servers]})


def add_connector(resolved: ResolvedConnector) -> MCPServerConfig:
    """Persist a resolved connector (replacing any with the same name). Returns it."""
    servers = [s for s in list_connectors() if s.name != resolved.config.name]
    servers.append(resolved.config)
    _save_connectors(servers)
    return resolved.config


def set_connector_enabled(name: str, enabled: bool) -> bool:
    """Toggle a saved connector on/off. Returns True if one was found."""
    servers = list_connectors()
    found = False
    for s in servers:
        if s.name == name:
            s.enabled = enabled
            found = True
    if found:
        _save_connectors(servers)
    return found


def remove_connector(name: str) -> bool:
    """Delete a saved connector. Returns True if one was removed."""
    servers = list_connectors()
    kept = [s for s in servers if s.name != name]
    if len(kept) == len(servers):
        return False
    _save_connectors(kept)
    return True
