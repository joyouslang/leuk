"""Tests for MCP connector discovery/import (registry mappers + persistence)."""

from __future__ import annotations

import pytest

from leuk.mcp import registry as reg


class TestRegistryMapping:
    def test_npm_package_to_stdio(self):
        entry = {
            "name": "io.github.acme/files",
            "description": "files",
            "packages": [{
                "registry_type": "npm",
                "identifier": "@acme/mcp-files",
                "version": "1.2.3",
                "environment_variables": [{"name": "ACME_TOKEN", "isRequired": True}],
            }],
        }
        res = reg._map_registry_entry(entry)
        assert res.config.transport == "stdio"
        assert res.config.command == "npx"
        assert res.config.args == ["-y", "@acme/mcp-files@1.2.3"]
        # required env with no default → an InputSpec to prompt for, not in env yet.
        assert [(i.id, i.target) for i in res.inputs] == [("ACME_TOKEN", "env")]
        assert res.config.env == {}
        assert res.config.name == "files"

    def test_pypi_package_uses_uvx(self):
        entry = {"name": "x/y", "packages": [{"registry_type": "pypi", "identifier": "mcp-y"}]}
        res = reg._map_registry_entry(entry)
        assert res.config.command == "uvx"
        assert res.config.args == ["mcp-y"]

    def test_runtime_hint_overrides_command(self):
        entry = {"name": "x/y", "packages": [
            {"registry_type": "npm", "identifier": "p", "runtime_hint": "bunx"}]}
        assert reg._map_registry_entry(entry).config.command == "bunx"

    def test_remote_maps_to_sse(self):
        entry = {"name": "x/y", "remotes": [{"transport_type": "sse", "url": "https://h/mcp"}]}
        res = reg._map_registry_entry(entry)
        assert res.config.transport == "sse"
        assert res.config.url == "https://h/mcp"

    def test_no_packages_or_remotes_errors(self):
        with pytest.raises(ValueError):
            reg._map_registry_entry({"name": "x/y"})

    def test_items_tolerates_nested_and_flat(self):
        nested = {"servers": [{"server": {"name": "a"}, "_meta": {}}]}
        flat = {"data": [{"name": "b"}]}
        assert reg._items(nested)[0]["name"] == "a"
        assert reg._items(flat)[0]["name"] == "b"

    def test_sanitize_name(self):
        assert reg._sanitize_name("io.github.acme/mcp-files") == "mcp-files"
        assert reg._sanitize_name("@scope/pkg@1.0") == "pkg"


class TestRegistryCamelCase:
    """The live registry uses camelCase + nested transport/args — must map right."""

    def test_camelcase_npm_package(self):
        entry = {
            "name": "com.pulsemcp/remote-filesystem",
            "description": "remote fs",
            "packages": [{
                "registryType": "npm",
                "identifier": "remote-filesystem-mcp-server",
                "version": "0.1.2",
                "runtimeHint": "npx",
                "transport": {"type": "stdio"},
                "runtimeArguments": [{"value": "-y", "type": "positional"}],
                "environmentVariables": [
                    {"name": "GCS_BUCKET", "isRequired": True},
                    {"name": "GCS_PROJECT_ID"},
                ],
            }],
        }
        res = reg._map_registry_entry(entry)
        assert res.config.command == "npx"
        # `-y` from runtimeArguments must not be duplicated.
        assert res.config.args == ["-y", "remote-filesystem-mcp-server@0.1.2"]
        # GCS_BUCKET is required (no default) → prompt; GCS_PROJECT_ID optional → skipped.
        assert [i.id for i in res.inputs] == ["GCS_BUCKET"]
        assert res.inputs[0].target == "env"

    def test_camelcase_remote(self):
        entry = {"name": "ac/x", "remotes": [{"type": "streamable-http", "url": "https://h/mcp"}]}
        res = reg._map_registry_entry(entry)
        assert res.config.transport == "sse" and res.config.url == "https://h/mcp"

    def test_named_required_arg_becomes_input_and_applies(self):
        """The filesystem-server case: a required named arg (no default) must become
        a prompt-able InputSpec with a `${slug}` placeholder, then fill in."""
        entry = {
            "name": "io.github.bytedance/mcp-server-filesystem",
            "packages": [{
                "registryType": "npm",
                "identifier": "@agent-infra/mcp-server-filesystem",
                "packageArguments": [
                    {"type": "named", "name": "allowed-directories",
                     "description": "Allowed dirs", "isRequired": True, "format": "string"},
                    {"type": "named", "name": "port", "isRequired": True,
                     "format": "number", "default": "8089"},  # has default → no prompt
                ],
            }],
        }
        res = reg._map_registry_entry(entry)
        # `--port 8089` filled from default; `--allowed-directories` left as a placeholder.
        assert res.config.args == [
            "-y", "@agent-infra/mcp-server-filesystem",
            "--allowed-directories", "${allowed-directories}", "--port", "8089",
        ]
        assert [(i.flag, i.target) for i in res.inputs] == [("--allowed-directories", "arg")]

        reg.apply_inputs(res, {"allowed-directories": "/home/me/work"})
        assert "--allowed-directories" in res.config.args
        assert "/home/me/work" in res.config.args
        assert "${allowed-directories}" not in res.config.args


class TestUrlResolve:
    def test_wraps_url(self):
        res = reg._url_resolve("https://host.example/mcp", None)
        assert res.config.transport == "sse"
        assert res.config.url == "https://host.example/mcp"

    def test_rejects_non_url(self):
        with pytest.raises(ValueError):
            reg._url_resolve("not-a-url", None)


class TestPersistence:
    @pytest.fixture(autouse=True)
    def _tmp_config(self, tmp_path, monkeypatch):
        import leuk.config as cfg

        monkeypatch.setattr(cfg, "persistent_config_path", lambda: tmp_path / "config.json")

    def test_add_list_toggle_remove(self):
        from leuk.config import MCPServerConfig

        res = reg.ResolvedConnector(MCPServerConfig(name="files", transport="stdio", command="npx"))
        reg.add_connector(res)
        assert [c.name for c in reg.list_connectors()] == ["files"]
        assert reg.list_connectors()[0].enabled is True

        assert reg.set_connector_enabled("files", False) is True
        assert reg.list_connectors()[0].enabled is False
        assert reg.set_connector_enabled("missing", False) is False

        assert reg.remove_connector("files") is True
        assert reg.list_connectors() == []
        assert reg.remove_connector("files") is False

    def test_add_replaces_same_name(self):
        from leuk.config import MCPServerConfig

        reg.add_connector(reg.ResolvedConnector(MCPServerConfig(name="dup", command="a")))
        reg.add_connector(reg.ResolvedConnector(MCPServerConfig(name="dup", command="b")))
        servers = reg.list_connectors()
        assert len(servers) == 1 and servers[0].command == "b"

    def test_update_connector_edits_command_args_env(self):
        from leuk.config import MCPServerConfig

        reg.add_connector(reg.ResolvedConnector(
            MCPServerConfig(name="fs", command="npx", args=["-y", "pkg"])))
        ok = reg.update_connector(
            "fs", args=["-y", "pkg", "--allowed-directories", "/x"], env={"TOK": "1"}
        )
        assert ok is True
        s = reg.list_connectors()[0]
        assert s.args == ["-y", "pkg", "--allowed-directories", "/x"]
        assert s.env == {"TOK": "1"}
        assert reg.update_connector("missing", args=[]) is False


class TestSearchResolve:
    def test_registry_search_and_resolve(self, monkeypatch):
        payload = {"servers": [
            {"name": "io.x/files", "description": "d",
             "packages": [{"registry_type": "npm", "identifier": "@x/files"}]},
        ]}

        class _Resp:
            def raise_for_status(self): ...
            def json(self): return payload

        class _Client:
            def __init__(self, *a, **k): ...
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def get(self, url, params=None): return _Resp()

        monkeypatch.setattr(reg.httpx, "Client", _Client)
        hits = reg.search("files", "mcp")
        assert hits[0].id == "io.x/files"
        # search precomputes the resolved config so "add" needs no second call.
        assert hits[0].resolved is not None
        assert hits[0].resolved.config.args == ["-y", "@x/files"]
        res = reg.resolve("io.x/files", "mcp")
        assert res.config.command == "npx" and res.config.args == ["-y", "@x/files"]

    def test_clawhub_not_searchable(self):
        with pytest.raises(ValueError, match="mcp"):
            reg.search("x", "clawhub")
