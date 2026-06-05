"""Tests for the skills/mcp management CLI (shared logic behind the TUI)."""

from __future__ import annotations

from pathlib import Path

import pytest

from leuk.cli.extensions_manager import run_mcp_cli, run_skills_cli


@pytest.fixture(autouse=True)
def _tmp_config(tmp_path, monkeypatch):
    """Point all leuk config (config.json, config.env, skills dir) at a tmp dir."""
    cfgdir = tmp_path / "cfg"
    cfgdir.mkdir()
    skills = tmp_path / "skills"
    monkeypatch.setattr("leuk.config.config_dir", lambda: cfgdir)
    import json

    (cfgdir / "config.json").write_text(json.dumps({"skills": {"directory": str(skills)}}))
    return tmp_path


def _local_skill(root: Path, slug: str = "demo") -> Path:
    d = root / "src" / slug
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: Demo\ndescription: A demo skill\n---\nstep one\n")
    (d / "run.sh").write_text("echo hi")
    return d


class TestSkillsCli:
    def test_add_list_lifecycle(self, _tmp_config, capsys):
        src = _local_skill(_tmp_config)
        assert run_skills_cli(["add", str(src), "--source", "local", "--trust"]) == 0
        assert "installed and trusted demo" in capsys.readouterr().out

        assert run_skills_cli(["list"]) == 0
        out = capsys.readouterr().out
        assert "demo" in out and "trusted, on" in out

        assert run_skills_cli(["disable", "demo"]) == 0
        run_skills_cli(["list"])
        assert "trusted, off" in capsys.readouterr().out

        assert run_skills_cli(["untrust", "demo"]) == 0
        run_skills_cli(["list"])
        assert "UNTRUSTED" in capsys.readouterr().out

        assert run_skills_cli(["remove", "demo"]) == 0
        run_skills_cli(["list"])
        assert "No skills installed" in capsys.readouterr().out

    def test_add_untrusted_by_default(self, _tmp_config, capsys):
        src = _local_skill(_tmp_config)
        run_skills_cli(["add", str(src), "--source", "local"])
        assert "untrusted" in capsys.readouterr().out

    def test_action_on_missing_skill(self, capsys):
        assert run_skills_cli(["enable", "nope"]) == 1
        assert "no such skill" in capsys.readouterr().out


class TestMcpCli:
    def test_add_url_lifecycle(self, capsys):
        assert run_mcp_cli(["add", "https://host.example/mcp", "--source", "url", "--name", "demo"]) == 0
        assert "added demo" in capsys.readouterr().out

        assert run_mcp_cli(["list"]) == 0
        out = capsys.readouterr().out
        assert "demo" in out and "on, sse" in out and "https://host.example/mcp" in out

        assert run_mcp_cli(["disable", "demo"]) == 0
        run_mcp_cli(["list"])
        assert "off, sse" in capsys.readouterr().out

        assert run_mcp_cli(["remove", "demo"]) == 0
        assert run_mcp_cli(["remove", "demo"]) == 1  # already gone

    def test_add_invalid_url_errors(self, capsys):
        assert run_mcp_cli(["add", "not-a-url", "--source", "url"]) == 1
        assert "error:" in capsys.readouterr().out

    def test_search_mocked(self, capsys, monkeypatch):
        from leuk.mcp import registry

        def _fake_search(query, source, *, registry_url):
            return [registry.ConnectorHit("io.x/files", "files", "file tools", "mcp")]

        monkeypatch.setattr(registry, "search", _fake_search)
        assert run_mcp_cli(["search", "files"]) == 0
        assert "io.x/files" in capsys.readouterr().out
