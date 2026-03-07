"""Tests for configuration loading."""

import json
import os
from pathlib import Path
from unittest.mock import patch

from leuk.config import (
    AgentConfig,
    LLMConfig,
    SQLiteConfig,
    Settings,
    config_dir,
    config_env_path,
    credentials_path,
    load_credentials,
    load_settings,
    load_state,
    save_credentials,
    save_state,
    state_path,
)


def test_default_settings():
    s = Settings()
    assert s.llm.provider == "zen"
    assert s.llm.temperature == 0.0
    assert s.llm.max_tokens == 16384
    assert s.agent.max_tool_rounds == 50
    assert s.agent.max_context_tokens == 100_000
    assert s.agent.context_strategy == "sliding_window"


def test_llm_config_defaults():
    cfg = LLMConfig()
    assert cfg.provider == "zen"
    assert cfg.model == "big-pickle"
    assert cfg.local_base_url == "http://localhost:11434/v1"
    assert cfg.anthropic_auth_token == ""
    assert cfg.zen_api_key == ""


def test_agent_config_defaults():
    cfg = AgentConfig()
    assert cfg.max_tool_result_tokens == 8_000
    assert "leuk" in cfg.system_prompt.lower()


def test_sqlite_default_path():
    cfg = SQLiteConfig()
    assert cfg.path == "~/.config/leuk/leuk.db"


def test_config_dir():
    d = config_dir()
    assert d == Path.home() / ".config" / "leuk"
    assert d.exists()


def test_credentials_path():
    p = credentials_path()
    assert p == Path.home() / ".config" / "leuk" / "credentials.json"


def test_save_and_load_credentials(tmp_path: Path):
    creds_file = tmp_path / "credentials.json"
    with patch("leuk.config.credentials_path", return_value=creds_file):
        save_credentials({"anthropic_api_key": "sk-ant-test123"})
        assert creds_file.exists()
        # Check file permissions (0600)
        assert oct(creds_file.stat().st_mode & 0o777) == "0o600"

        loaded = load_credentials()
        assert loaded["anthropic_api_key"] == "sk-ant-test123"


def test_load_credentials_missing_file(tmp_path: Path):
    creds_file = tmp_path / "nonexistent.json"
    with patch("leuk.config.credentials_path", return_value=creds_file):
        assert load_credentials() == {}


def test_load_credentials_corrupt_file(tmp_path: Path):
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text("not json!")
    with patch("leuk.config.credentials_path", return_value=creds_file):
        assert load_credentials() == {}


def test_load_settings_overlays_credentials(tmp_path: Path):
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text(json.dumps({"openai_api_key": "sk-test-overlay"}))
    with patch("leuk.config.credentials_path", return_value=creds_file):
        s = load_settings()
        assert s.llm.openai_api_key == "sk-test-overlay"


def test_load_settings_env_takes_precedence(tmp_path: Path, monkeypatch):
    """Environment variable should take precedence over stored credentials."""
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text(json.dumps({"openai_api_key": "sk-from-creds"}))
    monkeypatch.setenv("LEUK_LLM_OPENAI_API_KEY", "sk-from-env")
    with patch("leuk.config.credentials_path", return_value=creds_file):
        s = load_settings()
        assert s.llm.openai_api_key == "sk-from-env"


def test_config_env_path():
    p = config_env_path()
    assert p == Path.home() / ".config" / "leuk" / "config.env"


def test_load_settings_reads_config_env(tmp_path: Path):
    """load_settings should read values from config.env."""
    env_file = tmp_path / "config.env"
    env_file.write_text("LEUK_LLM_PROVIDER=openai\nLEUK_LLM_MODEL=gpt-4o\n")
    with patch("leuk.config.config_env_path", return_value=env_file):
        s = load_settings()
        assert s.llm.provider == "openai"
        assert s.llm.model == "gpt-4o"


def test_load_settings():
    s = load_settings()
    assert isinstance(s, Settings)


# ── State persistence ─────────────────────────────────────────────


def test_state_path():
    p = state_path()
    assert p == Path.home() / ".config" / "leuk" / "state.json"


def test_save_and_load_state(tmp_path: Path):
    sf = tmp_path / "state.json"
    with patch("leuk.config.state_path", return_value=sf):
        save_state({"last_provider": "anthropic", "last_model": "claude-sonnet-4-20250514"})
        assert sf.exists()
        s = load_state()
        assert s["last_provider"] == "anthropic"
        assert s["last_model"] == "claude-sonnet-4-20250514"


def test_save_state_merges(tmp_path: Path):
    """save_state should merge with existing state, not clobber it."""
    sf = tmp_path / "state.json"
    with patch("leuk.config.state_path", return_value=sf):
        save_state({"last_provider": "openai", "last_model": "gpt-4o"})
        save_state({"last_model": "gpt-4o-mini"})
        s = load_state()
        assert s["last_provider"] == "openai"  # preserved
        assert s["last_model"] == "gpt-4o-mini"  # updated


def test_load_state_missing_file(tmp_path: Path):
    sf = tmp_path / "nonexistent.json"
    with patch("leuk.config.state_path", return_value=sf):
        assert load_state() == {}


def test_load_state_corrupt_file(tmp_path: Path):
    sf = tmp_path / "state.json"
    sf.write_text("not json!")
    with patch("leuk.config.state_path", return_value=sf):
        assert load_state() == {}


def test_load_settings_applies_last_used(tmp_path: Path):
    """When no env/config override, state.json should supply provider/model."""
    sf = tmp_path / "state.json"
    sf.write_text(
        json.dumps({"last_provider": "anthropic", "last_model": "claude-sonnet-4-20250514"})
    )
    with patch("leuk.config.state_path", return_value=sf):
        s = load_settings()
        assert s.llm.provider == "anthropic"
        assert s.llm.model == "claude-sonnet-4-20250514"


def test_load_settings_env_overrides_state(tmp_path: Path, monkeypatch):
    """Explicit env vars should take precedence over state.json."""
    sf = tmp_path / "state.json"
    sf.write_text(
        json.dumps({"last_provider": "anthropic", "last_model": "claude-sonnet-4-20250514"})
    )
    monkeypatch.setenv("LEUK_LLM_PROVIDER", "google")
    monkeypatch.setenv("LEUK_LLM_MODEL", "gemini-pro")
    with patch("leuk.config.state_path", return_value=sf):
        s = load_settings()
        assert s.llm.provider == "google"
        assert s.llm.model == "gemini-pro"


def test_load_settings_config_env_overrides_state(tmp_path: Path):
    """config.env settings should take precedence over state.json."""
    sf = tmp_path / "state.json"
    sf.write_text(
        json.dumps({"last_provider": "anthropic", "last_model": "claude-sonnet-4-20250514"})
    )
    env_file = tmp_path / "config.env"
    env_file.write_text("LEUK_LLM_PROVIDER=openai\nLEUK_LLM_MODEL=gpt-4o\n")
    with (
        patch("leuk.config.state_path", return_value=sf),
        patch("leuk.config.config_env_path", return_value=env_file),
    ):
        s = load_settings()
        assert s.llm.provider == "openai"
        assert s.llm.model == "gpt-4o"
