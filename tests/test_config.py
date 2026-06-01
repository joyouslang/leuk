"""Tests for configuration loading."""

import json
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
    load_persistent_config,
    load_settings,
    migrate_legacy_config_env,
    persistent_config_path,
    save_credentials,
    save_persistent_config,
)


def test_default_settings():
    s = Settings()
    assert s.llm.provider == "zen"
    assert s.llm.temperature == 0.0
    assert s.llm.max_tokens == 16384
    assert s.agent.max_tool_rounds == 50
    # None = derive the compaction budget from the model's queried context window.
    assert s.agent.max_context_tokens is None
    assert s.agent.max_tool_result_tokens == 8_000


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


def test_migrate_legacy_config_env_into_json(tmp_path: Path):
    """A legacy config.env is folded into config.json (nested) and retired."""
    env_file = tmp_path / "config.env"
    env_file.write_text(
        "LEUK_LLM_PROVIDER=openai\nLEUK_LLM_MODEL=gpt-4o\nLEUK_MAX_TOOL_ROUNDS=77\n"
    )
    cf = tmp_path / "config.json"
    with (
        patch("leuk.config.config_env_path", return_value=env_file),
        patch("leuk.config.persistent_config_path", return_value=cf),
    ):
        migrate_legacy_config_env()
        cfg = json.loads(cf.read_text())
        assert cfg["llm"]["provider"] == "openai"
        assert cfg["llm"]["model"] == "gpt-4o"
        assert cfg["agent"]["max_tool_rounds"] == 77  # JSON-typed, not "77"
        assert not env_file.exists()  # renamed to config.env.migrated
        assert (tmp_path / "config.env.migrated").exists()
        # And load_settings then applies them.
        s = load_settings()
        assert s.llm.provider == "openai"
        assert s.llm.model == "gpt-4o"
        assert s.agent.max_tool_rounds == 77


def test_load_settings():
    s = load_settings()
    assert isinstance(s, Settings)


# ── Persistent config ─────────────────────────────────────────────


def test_persistent_config_path():
    p = persistent_config_path()
    assert p == Path.home() / ".config" / "leuk" / "config.json"


def test_save_and_load_persistent_config(tmp_path: Path):
    cf = tmp_path / "config.json"
    with patch("leuk.config.persistent_config_path", return_value=cf):
        save_persistent_config(
            {"last_provider": "anthropic", "last_model": "claude-sonnet-4-20250514"}
        )
        assert cf.exists()
        c = load_persistent_config()
        assert c["last_provider"] == "anthropic"
        assert c["last_model"] == "claude-sonnet-4-20250514"


def test_save_persistent_config_merges(tmp_path: Path):
    """save_persistent_config should merge with existing config, not clobber it."""
    cf = tmp_path / "config.json"
    with patch("leuk.config.persistent_config_path", return_value=cf):
        save_persistent_config({"last_provider": "openai", "last_model": "gpt-4o"})
        save_persistent_config({"last_model": "gpt-4o-mini"})
        c = load_persistent_config()
        assert c["last_provider"] == "openai"  # preserved
        assert c["last_model"] == "gpt-4o-mini"  # updated


def test_load_persistent_config_missing_file(tmp_path: Path):
    cf = tmp_path / "nonexistent.json"
    with patch("leuk.config.persistent_config_path", return_value=cf):
        assert load_persistent_config() == {}


def test_load_persistent_config_corrupt_file(tmp_path: Path):
    cf = tmp_path / "config.json"
    cf.write_text("not json!")
    with patch("leuk.config.persistent_config_path", return_value=cf):
        assert load_persistent_config() == {}


def test_load_settings_applies_last_used(tmp_path: Path):
    """When no env/config override, config.json should supply provider/model."""
    cf = tmp_path / "config.json"
    cf.write_text(
        json.dumps({"last_provider": "anthropic", "last_model": "claude-sonnet-4-20250514"})
    )
    with patch("leuk.config.persistent_config_path", return_value=cf):
        s = load_settings()
        assert s.llm.provider == "anthropic"
        assert s.llm.model == "claude-sonnet-4-20250514"


def test_load_settings_env_overrides_persistent_config(tmp_path: Path, monkeypatch):
    """Explicit env vars should take precedence over config.json."""
    cf = tmp_path / "config.json"
    cf.write_text(
        json.dumps({"last_provider": "anthropic", "last_model": "claude-sonnet-4-20250514"})
    )
    monkeypatch.setenv("LEUK_LLM_PROVIDER", "google")
    monkeypatch.setenv("LEUK_LLM_MODEL", "gemini-pro")
    with patch("leuk.config.persistent_config_path", return_value=cf):
        s = load_settings()
        assert s.llm.provider == "google"
        assert s.llm.model == "gemini-pro"


def test_nested_config_json_overlay(tmp_path: Path):
    """A nested config.json section configures a sub-model (what config.env did)."""
    cf = tmp_path / "config.json"
    cf.write_text(json.dumps({"llm": {"temperature": 0.4, "max_tokens": 9000}}))
    with patch("leuk.config.persistent_config_path", return_value=cf):
        s = load_settings()
        assert s.llm.temperature == 0.4  # coerced from JSON number
        assert s.llm.max_tokens == 9000


def test_env_overrides_nested_config_json(tmp_path: Path, monkeypatch):
    """Env vars win over a nested config.json section."""
    cf = tmp_path / "config.json"
    cf.write_text(json.dumps({"llm": {"temperature": 0.4}}))
    monkeypatch.setenv("LEUK_LLM_TEMPERATURE", "0.9")
    with patch("leuk.config.persistent_config_path", return_value=cf):
        s = load_settings()
        assert s.llm.temperature == 0.9


def test_migration_does_not_clobber_existing_config_json(tmp_path: Path):
    """On migration, existing config.json values win over migrated ones."""
    cf = tmp_path / "config.json"
    cf.write_text(json.dumps({"llm": {"provider": "anthropic"}}))
    env_file = tmp_path / "config.env"
    env_file.write_text("LEUK_LLM_PROVIDER=openai\nLEUK_LLM_MODEL=gpt-4o\n")
    with (
        patch("leuk.config.persistent_config_path", return_value=cf),
        patch("leuk.config.config_env_path", return_value=env_file),
    ):
        migrate_legacy_config_env()
        cfg = json.loads(cf.read_text())
        assert cfg["llm"]["provider"] == "anthropic"  # existing wins
        assert cfg["llm"]["model"] == "gpt-4o"  # migrated new key added


# ── Voice/TTS/STT config keys ────────────────────────────────────


def test_persistent_config_voice_keys(tmp_path: Path):
    """config.json can store voice/TTS/STT settings."""
    cf = tmp_path / "config.json"
    with patch("leuk.config.persistent_config_path", return_value=cf):
        save_persistent_config(
            {
                "stt_model_size": "small",
                "stt_language": "en",
                "tts_speaker": "ru_karina",
                "tts_language": "ru",
                "vad_sensitivity": "0.7",
                "audio_input_device": 13,
            }
        )
        c = load_persistent_config()
        assert c["stt_model_size"] == "small"
        assert c["stt_language"] == "en"
        assert c["tts_speaker"] == "ru_karina"
        assert c["tts_language"] == "ru"
        assert c["vad_sensitivity"] == "0.7"
        assert c["audio_input_device"] == 13


def test_persistent_config_null_values(tmp_path: Path):
    """config.json handles None values for optional fields."""
    cf = tmp_path / "config.json"
    with patch("leuk.config.persistent_config_path", return_value=cf):
        save_persistent_config({"stt_language": None, "tts_speaker": None})
        c = load_persistent_config()
        assert c["stt_language"] is None
        assert c["tts_speaker"] is None
