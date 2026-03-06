"""Application configuration via environment variables and ~/.config/leuk/."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ------------------------------------------------------------------
# Config directory helpers
# ------------------------------------------------------------------


def config_dir() -> Path:
    """Return the leuk configuration directory (~/.config/leuk), creating it if needed."""
    p = Path.home() / ".config" / "leuk"
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_env_path() -> Path:
    """Return the path to the config.env file."""
    return config_dir() / "config.env"


def credentials_path() -> Path:
    """Return the path to the credentials file."""
    return config_dir() / "credentials.json"


def load_credentials() -> dict[str, str]:
    """Load saved credentials from disk.

    Returns a dict like:
        {"anthropic_api_key": "sk-...", "anthropic_auth_token": "...", ...}
    """
    path = credentials_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError, OSError:
            return {}
    return {}


def save_credentials(creds: dict[str, str]) -> None:
    """Persist credentials to disk (mode 0600)."""
    path = credentials_path()
    path.write_text(json.dumps(creds, indent=2))
    path.chmod(0o600)


# ------------------------------------------------------------------
# Configuration models
# ------------------------------------------------------------------


class LLMConfig(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(env_prefix="LEUK_LLM_", extra="ignore")

    provider: str = Field(
        default="anthropic",
        description="Active LLM provider: anthropic, openai, google, openrouter, local",
    )
    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model identifier to use",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=16384, gt=0)

    # Provider-specific keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    openrouter_api_key: str = ""

    # Anthropic auth token (Bearer token from Claude Pro/Max subscription)
    anthropic_auth_token: str = ""

    # Local model settings (vLLM / Ollama)
    local_base_url: str = "http://localhost:11434/v1"
    local_api_key: str = "ollama"  # Ollama ignores this; vLLM may need a real key


class SQLiteConfig(BaseSettings):
    """SQLite storage settings."""

    model_config = SettingsConfigDict(env_prefix="LEUK_SQLITE_", extra="ignore")

    path: str = Field(default="~/.config/leuk/leuk.db")


class AgentConfig(BaseSettings):
    """Top-level agent behaviour settings."""

    model_config = SettingsConfigDict(env_prefix="LEUK_", extra="ignore")

    max_tool_rounds: int = Field(
        default=50,
        description="Maximum consecutive tool-use rounds before forcing a text reply",
    )
    max_context_tokens: int = Field(
        default=100_000,
        description="Maximum estimated tokens in context window before truncation/summarization",
    )
    max_tool_result_tokens: int = Field(
        default=8_000,
        description="Maximum tokens for a single tool result before truncation",
    )
    context_strategy: str = Field(
        default="sliding_window",
        description="Context management strategy: 'sliding_window' or 'summarize'",
    )
    system_prompt: str = Field(
        default=(
            "You are leuk, a persistent AI agent with access to the local environment. "
            "You can execute shell commands, read files, and edit files. "
            "Think step-by-step and use tools when needed."
        ),
    )


class MCPServerConfig(BaseSettings):
    """Configuration for a single MCP server connection."""

    name: str = ""
    transport: str = Field(default="stdio", description="'stdio' or 'sse'")
    command: str = Field(default="", description="Command to run (stdio transport)")
    args: list[str] = Field(default_factory=list, description="Command arguments (stdio)")
    url: str = Field(default="", description="SSE endpoint URL (sse transport)")


class Settings(BaseSettings):
    """Root configuration container."""

    model_config = SettingsConfigDict(env_prefix="LEUK_SETTINGS_")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    mcp_servers: list[MCPServerConfig] = Field(
        default_factory=list,
        description="MCP servers to connect to on startup",
    )


def _env_file() -> Path | None:
    """Return the config.env path if it exists, else None."""
    p = config_env_path()
    return p if p.exists() else None


def load_settings() -> Settings:
    """Load settings from ~/.config/leuk/config.env + env vars, overlaying stored credentials.

    Precedence (highest first):
        1. Environment variables (LEUK_LLM_*, etc.)
        2. ~/.config/leuk/config.env
        3. ~/.config/leuk/credentials.json
        4. Defaults
    """
    env_file = _env_file()

    # pydantic-settings reads env_file on each sub-model; we pass it through
    # by constructing sub-models with the shared env_file path.
    kwargs: dict[str, object] = {}
    if env_file:
        kwargs["llm"] = LLMConfig(_env_file=env_file)
        kwargs["sqlite"] = SQLiteConfig(_env_file=env_file)
        kwargs["agent"] = AgentConfig(_env_file=env_file)

    settings = Settings(**kwargs)

    # Overlay credentials from ~/.config/leuk/credentials.json
    creds = load_credentials()
    if creds:
        if creds.get("anthropic_api_key") and not settings.llm.anthropic_api_key:
            settings.llm.anthropic_api_key = creds["anthropic_api_key"]
        if creds.get("anthropic_auth_token") and not settings.llm.anthropic_auth_token:
            settings.llm.anthropic_auth_token = creds["anthropic_auth_token"]
        if creds.get("openai_api_key") and not settings.llm.openai_api_key:
            settings.llm.openai_api_key = creds["openai_api_key"]
        if creds.get("google_api_key") and not settings.llm.google_api_key:
            settings.llm.google_api_key = creds["google_api_key"]
        if creds.get("openrouter_api_key") and not settings.llm.openrouter_api_key:
            settings.llm.openrouter_api_key = creds["openrouter_api_key"]
        if creds.get("local_api_key") and not settings.llm.local_api_key:
            settings.llm.local_api_key = creds["local_api_key"]

    return settings
