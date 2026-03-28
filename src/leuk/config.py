"""Application configuration via environment variables and ~/.config/leuk/."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
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


def persistent_config_path() -> Path:
    """Return the path to the persistent config file (config.json)."""
    return config_dir() / "config.json"


def load_persistent_config() -> dict[str, Any]:
    """Load persistent configuration (e.g. last-used provider/model, voice settings).

    Returns a dict like::

        {"last_provider": "anthropic", "last_model": "claude-sonnet-4-20250514",
         "stt_model_size": "turbo", "stt_language": "ru",
         "tts_speaker": "ru_karina", "tts_en_speaker": "en_0",
         "tts_language": "ru",
         "vad_sensitivity": "0.5", "vad_silence_timeout": "1.0",
         "vad_min_speech": "0.5"}
    """
    path = persistent_config_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_persistent_config(values: dict[str, Any]) -> None:
    """Persist configuration to disk."""
    path = persistent_config_path()
    # Merge with existing config so we don't clobber unrelated keys.
    existing = load_persistent_config()
    existing.update(values)
    path.write_text(json.dumps(existing, indent=2))


def load_credentials() -> dict[str, str]:
    """Load saved credentials from disk.

    Returns a dict like:
        {"anthropic_api_key": "sk-...", "anthropic_auth_token": "...", ...}
    """
    path = credentials_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
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
        default="zen",
        description="Active LLM provider: zen, anthropic, openai, google, openrouter, local",
    )
    model: str = Field(
        default="big-pickle",
        description="Model identifier to use",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=16384, gt=0)

    # Provider-specific keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    openrouter_api_key: str = ""
    zen_api_key: str = ""

    # Anthropic auth token (Bearer token from Claude Pro/Max subscription)
    anthropic_auth_token: str = ""

    # Local model settings (vLLM / Ollama)
    local_base_url: str = "http://localhost:11434/v1"
    local_api_key: str = "ollama"  # Ollama ignores this; vLLM may need a real key


class LocalLLMConfig(BaseSettings):
    """Local LLM tool configuration (Ollama)."""

    model_config = SettingsConfigDict(env_prefix="LEUK_LOCAL_LLM_", extra="ignore")

    enabled: bool = Field(default=False, description="Enable the local_llm tool")
    base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    default_model: str = Field(default="llama3.2", description="Default Ollama model")


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


class PermissionAction(StrEnum):
    """What to do when a safety rule matches."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class ToolRule(BaseModel):
    """A single permission rule for a tool.

    Rules are evaluated in priority order: *deny* first, then *ask*, then
    *allow*.  The first match within each priority level wins.

    Examples::

        ToolRule(tool="shell", pattern="git status*", action="allow")
        ToolRule(tool="shell", pattern="rm *", action="ask")
        ToolRule(tool="file_edit", pattern="/etc/**", action="deny")
    """

    tool: str = Field(description="Tool name to match, or '*' for all tools")
    pattern: str = Field(description="Glob pattern matched against the tool's primary argument")
    action: PermissionAction = Field(description="Action when this rule matches")


def _default_safety_rules() -> list[ToolRule]:
    """Sensible defaults: allow common dev commands, ask for dangerous ops."""
    return [
        # ── DENY ──────────────────────────────────────────────────
        ToolRule(tool="file_read", pattern=".env", action="deny"),
        ToolRule(tool="file_read", pattern=".env.*", action="deny"),
        ToolRule(tool="file_read", pattern="**/*.pem", action="deny"),
        ToolRule(tool="file_read", pattern="**/*.key", action="deny"),
        ToolRule(tool="file_read", pattern="**/secrets/**", action="deny"),
        ToolRule(tool="file_edit", pattern="/etc/**", action="deny"),
        ToolRule(tool="file_edit", pattern="~/.ssh/**", action="deny"),
        # ── ASK (dangerous ops) ───────────────────────────────────
        ToolRule(tool="shell", pattern="rm *", action="ask"),
        ToolRule(tool="shell", pattern="sudo *", action="ask"),
        ToolRule(tool="shell", pattern="docker *", action="ask"),
        ToolRule(tool="shell", pattern="pip install *", action="ask"),
        ToolRule(tool="shell", pattern="npm install *", action="ask"),
        # ── ALLOW ─────────────────────────────────────────────────
        ToolRule(tool="file_read", pattern="*", action="allow"),
        ToolRule(tool="shell", pattern="*", action="allow"),
        ToolRule(tool="file_edit", pattern="*", action="allow"),
        ToolRule(tool="web_fetch", pattern="*", action="allow"),
        ToolRule(tool="sub_agent", pattern="*", action="allow"),
    ]


class SafetyConfig(BaseModel):
    """Safety guardrails configuration."""

    read_only: bool = Field(
        default=False,
        description="When true, all write operations are blocked",
    )
    project_root: str = Field(
        default="",
        description="Root directory for file operations (defaults to cwd at startup)",
    )
    protected_paths: list[str] = Field(
        default_factory=lambda: [
            "/etc",
            "/boot",
            "/usr",
            "/sbin",
            "/bin",
            "/lib",
            "/lib64",
            "/proc",
            "/sys",
            "/dev",
            "/var/log",
            "/var/run",
            "~/.ssh",
            "~/.gnupg",
            "~/.aws",
            "~/.kube",
            "~/.docker",
        ],
        description="Paths that are always denied for write operations",
    )
    rules: list[ToolRule] = Field(
        default_factory=_default_safety_rules,
        description="Permission rules (deny > ask > allow, first match wins)",
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
    local_llm: LocalLLMConfig = Field(default_factory=LocalLLMConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
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
        if creds.get("zen_api_key") and not settings.llm.zen_api_key:
            settings.llm.zen_api_key = creds["zen_api_key"]
        if creds.get("local_api_key") and not settings.llm.local_api_key:
            settings.llm.local_api_key = creds["local_api_key"]

    # Apply last-used provider/model from config.json when the user hasn't
    # explicitly overridden them via env vars or config.env.  We detect this
    # by checking whether both values are still at their compile-time defaults.
    # NB: We read Field.default directly — constructing LLMConfig() would pick
    # up the current env vars and defeat the purpose of the check.
    _default_provider = LLMConfig.model_fields["provider"].default
    _default_model = LLMConfig.model_fields["model"].default
    if settings.llm.provider == _default_provider and settings.llm.model == _default_model:
        pconfig = load_persistent_config()
        if pconfig.get("last_provider"):
            settings.llm.provider = pconfig["last_provider"]
        if pconfig.get("last_model"):
            settings.llm.model = pconfig["last_model"]

    return settings
