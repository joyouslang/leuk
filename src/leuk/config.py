"""Application configuration.

Primary source is ``~/.config/leuk/config.json`` (written by ``/settings``);
``LEUK_*`` environment variables override it for CI/Docker/power users. A legacy
``config.env`` is auto-migrated into config.json on first run.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

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
    context_window: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Override the model's context-window size (tokens) used for the "
            "usage gauge. When unset, it is queried from the provider / a "
            "lookup table."
        ),
    )
    thinking: bool = Field(
        default=False,
        description=(
            "Request extended thinking/reasoning from the model (Anthropic "
            "thinking, Gemini thought summaries). Models that always reason "
            "(DeepSeek-style reasoning_content) surface it regardless. If the "
            "active model doesn't support the parameter, the API will say so."
        ),
    )
    thinking_budget: int = Field(
        default=8192,
        gt=0,
        description="Token budget for extended thinking (Anthropic budget_tokens)",
    )

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
    max_concurrent_sub_agents: int = Field(
        default=5,
        description="Maximum number of sub-agents that can run concurrently; additional spawns queue",
        gt=0,
    )
    max_context_tokens: int | None = Field(
        default=None,
        description=(
            "Optional override (tokens) for the compaction budget. When unset "
            "(default), the budget is derived from the model's own context window "
            "— queried from the provider (or LEUK_LLM_CONTEXT_WINDOW) — reserving "
            "room for the reply. Not a hardcoded value."
        ),
    )
    max_tool_result_tokens: int = Field(
        default=8_000,
        description="Maximum tokens for a single tool result before truncation",
    )
    system_prompt: str = Field(
        default=(
            "You are leuk, a persistent AI agent with access to the local environment. "
            "You can execute shell commands, read and edit files, and use any other "
            "tools provided to you (e.g. browser, desktop control) when they are "
            "available. You are multimodal: when the user shares images or audio, or "
            "a tool returns a screenshot, you can see/hear them directly — analyse "
            "them; never claim you can only handle text. Prefer using your tools to "
            "act rather than saying you cannot. "
            "Every change you make to an existing file must be a PATCH: use file_edit "
            "with old_string/new_string to change only the text that differs, never "
            "re-emit the whole file, and don't rewrite files by piping/redirecting "
            "through the shell. Replacing a file's entire contents (file_edit "
            "overwrite=true) is a last resort that needs explicit user approval. "
            "Think step-by-step."
        ),
    )


class PermissionAction(StrEnum):
    """What to do when a safety rule matches."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class ReviewPolicy(StrEnum):
    """Review policy mode controlling which tools require user approval.

    Modes from least to most restrictive:

    - ``auto``     — never ask, all tools proceed automatically
    - ``agent``    — heuristic: ask only on dangerous shell ops (rm, sudo, …)
    - ``cautious`` — ask on all writes (file_edit, shell); reads auto-allowed
    - ``strict``   — also ask on reads (file_read, web_fetch)
    - ``paranoid`` — ask for every single tool call
    """

    AUTO = "auto"
    AGENT = "agent"
    CAUTIOUS = "cautious"
    STRICT = "strict"
    PARANOID = "paranoid"


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

    review_policy: ReviewPolicy = Field(
        default=ReviewPolicy.CAUTIOUS,
        description="Review policy mode: auto, agent, cautious (default), strict, paranoid",
    )
    approval_timeout: int = Field(
        default=120,
        gt=0,
        description="Seconds to wait for a channel approval button press before auto-denying",
    )
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
        default_factory=list,
        description="User-defined permission rules (prepended to policy rules, deny > ask > allow)",
    )


def _default_resource_limits() -> dict[str, str]:
    return {"memory": "512m", "cpus": "1.0", "pids": "256"}


class SandboxConfig(BaseModel):
    """Container sandbox configuration."""

    mode: Literal["none", "container"] = Field(
        default="none",
        description="Sandbox mode: 'none' (disabled) or 'container' (Docker isolation)",
    )
    image: str = Field(
        default="leuk-sandbox:latest",
        description="Docker image to use for the sandbox container",
    )
    allowed_mounts: list[str] = Field(
        default_factory=list,
        description=(
            "Host paths to bind-mount into the container. "
            "Format: 'host_path[:container_path][:rw]'. "
            "Defaults to read-only. Sensitive paths (.ssh, .aws, etc.) are blocked."
        ),
    )
    resource_limits: dict[str, str] = Field(
        default_factory=_default_resource_limits,
        description="Docker resource limits: memory, cpus, pids",
    )


class ChannelsConfig(BaseSettings):
    """Per-channel credentials and enable flags.

    Configure in ``config.json`` (the ``channels`` section) or via env vars with
    the ``LEUK_CHANNELS_`` prefix, e.g.::

        # ~/.config/leuk/config.json
        {"channels": {"telegram_bot_token": "123456:ABC-...",
                      "discord_bot_token": "MT..."}}

        # or, as env vars
        LEUK_CHANNELS_TELEGRAM_BOT_TOKEN=123456:ABC-...
    """

    model_config = SettingsConfigDict(env_prefix="LEUK_CHANNELS_", extra="ignore")

    # ── Pipe (non-interactive stdin/stdout) ───────────────────────────────
    pipe_enabled: bool = Field(
        default=True,
        description=(
            "Enable the non-interactive pipe channel (stdin/stdout). Active "
            "only when stdin is not a TTY, e.g. `echo '…' | leuk`."
        ),
    )

    # ── Telegram ──────────────────────────────────────────────────────────
    telegram_bot_token: str = Field(
        default="",
        description="Telegram Bot API token (from @BotFather)",
    )

    # ── Slack ─────────────────────────────────────────────────────────────
    slack_bot_token: str = Field(
        default="",
        description="Slack Bot User OAuth Token (xoxb-…)",
    )
    slack_app_token: str = Field(
        default="",
        description="Slack App-Level Token for Socket Mode (xapp-…)",
    )

    # ── Discord ───────────────────────────────────────────────────────────
    discord_bot_token: str = Field(
        default="",
        description="Discord bot token",
    )

    # ── Access control ───────────────────────────────────────────────────
    allowed_users: list[str] = Field(
        default_factory=list,
        description=(
            "User IDs allowed to interact via channels (Telegram, Slack, Discord). "
            "Empty list = unrestricted. Values are platform sender IDs."
        ),
    )


class MemoryConfig(BaseModel):
    """Hierarchical memory system configuration."""

    enabled: bool = Field(default=True, description="Load memory files at session init")
    memory_dir: str = Field(
        default="~/.config/leuk/memory",
        description="Root directory for memory files",
    )
    project_name: str = Field(
        default="",
        description="Project name for per-project memory (auto-detected from cwd/.git if empty)",
    )
    token_budget: int = Field(
        default=4000,
        description="Maximum tokens for combined memory context; excess truncated from top of global memory",
    )


class ArchiveConfig(BaseModel):
    """Conversation archive settings."""

    enabled: bool = Field(
        default=True,
        description="Write dropped messages to markdown archive files before compaction",
    )
    directory: str = Field(
        default="~/.local/share/leuk/archives",
        description="Directory where archive files are written",
    )


class BrowserConfig(BaseModel):
    """Browser automation settings."""

    enabled: bool = Field(default=False, description="Enable the browser tool")
    headless: bool = Field(
        default=False,
        description=(
            "Run the browser invisibly. Default False — the browser window is "
            "visible so you can watch the agent; set True for headless servers."
        ),
    )


class MonitoringConfig(BaseSettings):
    """Read-only host monitoring tool (screenshots, geometry, system info)."""

    model_config = SettingsConfigDict(env_prefix="LEUK_MONITORING_", extra="ignore")

    enabled: bool = Field(default=False, description="Enable the read-only monitoring tool")


class InputControlConfig(BaseSettings):
    """Keyboard/mouse desktop-control tool settings (Linux X11 + Wayland)."""

    model_config = SettingsConfigDict(env_prefix="LEUK_INPUT_CONTROL_", extra="ignore")

    enabled: bool = Field(default=False, description="Enable the input_control tool")
    backend: Literal["ydotool"] = Field(default="ydotool", description="Injection backend")
    ydotool_socket: str | None = Field(
        default=None, description="Path to the ydotoold socket (YDOTOOL_SOCKET)"
    )
    verify: Literal["on_failure", "each_action", "never"] = Field(
        default="on_failure",
        description="When to attach a verification screenshot to results",
    )
    auto_approve: bool = Field(
        default=False,
        description="Auto-approve desktop-control actions (DANGEROUS; agent self-verifies)",
    )


class RoleDefinition(BaseModel):
    """Definition of a named agent role for team-based orchestration."""

    system_prompt: str = Field(
        default="",
        description="Custom system prompt for this role (empty = inherit from agent config)",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Allowed tool names for this role (empty = all tools)",
    )
    provider: str = Field(
        default="",
        description="Provider override for this role (empty = inherit from agent config)",
    )
    max_rounds: int = Field(
        default=0,
        description="Max tool rounds override (0 = inherit from agent config)",
    )


def _default_roles() -> dict[str, RoleDefinition]:
    """Built-in role presets shipped as examples."""
    return {
        "researcher": RoleDefinition(
            system_prompt=(
                "You are a research agent. Your goal is to gather information, "
                "fetch web content, and synthesize findings into a clear summary."
            ),
            tools=["web_fetch", "local_llm"],
        ),
        "coder": RoleDefinition(
            system_prompt=(
                "You are a coding agent. Your goal is to write, read, and execute "
                "code to implement the requested functionality."
            ),
            tools=["shell", "file_edit", "file_read"],
        ),
        "reviewer": RoleDefinition(
            system_prompt=(
                "You are a code review agent. Your goal is to read code and provide "
                "thorough, actionable feedback on correctness, style, and potential issues."
            ),
            tools=["file_read"],
        ),
    }


class AgentTeamsConfig(BaseModel):
    """Configuration for agent teams and predefined role definitions."""

    roles: dict[str, RoleDefinition] = Field(
        default_factory=_default_roles,
        description="Named role definitions available to AgentTeam",
    )


class SchedulerConfig(BaseSettings):
    """Scheduled task runner configuration."""

    model_config = SettingsConfigDict(env_prefix="LEUK_SCHEDULER_", extra="ignore")

    enabled: bool = Field(default=False, description="Enable the background task scheduler")
    poll_interval: int = Field(
        default=60, gt=0, description="Seconds between polls for due tasks"
    )

class MCPServerConfig(BaseSettings):
    """Configuration for a single MCP server connection."""

    name: str = ""
    transport: str = Field(default="stdio", description="'stdio' or 'sse'")
    command: str = Field(default="", description="Command to run (stdio transport)")
    args: list[str] = Field(default_factory=list, description="Command arguments (stdio)")
    url: str = Field(default="", description="SSE endpoint URL (sse transport)")
    env: dict[str, str] = Field(
        default_factory=dict, description="Extra environment variables for the stdio subprocess"
    )
    enabled: bool = Field(default=True, description="Connect on startup (toggle off to keep but disable)")


class MCPExposureConfig(BaseModel):
    """Configuration for exposing leuk itself as an MCP server."""

    enabled: bool = Field(default=False, description="Start the MCP server on launch")
    transport: str = Field(default="stdio", description="'stdio' (subprocess) or 'sse'")


class McpRegistryConfig(BaseSettings):
    """Settings for importing MCP connectors/plugins from registries."""

    model_config = SettingsConfigDict(env_prefix="LEUK_MCP_REGISTRY_", extra="ignore")

    url: str = Field(
        default="https://registry.modelcontextprotocol.io",
        description="Base URL of the official MCP server registry",
    )
    default_source: str = Field(
        default="mcp", description="Default import source: 'mcp', 'clawhub', or 'url'"
    )


class SkillsConfig(BaseSettings):
    """Agent Skills (SKILL.md) runtime settings."""

    model_config = SettingsConfigDict(env_prefix="LEUK_SKILLS_", extra="ignore")

    enabled: bool = Field(default=False, description="Enable the skills runtime + the 'skill' tool")
    directory: str = Field(
        default="~/.config/leuk/skills", description="Directory holding installed skill bundles"
    )
    disabled: list[str] = Field(
        default_factory=list, description="Slugs of installed-but-disabled skills"
    )
    trusted: list[str] = Field(
        default_factory=list,
        description="Slugs the user has reviewed and trusted (a skill is inert until trusted)",
    )
    max_index_skills: int = Field(
        default=50, gt=0, description="Maximum skills listed in the tool's index"
    )


class UIConfig(BaseSettings):
    """Terminal UI preferences."""

    model_config = SettingsConfigDict(env_prefix="LEUK_UI_", extra="ignore")

    media_render: str = Field(
        default="metadata",
        description=(
            "How media blocks render in the history browser: 'metadata' (a compact "
            "info line, no binary) or 'inline' (ANSI image thumbnail; Enter opens/plays)"
        ),
    )


class Settings(BaseSettings):
    """Root configuration container."""

    model_config = SettingsConfigDict(env_prefix="LEUK_SETTINGS_")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    local_llm: LocalLLMConfig = Field(default_factory=LocalLLMConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    archive: ArchiveConfig = Field(default_factory=ArchiveConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    input_control: InputControlConfig = Field(default_factory=InputControlConfig)
    agent_teams: AgentTeamsConfig = Field(default_factory=AgentTeamsConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    mcp_servers: list[MCPServerConfig] = Field(
        default_factory=list,
        description="MCP servers to connect to on startup",
    )
    mcp_server: MCPExposureConfig = Field(
        default_factory=MCPExposureConfig,
        description="Expose leuk itself as an MCP server",
    )
    mcp_registry: McpRegistryConfig = Field(default_factory=McpRegistryConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    channels: ChannelsConfig = Field(
        default_factory=ChannelsConfig,
        description="Multi-channel messaging credentials and settings",
    )


# Legacy ``config.env`` key prefixes → the Settings sub-model they configure.
# Used only to migrate an existing config.env into config.json (see below).
_ENV_PREFIX_TO_MODEL: list[tuple[str, str]] = [
    ("LEUK_LOCAL_LLM_", "local_llm"),
    ("LEUK_INPUT_CONTROL_", "input_control"),
    ("LEUK_MONITORING_", "monitoring"),
    ("LEUK_MCP_REGISTRY_", "mcp_registry"),
    ("LEUK_CHANNELS_", "channels"),
    ("LEUK_SCHEDULER_", "scheduler"),
    ("LEUK_BROWSER_", "browser"),
    ("LEUK_SKILLS_", "skills"),
    ("LEUK_SQLITE_", "sqlite"),
    ("LEUK_LLM_", "llm"),
    ("LEUK_UI_", "ui"),
    ("LEUK_", "agent"),  # AgentConfig — keep last (shortest prefix)
]


def _env_key_to_field(key: str) -> tuple[str | None, str | None]:
    """Map a ``LEUK_*`` env key to ``(submodel, field)``, longest prefix first."""
    for prefix, model in sorted(_ENV_PREFIX_TO_MODEL, key=lambda p: -len(p[0])):
        if key.startswith(prefix):
            return model, key[len(prefix) :].lower()
    return None, None


def migrate_legacy_config_env() -> None:
    """One-time: fold a legacy ``~/.config/leuk/config.env`` into config.json.

    leuk now keeps all persistent configuration in the single ``config.json``
    file (env vars still override it). If an old ``config.env`` exists, its
    ``LEUK_*=value`` lines are converted to nested config.json sections and the
    file is renamed so this runs only once.
    """
    src = config_env_path()
    if not src.exists():
        return
    nested: dict[str, dict[str, Any]] = {}
    try:
        lines = src.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        model, field = _env_key_to_field(key.strip())
        if model is None or field is None:
            continue
        value: Any = val.strip().strip('"').strip("'")
        try:  # turn "0.7"/"true"/"[1,2]" into real JSON types
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        nested.setdefault(model, {})[field] = value
    if nested:
        pconfig = load_persistent_config()
        for model, fields in nested.items():
            existing = pconfig.get(model)
            # Existing config.json wins over the migrated values.
            pconfig[model] = {**fields, **existing} if isinstance(existing, dict) else fields
        save_persistent_config(pconfig)
    try:
        src.rename(src.with_name("config.env.migrated"))
    except OSError:
        pass


def _overlay_config_json(settings: Settings, pconfig: dict[str, Any]) -> None:
    """Apply nested config.json sections (e.g. ``{"llm": {"temperature": 0.2}}``)
    onto *settings*, but never override a field an env var already set."""
    for name in type(settings).model_fields:
        sub = getattr(settings, name)
        override = pconfig.get(name)
        if not isinstance(sub, BaseModel) or not isinstance(override, dict):
            continue
        env_set = sub.model_fields_set
        merged = sub.model_dump()
        changed = False
        for field, value in override.items():
            if field in type(sub).model_fields and field not in env_set:
                merged[field] = value
                changed = True
        if changed:
            # model_validate coerces types (e.g. "0.2" → float) and does NOT
            # re-read env vars, so env-set fields (already in merged) are kept.
            setattr(settings, name, type(sub).model_validate(merged))


def load_settings() -> Settings:
    """Load settings from env vars + ~/.config/leuk/config.json, overlaying credentials.

    Precedence (highest first):
        1. Environment variables (``LEUK_LLM_*`` …) — for power users / CI / Docker
        2. ~/.config/leuk/config.json (written by ``/settings``)
        3. ~/.config/leuk/credentials.json (API keys, mode 0600)
        4. Compiled-in defaults

    A legacy ``config.env`` is migrated into config.json by
    :func:`migrate_legacy_config_env` (called once at startup), so this function
    only reads config.json — it has no side effects.
    """
    settings = Settings()

    # Overlay nested config.json sections (the home of what config.env used to
    # carry) before the flat keys below; env vars still win.
    pconfig = load_persistent_config()
    _overlay_config_json(settings, pconfig)

    # ``mcp_servers`` is a list, which ``_overlay_config_json`` (BaseModel-only)
    # skips and env vars can't carry — load it explicitly from config.json.
    if not settings.mcp_servers and isinstance(pconfig.get("mcp_servers"), list):
        settings.mcp_servers = [
            MCPServerConfig.model_validate(s)
            for s in pconfig["mcp_servers"]
            if isinstance(s, dict)
        ]

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
        # Channel credentials
        if creds.get("telegram_bot_token") and not settings.channels.telegram_bot_token:
            settings.channels.telegram_bot_token = creds["telegram_bot_token"]

    # Apply last-used provider/model from config.json when the user hasn't
    # explicitly overridden them via an env var.  We detect this by checking
    # whether both values are still at their compile-time defaults.
    # NB: We read Field.default directly — constructing LLMConfig() would pick
    # up the current env vars and defeat the purpose of the check.
    _default_provider = LLMConfig.model_fields["provider"].default
    _default_model = LLMConfig.model_fields["model"].default
    if settings.llm.provider == _default_provider and settings.llm.model == _default_model:
        if pconfig.get("last_provider"):
            settings.llm.provider = pconfig["last_provider"]
        if pconfig.get("last_model"):
            settings.llm.model = pconfig["last_model"]

    # Apply last-used review policy from config.json
    if pconfig.get("review_policy"):
        try:
            settings.safety.review_policy = ReviewPolicy(pconfig["review_policy"])
        except ValueError:
            pass

    # Apply persisted feature toggles from config.json (set via /settings) when
    # not already forced on via env vars.
    if not settings.browser.enabled and pconfig.get("browser_enabled"):
        settings.browser.enabled = True
    if not settings.input_control.enabled and pconfig.get("input_control_enabled"):
        settings.input_control.enabled = True
    if not settings.monitoring.enabled and pconfig.get("monitoring_enabled"):
        settings.monitoring.enabled = True
    if not settings.input_control.auto_approve and pconfig.get("input_control_auto_approve"):
        settings.input_control.auto_approve = True
    if not settings.skills.enabled and pconfig.get("skills_enabled"):
        settings.skills.enabled = True
    if pconfig.get("media_render") in ("metadata", "inline"):
        settings.ui.media_render = pconfig["media_render"]

    return settings
