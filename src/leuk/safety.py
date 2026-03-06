"""Safety guardrails for tool execution.

Provides three layers of protection:

1. **Dangerous-operation detection** — regex-based detection of destructive
   shell commands, with automatic escalation to user confirmation.
2. **Read-only sandbox mode** — a master toggle that blocks all write
   operations (shell, file_edit, sub_agent).
3. **Allowlist / blocklist rules** — configurable per-tool ``ToolRule``
   entries evaluated in priority order (deny > ask > allow).
"""

from __future__ import annotations

import fnmatch
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from leuk.config import PermissionAction, SafetyConfig, ToolRule
from leuk.types import ToolCall

logger = logging.getLogger(__name__)

# ── Dangerous shell-command patterns ───────────────────────────────
# Each tuple is (compiled_regex, human-readable reason).

_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Filesystem destruction
    (re.compile(r"\brm\s+.*-[a-zA-Z]*r"), "recursive delete"),
    (re.compile(r"\brm\s+.*-[a-zA-Z]*f"), "force delete"),
    (re.compile(r"\bmkfs\b"), "format filesystem"),
    (re.compile(r"\bshred\b"), "secure-erase file"),
    (re.compile(r"\bdd\s+.*\bof=/"), "raw write to device/path"),
    # Permission / ownership escalation
    (re.compile(r"\bchmod\s+.*777\b"), "world-writable permission"),
    (re.compile(r"\bchmod\s+.*\+s\b"), "setuid/setgid bit"),
    (re.compile(r"\bchown\s+.*\broot\b"), "change owner to root"),
    (re.compile(r"\bsudo\b"), "superuser command"),
    # Network / remote-code execution
    (re.compile(r"\bcurl\b.*\|\s*(ba)?sh"), "pipe download to shell"),
    (re.compile(r"\bwget\b.*\|\s*(ba)?sh"), "pipe download to shell"),
    (re.compile(r"\bcurl\b.*\|\s*python"), "pipe download to python"),
    # System modification
    (re.compile(r"\bsystemctl\s+(stop|disable|mask|restart)"), "modify systemd service"),
    (re.compile(r"\breboot\b"), "reboot system"),
    (re.compile(r"\bshutdown\b"), "shutdown system"),
    (re.compile(r"\bkillall\b"), "kill processes by name"),
    # Git destructive
    (re.compile(r"\bgit\s+push\s+.*--force"), "force push"),
    (re.compile(r"\bgit\s+push\s+-f\b"), "force push"),
    (re.compile(r"\bgit\s+reset\s+--hard"), "hard reset"),
    (re.compile(r"\bgit\s+clean\s+-[a-zA-Z]*f"), "force clean untracked files"),
]

# Tools that perform write / mutation operations.
_WRITE_TOOLS = frozenset({"shell", "file_edit", "sub_agent"})

# Shell operators used to chain commands.
_SHELL_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")
# Command-substitution patterns.
_CMD_SUB_RE = re.compile(r"\$\(([^)]+)\)")
_BACKTICK_RE = re.compile(r"`([^`]+)`")


# ── Data types ─────────────────────────────────────────────────────


@dataclass(slots=True)
class SafetyCheck:
    """Result of evaluating a tool call against safety rules."""

    verdict: PermissionAction
    reason: str
    rule: ToolRule | None = None
    dangerous_match: str = ""


# ── SafetyGuard ────────────────────────────────────────────────────


class SafetyGuard:
    """Evaluates tool calls against safety rules before execution.

    Parameters
    ----------
    config:
        The safety configuration (rules, protected paths, read_only toggle).
    confirm_callback:
        An async callable that presents the user with a confirmation prompt
        and returns ``True`` if the user approves.  Signature::

            async def confirm(reason: str, tool_call: ToolCall) -> bool
    project_root:
        Resolved project root directory.  Defaults to cwd.
    """

    def __init__(
        self,
        config: SafetyConfig,
        confirm_callback: Callable[[str, ToolCall], Awaitable[bool]],
        project_root: Path | None = None,
    ) -> None:
        self.config = config
        self._confirm = confirm_callback
        self.project_root = (project_root or Path.cwd()).resolve()
        self._session_approvals: set[str] = set()

        # Resolve protected paths once.
        self._protected: list[Path] = [
            Path(p).expanduser().resolve() for p in config.protected_paths
        ]

    # ── public API ─────────────────────────────────────────────

    async def gate(self, tool_call: ToolCall) -> SafetyCheck:
        """Full gate: evaluate rules, prompt user if needed.

        Returns the *final* verdict after any user interaction.
        """
        check = self.check(tool_call)

        if check.verdict == PermissionAction.ASK:
            # Check session-level approvals first.
            approval_key = f"{tool_call.name}:{_primary_arg(tool_call)}"
            if approval_key in self._session_approvals:
                return SafetyCheck(
                    verdict=PermissionAction.ALLOW,
                    reason="Approved earlier this session",
                )

            approved = await self._confirm(check.reason, tool_call)
            if approved:
                self._session_approvals.add(approval_key)
                return SafetyCheck(
                    verdict=PermissionAction.ALLOW,
                    reason="User approved",
                )
            return SafetyCheck(
                verdict=PermissionAction.DENY,
                reason="User denied",
            )

        return check

    def check(self, tool_call: ToolCall) -> SafetyCheck:
        """Evaluate a tool call without prompting.

        Returns a :class:`SafetyCheck` whose *verdict* may be
        ``ALLOW``, ``ASK``, or ``DENY``.
        """
        tool = tool_call.name

        # 1. Read-only mode blocks all writes.
        if self.config.read_only and tool in _WRITE_TOOLS:
            return SafetyCheck(
                verdict=PermissionAction.DENY,
                reason="Read-only mode is enabled",
            )

        # 2. Path containment for file operations.
        if tool in ("file_read", "file_edit"):
            path_str = tool_call.arguments.get("path", "")
            if path_str:
                path_check = self._check_path(path_str, write=(tool == "file_edit"))
                if path_check.verdict != PermissionAction.ALLOW:
                    return path_check

        # 3. Evaluate DENY rules first — explicit deny always wins.
        deny_check = self._evaluate_rules_for(tool_call, PermissionAction.DENY)
        if deny_check is not None:
            return deny_check

        # 4. Dangerous-command detection for shell.
        if tool == "shell":
            command = tool_call.arguments.get("command", "")
            danger = self._check_dangerous_command(command)
            if danger is not None:
                return danger

        # 5. Evaluate remaining rules (ask, allow).
        return self._evaluate_rules(tool_call)

    # ── internal helpers ───────────────────────────────────────

    def _check_path(self, path_str: str, *, write: bool) -> SafetyCheck:
        """Check a file path against project boundary and protected paths."""
        try:
            resolved = Path(path_str).expanduser().resolve()
        except (OSError, ValueError) as exc:
            return SafetyCheck(
                verdict=PermissionAction.DENY,
                reason=f"Cannot resolve path: {exc}",
            )

        # Protected paths — always deny writes, warn on reads.
        if write:
            for pp in self._protected:
                try:
                    resolved.relative_to(pp)
                    return SafetyCheck(
                        verdict=PermissionAction.DENY,
                        reason=f"Write to protected path: {resolved}",
                    )
                except ValueError:
                    pass

        # Project boundary — deny writes outside project root.
        if write and self.project_root:
            try:
                resolved.relative_to(self.project_root)
            except ValueError:
                return SafetyCheck(
                    verdict=PermissionAction.ASK,
                    reason=f"Write outside project root ({self.project_root}): {resolved}",
                )

        return SafetyCheck(verdict=PermissionAction.ALLOW, reason="Path OK")

    def _check_dangerous_command(self, command: str) -> SafetyCheck | None:
        """Scan a shell command (including chained sub-commands) for danger.

        Returns a ``SafetyCheck`` with ``ASK`` or ``DENY`` if a dangerous
        pattern is found, or ``None`` if the command looks safe.
        """
        # First check the full command string — this catches patterns
        # that span across pipes (e.g. ``curl ... | bash``).
        for pattern, reason in _DANGEROUS_PATTERNS:
            if pattern.search(command):
                return SafetyCheck(
                    verdict=PermissionAction.ASK,
                    reason=f"Dangerous operation detected: {reason}",
                    dangerous_match=command.strip(),
                )
        # Then split compound commands and check each part individually
        # for patterns that only appear in sub-commands.
        parts = _split_shell_command(command)
        for part in parts:
            for pattern, reason in _DANGEROUS_PATTERNS:
                if pattern.search(part):
                    return SafetyCheck(
                        verdict=PermissionAction.ASK,
                        reason=f"Dangerous operation detected: {reason}",
                        dangerous_match=part.strip(),
                    )
        return None

    def _evaluate_rules_for(
        self, tool_call: ToolCall, action: PermissionAction
    ) -> SafetyCheck | None:
        """Check rules for a single action level.  Returns the first match or None."""
        primary = _primary_arg(tool_call)
        for rule in self.config.rules:
            if rule.action != action:
                continue
            if rule.tool != "*" and rule.tool != tool_call.name:
                continue
            if fnmatch.fnmatch(primary, rule.pattern):
                return SafetyCheck(
                    verdict=rule.action,
                    reason=f"Matched rule: {rule.tool}:{rule.pattern} → {rule.action}",
                    rule=rule,
                )
        return None

    def _evaluate_rules(self, tool_call: ToolCall) -> SafetyCheck:
        """Walk through the configured rules in priority order.

        Deny rules are skipped here (already checked earlier in ``check``).
        Evaluates ask, then allow.  The first match wins.
        """
        for action in (PermissionAction.ASK, PermissionAction.ALLOW):
            match = self._evaluate_rules_for(tool_call, action)
            if match is not None:
                return match

        # No rule matched — default to allow.
        return SafetyCheck(
            verdict=PermissionAction.ALLOW,
            reason="No matching rule (default allow)",
        )


# ── Utility functions ──────────────────────────────────────────────


def _primary_arg(tool_call: ToolCall) -> str:
    """Extract the primary argument used for rule matching."""
    match tool_call.name:
        case "shell":
            return tool_call.arguments.get("command", "")
        case "file_read" | "file_edit":
            return tool_call.arguments.get("path", "")
        case "web_fetch":
            return tool_call.arguments.get("url", "")
        case "sub_agent":
            return tool_call.arguments.get("task", "")
        case _:
            # For unknown tools, try common argument names.
            for key in ("command", "path", "url", "input"):
                if key in tool_call.arguments:
                    return str(tool_call.arguments[key])
            return ""


def _split_shell_command(command: str) -> list[str]:
    """Split a compound shell command into individual sub-commands.

    Handles ``&&``, ``||``, ``;``, ``|``, ``$(...)`` and backtick
    substitutions.
    """
    parts = _SHELL_SPLIT_RE.split(command)
    # Also extract command substitutions.
    parts.extend(_CMD_SUB_RE.findall(command))
    parts.extend(_BACKTICK_RE.findall(command))
    return [p for p in parts if p.strip()]
