"""Tool-approval prompt for the REPL.

Replaces the old typed ``y/n`` prompt (which used a second ``PromptSession`` that
collided with ``rich.Live`` — keypresses were swallowed and Enter resolved as
empty → silent auto-deny, plan §3.2/§4.2). This reuses the themed, Enter-to-select
``_radio`` dialog every other leuk dialog uses, so approval is consistent: arrow
to an option, Enter to choose, Esc/q to cancel (= deny). No second PromptSession.

The full event-driven version (approval emitted from ``SafetyGuard.gate()`` and
resolved in the render loop) lands with the persistent-input TUI, where the
approval becomes a modal overlay — see docs/repl-tui-design.md.
"""

from __future__ import annotations

from html import escape as _esc

from leuk.safety import ApprovalResult
from leuk.types import ToolCall

# Human phrasing for the action header (not "shell(...)").
_TOOL_VERBS: dict[str, str] = {
    "shell": "Run a shell command",
    "file_edit": "Edit a file",
    "file_read": "Read a file",
    "web_fetch": "Fetch a web page",
    "web_search": "Search the web",
    "browser": "Drive the browser",
    "input_control": "Control the desktop (keyboard/mouse)",
    "monitoring": "Read host data",
    "memory_write": "Write to memory",
    "sub_agent": "Spawn a sub-agent",
    "skill": "Use a skill",
}


def humanise(tool_call: ToolCall) -> str:
    """A human-readable header for the action."""
    return _TOOL_VERBS.get(tool_call.name, f"Use the {tool_call.name} tool")


def primary_detail(tool_call: ToolCall) -> str:
    """The most relevant argument (command / path / url), for the dialog body."""
    a = tool_call.arguments or {}
    for key in ("command", "path", "url", "query", "action", "text"):
        if a.get(key):
            return f"{key}: {a[key]}"
    if a:
        return ", ".join(f"{k}={v!r}" for k, v in a.items())
    return "(no arguments)"


def choice_to_result(choice: str | None) -> ApprovalResult:
    """Map a dialog choice to an ApprovalResult. None (Esc/q) → deny once."""
    return {
        "allow": ApprovalResult(approved=True),
        "allow_always": ApprovalResult(approved=True, remember=True),
        "deny_always": ApprovalResult(approved=False, remember=True),
    }.get(choice or "", ApprovalResult(approved=False))


def approval_dialog(reason: str, tool_call: ToolCall) -> ApprovalResult:
    """Blocking themed approval dialog. Run via ``asyncio.to_thread``."""
    from leuk.cli.settings_dialog import _radio

    body = f"{humanise(tool_call)}\n\n{primary_detail(tool_call)}\n\n{reason}"
    scope = tool_call.name
    choice = _radio(
        "Permission required",
        _esc(body),
        [
            ("allow", "  Allow once"),
            ("allow_always", f"  Always allow {scope}"),
            ("deny", "  Deny once"),
            ("deny_always", f"  Always deny {scope}"),
        ],
        "allow",  # Enter on the default = allow (Esc/q = deny)
    )
    return choice_to_result(choice)
