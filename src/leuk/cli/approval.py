"""Tool-approval prompt for the REPL.

Approval is consistent with every other leuk dialog: arrow to an option, Enter
to choose, Esc/q to cancel (= deny). The full-screen TUI shows this as an in-app
overlay with Tab→amend and Ctrl+E explanation (see ``cli/tui.py``); the classic
line-prompt uses the themed ``_radio`` dialog below.

The "always allow / deny" options are scoped to a **meaningful pattern** derived
from the tool + args (e.g. "`pkg-config` commands", "edits in `src/game/`")
rather than the whole tool or the verbatim argument — so the grant is semantic,
not a blanket trust of every shell command nor a one-off exact-string match.
"""

from __future__ import annotations

from html import escape as _esc

from leuk.safety import ApprovalResult, command_danger
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
    "history": "Search the conversation history",
    "sub_agent": "Spawn a sub-agent",
    "skill": "Use a skill",
}

# Tool whose primary argument the user can edit at the approval prompt (Tab).
_AMENDABLE_ARG: dict[str, str] = {
    "shell": "command",
    "file_edit": "path",
    "file_read": "path",
    "web_fetch": "url",
    "web_search": "query",
    "browser": "url",
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


def amendable_field(tool_call: ToolCall) -> str | None:
    """The argument name the user may edit via Tab→amend, or None."""
    field = _AMENDABLE_ARG.get(tool_call.name)
    return field if field and field in (tool_call.arguments or {}) else None


def approval_scope(tool_call: ToolCall) -> tuple[str, str]:
    """Return ``(glob_pattern, human_label)`` for an always-allow/deny grant.

    The pattern is what SafetyGuard persists and matches future calls against
    (via fnmatch on the tool's primary argument); the label is shown to the user.
    Scoped to a meaningful unit per tool, falling back to the whole tool.
    """
    name = tool_call.name
    args = tool_call.arguments or {}

    if name == "shell":
        prog = (args.get("command", "").strip().split() or [""])[0]
        prog = prog.rsplit("/", 1)[-1]  # /usr/bin/find → find
        if prog:
            return f"{prog} *", f"`{prog}` commands"
    elif name in ("file_edit", "file_read"):
        path = args.get("path", "").strip()
        if path:
            parent = path.rsplit("/", 1)[0] if "/" in path else "."
            verb = "edits" if name == "file_edit" else "reads"
            return f"{parent}/*", f"{verb} in `{parent}/`"
    elif name in ("web_fetch", "browser"):
        host = _url_host(args.get("url", ""))
        if host:
            verb = "fetching" if name == "web_fetch" else "browsing"
            return f"*{host}*", f"{verb} `{host}`"
    elif name == "input_control":
        action = args.get("action", "").strip()
        if action:
            return f"{action}*", f"`{action}` actions"

    return "*", f"all {name} calls"


def _url_host(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return urlparse(url).netloc or ""
    except ValueError:
        return ""


def risk_assessment(tool_call: ToolCall, reason: str) -> tuple[str, str]:
    """Return ``(level, explanation)`` — level is 'low' | 'medium' | 'high'.

    Heuristic, not name-guessed beyond the shared dangerous-command patterns:
    destructive/elevated shell ops and desktop control rank high; whole-file
    overwrites and writes rank medium; reads/searches rank low.
    """
    name = tool_call.name
    args = tool_call.arguments or {}

    if name == "shell":
        danger = command_danger(args.get("command", ""))
        if danger:
            return "high", danger
        return "low", "Shell command; not matched as destructive or elevated."
    if name == "input_control" and args.get("action") not in ("screenshot", "geometry"):
        return "high", "Controls the real keyboard/mouse — effects are immediate."
    if name == "file_edit":
        if args.get("overwrite"):
            return "medium", "Overwrites the file's entire contents."
        return "medium", "Edits a file on disk."
    if name in ("file_read", "web_fetch", "web_search", "history", "monitoring"):
        return "low", "Read-only — no changes to your system."
    if "overwrite" in reason.lower() or "dangerous" in reason.lower():
        return "high", reason
    return "medium", reason


def choice_to_result(choice: str | None, scope: str = "*") -> ApprovalResult:
    """Map a dialog choice to an ApprovalResult. None (Esc/q) → deny once."""
    return {
        "allow": ApprovalResult(approved=True),
        "allow_always": ApprovalResult(approved=True, remember=True, scope_pattern=scope),
        "deny_always": ApprovalResult(approved=False, remember=True, scope_pattern=scope),
    }.get(choice or "", ApprovalResult(approved=False))


def approval_dialog(reason: str, tool_call: ToolCall) -> ApprovalResult:
    """Blocking themed approval dialog (classic line-prompt). Run via to_thread."""
    from leuk.cli.settings_dialog import _radio

    pattern, label = approval_scope(tool_call)
    level, explanation = risk_assessment(tool_call, reason)
    body = (
        f"{humanise(tool_call)}\n\n{primary_detail(tool_call)}\n\n"
        f"{level.upper()} risk · {explanation}\n\n{reason}"
    )
    choice = _radio(
        "Permission required",
        _esc(body),
        [
            ("allow", "  Allow once"),
            ("allow_always", f"  Always allow {label}"),
            ("deny", "  Deny once"),
            ("deny_always", f"  Always deny {label}"),
        ],
        "allow",  # Enter on the default = allow (Esc/q = deny)
    )
    return choice_to_result(choice, pattern)
