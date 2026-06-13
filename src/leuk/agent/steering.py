"""Steering for weaker/local models — pure helpers (no I/O).

Small local models (Ollama/vLLM) tend to give up: they stop mid-task, refuse
outright, or abandon a task after a single tool error, even with all the context
and tools they need. The agent loop in :mod:`leuk.agent.core` uses these helpers
to supply the discipline a strong model already has:

* :func:`compose_system_prompt` — append persistence/recovery rules to the system
  prompt when steering is active.
* :data:`STEERING_REMINDER` — a short reminder re-injected mid-loop.
* :data:`REFLECTION_PROMPT` + :func:`parse_reflection` — the bounded
  self-reflection check that decides whether a no-tool-call stop is a genuine
  completion or a premature give-up.
* :func:`continue_nudge` / :func:`truncation_nudge` — the messages injected to
  push the model to keep going.
* :data:`RECOVERY_HINT` — appended to errored tool results.

Gating is by config + the *provider* signal (:func:`steering_active`), never by
guessing model identity.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from leuk.types import ToolCall

if TYPE_CHECKING:
    from leuk.config import SteeringConfig


# ── Prompt text ───────────────────────────────────────────────────────────

STEERING_INSTRUCTIONS = """\
## Operating discipline
1. Persist until the task is verifiably done. Do not stop after one tool call,
   and do not ask the user to continue what you can do yourself.
2. Never refuse a task you have the tools for. Try with the tools rather than
   declining — you have a real local environment.
3. On a tool error or unexpected result, do NOT give up and do NOT repeat the
   identical call. First re-observe the actual state (re-read the file, re-run
   the check, take a fresh screenshot, re-read the page), then try a DIFFERENT
   approach.
4. Act, then verify the action had its intended effect before moving on.
5. When the task is genuinely complete, say so explicitly and stop — don't trail
   off mid-work."""

STEERING_REMINDER = (
    "[STEERING] Reminder: keep going until the task is verifiably done. If the "
    "last tool call errored or returned something unexpected, re-observe the "
    "current state and try a DIFFERENT approach — do not repeat the same call or "
    "give up. State explicitly when you are truly done."
)

REFLECTION_PROMPT = """\
[STEERING CHECK] Re-read the user's ORIGINAL request and review what you have
actually accomplished so far. Answer on the FIRST line with exactly one word:
- DONE — every part is finished AND verified; nothing remains.
- CONTINUE — anything is incomplete, you stopped due to an error or uncertainty,
  or you have not verified the result yet.
If CONTINUE, add a second line with the single concrete next action you will
take. Do not give up: if an approach failed, pick a DIFFERENT one."""

RECOVERY_HINT = (
    "\n\n[recovery hint] This error is recoverable — do not give up. Re-observe "
    "the current state, then adjust the arguments or try a different "
    "approach/tool and retry."
)

# Cues that signal an incomplete state even when the model doesn't emit the
# literal CONTINUE keyword (weak models often phrase it as "not done yet").
_NEGATIVE_CUES = (
    "not done",
    "n't done",
    "not complete",
    "not finished",
    "incomplete",
    "still need",
    "still have to",
    "not yet",
)


# ── Gating ────────────────────────────────────────────────────────────────


def steering_active(cfg: SteeringConfig, provider: str) -> bool:
    """Whether steering is active for *provider* under config *cfg*.

    ``'on'`` / ``'off'`` force the answer; ``'auto'`` (the default) activates
    steering only for the local/self-hosted provider — the population that needs
    it — leaving frontier providers untouched.
    """
    if cfg.enabled == "on":
        return True
    if cfg.enabled == "off":
        return False
    return provider == "local"  # "auto"


# ── System-prompt composition ─────────────────────────────────────────────


def compose_system_prompt(base: str, cfg: SteeringConfig, provider: str) -> str:
    """Return *base* augmented with steering instructions when active.

    When steering is inactive this returns *base* unchanged (exact equality), so
    strong models see no added tokens and no behaviour change. User overrides
    stay first as *base*; steering augments, never replaces.
    """
    if not steering_active(cfg, provider):
        return base
    out = f"{base}\n\n{STEERING_INSTRUCTIONS}"
    extra = cfg.extra_instructions.strip()
    if extra:
        out = f"{out}\n\n{extra}"
    return out


# ── Self-reflection parsing ───────────────────────────────────────────────


def parse_reflection(text: str | None) -> tuple[bool, str]:
    """Parse a self-reflection reply into ``(should_continue, next_action_hint)``.

    Robust to weak-model phrasing: explicit ``CONTINUE``/``DONE`` keywords,
    "not done"-style negations, and a ``CONTINUE: <action>`` one-liner are all
    handled. An unparseable reply defaults to *accept* (``should_continue`` is
    ``False``) — safe, since the caller is bounded by ``max_continuations``.
    """
    if not text or not text.strip():
        return (False, "")
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    hint = lines[1] if len(lines) > 1 else ""
    if not hint and ":" in first:
        hint = first.split(":", 1)[1].strip()
    low = text.lower()
    first_low = first.lower()

    # Incomplete-state cues win even without the literal keyword.
    if any(cue in low for cue in _NEGATIVE_CUES):
        return (True, hint)
    if "continue" in first_low:
        return (True, hint)
    if "done" in first_low:
        return (False, hint)
    if "continue" in low and "done" not in low:
        return (True, hint)
    return (False, hint)


# ── Nudge messages ────────────────────────────────────────────────────────


def continue_nudge(hint: str = "") -> str:
    """A steering message (Role.USER) pushing the model to keep working."""
    tail = (
        "Keep working with your tools until the task is actually complete and "
        "verified. Do not ask for confirmation, and if an approach failed, try a "
        "different one rather than repeating it."
    )
    if hint.strip():
        return f"[STEERING] You are not finished yet. Your stated next step: {hint.strip()} {tail}"
    return f"[STEERING] You are not finished yet. {tail}"


def truncation_nudge() -> str:
    """Nudge for a length-truncated reply — a plain continuation cue."""
    return "continue"


# ── Circle / repeated-loop detection ───────────────────────────────────────
#
# The opposite failure from giving up: a model that spins, re-issuing the same
# tool calls without progress. The hard ``max_tool_rounds`` ceiling eventually
# stops it, but only after wasting many rounds. These helpers detect a tight
# loop early (in a lengthy session) so the agent can redirect, then consolidate.

# How many of the most recent rounds must be identical to count as repetition.
_CIRCLE_REPEAT = 3


def tool_call_signature(tool_calls: list[ToolCall]) -> str:
    """A stable signature for a round's tool calls (name + canonical args).

    Two rounds that request the same tools with the same arguments produce the
    same signature, which is how repetition is detected. Argument key order is
    normalised so semantically-identical calls match.
    """
    parts: list[str] = []
    for tc in tool_calls:
        try:
            args = json.dumps(tc.arguments, sort_keys=True, default=str)
        except (TypeError, ValueError):
            args = str(tc.arguments)
        parts.append(f"{tc.name}({args})")
    return " | ".join(parts)


def detect_circling(
    round_sigs: list[str], *, min_rounds: int, repeat: int = _CIRCLE_REPEAT
) -> bool:
    """Whether the recent tool rounds look like a no-progress loop.

    Only fires once the session is *lengthy* (``len(round_sigs) >= min_rounds``),
    then flags either an **identical repeat** (the last ``repeat`` round
    signatures are all the same) or an **ABAB 2-cycle**. A model that varies its
    calls — even after one repeat — is not flagged, so a redirect nudge gets a
    fair chance before any escalation.
    """
    if len(round_sigs) < max(min_rounds, repeat):
        return False
    last = round_sigs[-repeat:]
    if len(set(last)) == 1:
        return True
    if len(round_sigs) >= 4:
        a, b, c, d = round_sigs[-4:]
        if a == c and b == d and a != b:
            return True
    return False


def circle_redirect_nudge() -> str:
    """First-line intervention: tell the model to stop repeating and redirect."""
    return (
        "[STEERING] You have repeated the same action(s) several times without "
        "making progress. Stop — do not issue that tool call again. Review the "
        "results you have ALREADY gathered above, then either answer from what you "
        "have or take a clearly DIFFERENT next step."
    )


def circle_consolidation_nudge() -> str:
    """Escalation: stop calling tools and consolidate what's been gathered."""
    return (
        "[STEERING] You are stuck repeating the same actions. Stop calling tools. "
        "Using only the results already gathered above, give your best answer or "
        "conclusion now, and state clearly what (if anything) still remains."
    )


# ── Text tool-call salvage ──────────────────────────────────────────────────
#
# Weak models often emit a tool call as plain TEXT instead of using the native
# function-calling API, e.g.:
#   <tool_call> <function=browser> <parameter=action> navigate
#               <parameter=url> https://… </tool_call>
# leuk then sees an assistant turn with content but no tool_calls, and the round
# ends with nothing executed. These helpers recover such calls into real
# ToolCalls. Recovery is validated against the *registered* tool names to avoid
# misreading prose that merely talks about tools.

_FUNC_RE = re.compile(r"<function\s*=\s*([^>\s]+)\s*>")
_PARAM_RE = re.compile(
    r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)"
    r"(?=<parameter\s*=|</parameter>|</function>|</tool_call>|<function\s*=|\Z)",
    re.DOTALL,
)
_TOOLCALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.DOTALL)


def _coerce(val: str) -> Any:
    """Best-effort type coercion of a stringified argument value."""
    v = val.strip()
    if not v:
        return v
    try:
        return json.loads(v)  # "5"→5, "true"→True, '"x"'→"x"; URLs/words stay str
    except (ValueError, TypeError):
        return v


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Return every top-level ``{...}`` JSON object in *text* (string-aware)."""
    objs: list[dict[str, Any]] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except ValueError:
                    obj = None
                if isinstance(obj, dict):
                    objs.append(obj)
                start = -1
    return objs


def _name_args(obj: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Extract (tool name, arguments) from a JSON tool-call object of any common shape."""
    fn = obj.get("function")
    if isinstance(fn, dict):  # OpenAI-style {"function": {"name", "arguments"}}
        obj = fn
    name = obj.get("name")
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters")
    if args is None:
        args = obj.get("args")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return (name if isinstance(name, str) else None, args)


def _calls_from_pseudo_xml(text: str, valid_names: set[str]) -> list[tuple[str, dict[str, Any]]]:
    """Parse ``<function=NAME> <parameter=KEY> VALUE …`` pseudo-XML tool calls."""
    calls: list[tuple[str, dict[str, Any]]] = []
    funcs = list(_FUNC_RE.finditer(text))
    for i, fm in enumerate(funcs):
        name = fm.group(1).strip()
        if valid_names and name not in valid_names:
            continue
        end = funcs[i + 1].start() if i + 1 < len(funcs) else len(text)
        segment = text[fm.end() : end]
        args = {key.strip(): _coerce(val) for key, val in _PARAM_RE.findall(segment)}
        calls.append((name, args))
    return calls


def parse_text_tool_calls(content: str | None, valid_names: set[str]) -> list[ToolCall]:
    """Recover tool calls a model emitted as plain text into real ``ToolCall``s.

    Handles the ``<function=…><parameter=…>`` pseudo-XML form, Hermes-style JSON
    inside ``<tool_call>`` blocks, and bare ``{"name", "arguments"}`` JSON. Only
    calls naming a *registered* tool (in *valid_names*) are returned. Returns an
    empty list when nothing tool-call-shaped is found.
    """
    if not content or ("<function=" not in content and "{" not in content):
        return []
    blocks = _TOOLCALL_BLOCK_RE.findall(content)
    spaces = blocks if blocks else [content]
    found: list[tuple[str, dict[str, Any]]] = []
    for space in spaces:
        for obj in _extract_json_objects(space):
            name, args = _name_args(obj)
            if name and (not valid_names or name in valid_names):
                found.append((name, args))
        found.extend(_calls_from_pseudo_xml(space, valid_names))
    return [
        ToolCall(id=f"salvaged_{i}", name=name, arguments=args)
        for i, (name, args) in enumerate(found)
    ]
