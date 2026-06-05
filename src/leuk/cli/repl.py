"""Interactive REPL for the leuk agent.

Architecture
~~~~~~~~~~~~
The REPL uses an :class:`AgentSession` that drives the agent autonomously
in a background ``asyncio.Task``.  When the user submits a message:

1. The text is pushed into ``AgentSession.input_queue``.
2. A **render task** consumes events from ``AgentSession.event_queue`` and
   displays them via :class:`StreamRenderer`.
3. The **input task** continues accepting user text in parallel — new
   messages are queued as *comments* the agent sees on its next round,
   and ``Ctrl-C`` triggers :meth:`AgentSession.interrupt`.

The user can ``/detach`` to disconnect the view (session keeps running)
and on next startup the REPL automatically re-attaches to any running
session.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from leuk.agent.session import AgentSession
from leuk.agent.sub_agent import SubAgentManager
from leuk.config import config_dir, load_settings, migrate_legacy_config_env
from leuk.safety import SafetyGuard
from leuk.persistence import create_hot_store
from leuk.persistence.sqlite import SQLiteStore
from leuk.providers import create_provider
from leuk.providers.base import LLMProvider, NoCredentialsError
from leuk.tools import create_default_registry
from leuk.tools.sub_agent import SubAgentTool
from leuk.types import (
    Message,
    Role,
    Session,
)
from leuk.cli.render import StreamRenderer
from leuk.cli import theme as _theme
from leuk.cli.theme import LEUK_THEME


logger = logging.getLogger(__name__)


class _TransientPollingNoiseFilter(logging.Filter):
    """Drop aiogram's transient long-polling reconnect log records.

    During normal operation aiogram's long-poll connection is periodically
    dropped by Telegram's servers; aiogram logs this at ERROR and then retries
    after a short sleep (logged at WARNING). Both are recovered from
    automatically and are pure noise. This filter suppresses only those known
    transient messages on ``aiogram`` loggers — any other (real) error passes.
    """

    _PATTERNS = (
        "Failed to fetch updates",
        "Sleep for",
        "try again",
        "ServerDisconnectedError",
        "TelegramNetworkError",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith("aiogram"):
            msg = record.getMessage()
            if any(p in msg for p in self._PATTERNS):
                return False
        return True


console = Console(theme=LEUK_THEME)


def _build_pt_style(p: dict[str, str]) -> Style:
    """Build the prompt_toolkit style (prompt, footer, completion dropdown)
    from a theme palette. prompt_toolkit can't read the rich theme, so we
    mirror the active palette's colours here.

    The completion menu explicitly uses ``bg:default`` (the theme/terminal
    background) on every row — otherwise prompt_toolkit's built-in light-grey
    menu background bleeds through and makes the grey text unreadable. Only
    the selected row gets the accent background.
    """
    dark = "#11111b"  # near-black text for the highlighted (accent-bg) row
    return Style.from_dict(
        {
            "prompt": f"{p['green']} bold",
            "input": p["fg"],
            "bottom-toolbar": f"{p['grey']} bg:default",
            # Command-completion dropdown (shown below the input).
            "completion-menu": f"bg:default {p['fg']}",
            "completion-menu.completion": f"bg:default {p['fg']}",
            "completion-menu.completion.current": f"bg:{p['blue']} {dark} bold",
            "completion-menu.meta.completion": f"bg:default {p['grey']}",
            "completion-menu.meta.completion.current": f"bg:{p['blue']} {dark}",
            # Scrollbar (shown when the list overflows).
            "completion-menu.scrollbar.background": "bg:default",
            "completion-menu.scrollbar.button": f"bg:{p['grey']}",
        }
    )


STYLE = _build_pt_style(_theme.PALETTE)


# ── Slash commands ─────────────────────────────────────────────────
# Single source of truth for /help and Tab autocompletion.
# Each entry: (command, args-hint, description).
COMMANDS: list[tuple[str, str, str]] = [
    ("/help", "", "Show this help message"),
    ("/models", "", "Select model"),
    ("/new", "", "Start a new session"),
    ("/sessions", "", "List recent sessions"),
    ("/subagents", "[<id>]", "List sub-agent sessions, or view one's history"),
    ("/switch", "<id>", "Switch to session by id prefix"),
    ("/rename", "<name>", "Rename current session"),
    ("/delete", "[<id>]", "Delete current (or another) session"),
    ("/detach", "", "Detach from session (keeps running)"),
    ("/auth", "", "Select provider / manage credentials"),
    ("/readonly", "", "Toggle read-only mode (block all writes)"),
    ("/safety", "", "Show safety guardrail status"),
    ("/tasks", "", "List scheduled tasks"),
    ("/policy", "<mode>", "Show or set review policy"),
    ("/desktop-auto", "", "Toggle desktop-control auto-approval (DANGEROUS)"),
    ("/approvals", "", "List saved tool approvals (clear to reset)"),
    ("/status", "", "Show session stats and context usage"),
    ("/history", "", "Browse the conversation: Tab to open, ↑/↓ select, Enter expand"),
    ("/file", "<path>", "Attach a file (image/audio, auto-detected) to your next message"),
    ("/voice", "", "Toggle voice input"),
    ("/speak", "", "Toggle text-to-speech output"),
    ("/doctor", "", "Check optional-feature setup (ydotool, screenshots, …)"),
    ("/skills", "", "Add / enable / disable agent skills (SKILL.md)"),
    ("/mcp", "", "Add / enable / disable MCP connectors / plugins"),
    ("/settings", "", "Configure STT/TTS/VAD settings"),
    ("/retry", "", "Re-send the last message (after an error)"),
    ("/quit", "", "Exit leuk (alias: /exit)"),
]

# Sentinel returned by the prompt when the user presses Tab on an empty line —
# the main loop opens the interactive history browser instead of submitting.
_OPEN_HISTORY = object()

# Per-server cap on MCP startup connection so a hung/slow server can't freeze
# launch (it's skipped with a warning instead).
_MCP_CONNECT_TIMEOUT = 15.0


class SlashCommandCompleter(Completer):
    """Tab/while-typing completion for leading ``/commands``.

    Completes only the first token and only when it begins with ``/`` — once a
    space follows the command (i.e. the user is typing an argument), no
    completions are offered. Each entry shows its description as meta text in
    the dropdown rendered below the input.
    """

    def __init__(self, commands: list[tuple[str, str, str]]) -> None:
        self._commands = commands

    def get_completions(self, document, complete_event):  # noqa: ANN001
        text = document.text_before_cursor
        # Only when the whole line is a single (partial) slash token.
        if not text.startswith("/") or " " in text:
            return
        for cmd, args, desc in self._commands:
            if cmd.startswith(text):
                meta = f"{args}  {desc}".strip() if args else desc
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=meta,
                )


def _render_message(msg: Message) -> None:
    """Display a non-streamed message to the user."""
    if msg.role == Role.ASSISTANT:
        if msg.content:
            console.print(Markdown(msg.content))
        if msg.tool_calls:
            for tc in msg.tool_calls:
                console.print(
                    Panel(
                        f"[bold]{tc.name}[/bold]({', '.join(f'{k}={v!r}' for k, v in tc.arguments.items())})",
                        title="[yellow]Tool Call[/yellow]",
                        border_style="yellow",
                        expand=False,
                    )
                )
    elif msg.role == Role.TOOL and msg.tool_result:
        _render_tool_result(msg)


def _render_tool_result(msg: Message) -> None:
    """Render a tool result message."""
    tr = msg.tool_result
    if tr is None:
        return
    style = "red" if tr.is_error else "green"
    title = f"[{style}]{tr.name}[/{style}]"
    content = tr.content
    if len(content) > 2000:
        content = content[:2000] + f"\n... [{len(tr.content)} chars total]"
    console.print(Panel(content, title=title, border_style=style, expand=False))


# ── Continuous voice input via VAD ────────────────────────────────


async def _warmup_stt(stt: object) -> None:
    """Eagerly load the STT model so first transcription is fast."""
    if hasattr(stt, "_ensure_model"):
        logger.info("Warming up STT model …")
        await asyncio.to_thread(stt._ensure_model)  # type: ignore[union-attr]
        logger.info("STT model ready")


def _norm_speech(text: str) -> str:
    """Normalise text for echo comparison (lowercase, alnum/space only)."""
    return " ".join("".join(c if c.isalnum() else " " for c in text.lower()).split())


# Stock phrases Whisper hallucinates on silence / background noise — artefacts of
# its training data (YouTube captions). They are never real voice commands here,
# so they're discarded outright. Stored normalised (lowercase, alnum + spaces).
_WHISPER_HALLUCINATIONS = frozenset(
    _norm_speech(p)
    for p in (
        # English
        "thank you", "thank you.", "thanks for watching", "thanks for watching!",
        "thank you for watching", "thank you very much", "thank you so much",
        "please subscribe", "subscribe", "subscribe to my channel",
        "don't forget to subscribe", "like and subscribe",
        "we'll see you in the next video", "see you in the next video",
        "see you next time", "see you in the next one", "i'll see you next time",
        "bye", "bye bye", "goodbye", "you", "the end", "okay", "ok",
        "thanks for listening", "thank you for listening",
        # Russian
        "спасибо", "спасибо за просмотр", "спасибо за внимание",
        "подписывайтесь на канал", "продолжение следует", "субтитры",
        "субтитры подготовлены сообществом", "до новых встреч",
        # Spanish
        "gracias", "gracias por ver", "gracias por ver el video",
        "suscríbete", "suscríbete al canal", "hasta la próxima",
        "subtítulos realizados por la comunidad de amara org",
    )
)


def _is_hallucination(text: str) -> bool:
    """True if *text* looks like an STT hallucination on noise/echo.

    Catches two forms: degenerate repetition (a token/char repeated, e.g.
    "СССССС…") and the stock phrases Whisper emits on non-speech (e.g.
    "Thank you.", "We'll see you in the next video.") that otherwise get
    mistaken for a user interruption and loop the agent.
    """
    cand = _norm_speech(text)
    if not cand:
        return True
    if cand in _WHISPER_HALLUCINATIONS:
        return True
    compact = cand.replace(" ", "")
    if len(compact) >= 8 and len(set(compact)) <= 2:
        return True  # essentially one or two characters repeated
    words = cand.split()
    if len(words) >= 6 and len(set(words)) <= 2:
        return True  # one or two words repeated
    return False


# Phrases that signal the title model "answered" instead of labelling.
_BAD_TITLE_MARKERS = (
    "i can't", "i cannot", "i can not", "i'm a", "i am a", "as an", "as a ",
    "language model", "i don't", "i do not", "sorry", "unable to",
    "я не могу", "я не умею", "не могу", "извини", "к сожалению",
    "i'm unable", "i am unable", "cannot help", "can't help",
)


def _good_title(title: str) -> bool:
    """Heuristic: is *title* a usable short topic label (not a refusal/answer)?"""
    title = title.strip()
    if not title:
        return False
    low = title.lower()
    if any(m in low for m in _BAD_TITLE_MARKERS):
        return False
    if len(title) > 60 or len(title.split()) > 9:
        return False
    # Reject prose-like sentences (multiple clauses / ends like a sentence).
    if title.count(".") >= 2 or title.endswith((".", "!", "?")) and len(title.split()) > 6:
        return False
    return True


def _fallback_title(first_message: str) -> str:
    """Derive a title straight from the user's message (no model)."""
    cleaned = " ".join(first_message.strip().split())
    if not cleaned:
        return "New session"
    words = cleaned.split()
    label = " ".join(words[:7])
    if len(label) > 50:
        label = label[:50].rsplit(" ", 1)[0]
    return label[:1].upper() + label[1:] if label else "New session"


class _VoiceInputBridge:
    """Bridges ContinuousVAD → REPL by transcribing speech segments
    and pushing the text into an asyncio.Queue.

    The REPL's main loop reads from ``text_queue`` alongside the
    keyboard prompt, whichever fires first wins.

    Before transcription, the clip is trimmed with Silero VAD's
    ``get_speech_timestamps`` to strip leading/trailing silence and
    any silent gaps.  This prevents Whisper hallucinations on the
    non-speech portions of the recording.
    """

    def __init__(self, stt: object, console: Console,
                 vad: object | None = None) -> None:
        self.stt = stt
        self.console = console
        self.text_queue: asyncio.Queue[str] = asyncio.Queue()
        self._vad = vad  # ContinuousVAD instance — used for trim_silence
        self._live: Live | None = None

    async def on_speech(self, clip: "AudioClip") -> None:  # noqa: F821
        """Called by ContinuousVAD when a speech segment ends."""
        from leuk.voice.recorder import ContinuousVAD, trim_silence

        if clip.duration < 0.2:
            return

        # Trim non-speech audio before transcription
        if (
            self._vad is not None
            and isinstance(self._vad, ContinuousVAD)
            and self._vad.vad_model is not None
            and self._vad.get_speech_timestamps is not None
        ):
            trimmed = await asyncio.to_thread(
                trim_silence,
                clip,
                self._vad.vad_model,
                self._vad.get_speech_timestamps,
                threshold=self._vad.vad_threshold,
            )
            if trimmed is None:
                logger.debug("No speech in clip after trimming, skipping")
                return
            logger.debug(
                "Trimmed %.1fs → %.1fs", clip.duration, trimmed.duration
            )
            clip = trimmed

        if clip.duration < 0.2:
            return

        # Show a transient "transcribing" indicator
        self.console.print("[dim]🎤 Transcribing …[/dim]", end="\r")

        logger.debug("Transcribing %.1fs clip …", clip.duration)
        text = await self.stt.transcribe(clip)  # type: ignore[union-attr]
        logger.debug("Transcription result: %r", text)

        if not (text and text.strip()):
            logger.debug("Empty transcription, ignoring")
            return
        text = text.strip()

        # Discard Whisper hallucinations on background noise (e.g. "Thank you.",
        # "We'll see you in the next video.") — they're never a real command.
        if _is_hallucination(text):
            logger.debug("Discarding likely STT hallucination: %r", text)
            return

        self.console.print(f"[cyan]🎤 {text}[/cyan]")
        await self.text_queue.put(text)


# ── Agent turn (push + render) ───────────────────────────────────


async def _run_agent_turn(
    agent_session: AgentSession,
    text: str,
    renderer: StreamRenderer,
    *,
    speak_mode: bool = False,
    tts_backend: object | None = None,
    voice_vad: object | None = None,
) -> None:
    """Push a user message to the agent session and render the response.

    This blocks until the agent turn completes (TURN_COMPLETE event).

    A single Ctrl-C cancels this task (see ``_on_sigint`` in the REPL loop):
    generation is interrupted *and* TTS playback is stopped at once, returning
    promptly to the prompt.

    When *speak_mode* is active, a :class:`~leuk.voice.tts.StreamingTTSSpeaker`
    is attached to the renderer so sentences are spoken as soon as they arrive
    (overlapping with continued text output on screen). Voice input is
    half-duplex: the mic VAD is **paused** while the agent speaks (and resumed
    after) to avoid a speaker→mic→VAD feedback loop.
    """
    streaming_speaker = None

    if speak_mode and tts_backend is not None:
        from leuk.voice.tts import StreamingTTSSpeaker

        # Pause the mic VAD during speech to avoid a feedback loop.
        if voice_vad is not None and hasattr(voice_vad, "pause"):
            voice_vad.pause()

        streaming_speaker = StreamingTTSSpeaker(tts_backend)  # type: ignore[arg-type]
        await streaming_speaker.start()
        renderer.set_tts_speaker(streaming_speaker)

    agent_session.push(text)

    try:
        # Render events until the turn finishes.
        await renderer.render_queue(agent_session.event_queue)

        # Flush remaining text and wait for all speech to finish.
        if streaming_speaker is not None:
            renderer.set_tts_speaker(None)
            try:
                await streaming_speaker.flush()
            except Exception as tts_exc:
                logger.debug("Streaming TTS flush error", exc_info=tts_exc)
                console.print(
                    f"[red dim]TTS error: {type(tts_exc).__name__}: {tts_exc}[/red dim]"
                )
    except asyncio.CancelledError:
        # Ctrl-C: interrupt generation immediately. The ``finally`` below stops
        # TTS playback at once (``stop()`` signals the audio thread synchronously)
        # so the agent and its voice both halt before we return to the prompt.
        agent_session.interrupt()
        raise
    finally:
        if streaming_speaker is not None:
            renderer.set_tts_speaker(None)
            await streaming_speaker.stop()
            # Resume VAD now that speech has stopped.
            if voice_vad is not None and hasattr(voice_vad, "resume"):
                voice_vad.resume()


async def _run_repl() -> None:
    """Main REPL loop."""
    migrate_legacy_config_env()  # one-time: fold any old config.env into config.json
    settings = load_settings()

    # Apply the persisted colour theme (default gruvbox) before anything is
    # drawn, so the banner and all output use it.
    from leuk.config import load_persistent_config as _load_pc_theme

    _theme.apply_theme(_load_pc_theme().get("theme", _theme.DEFAULT_THEME))
    console.push_theme(_theme.LEUK_THEME)
    pt_style = _build_pt_style(_theme.PALETTE)

    # Initialise backends
    sqlite = SQLiteStore(settings.sqlite)
    await sqlite.init()
    hot_store = create_hot_store()

    # Safety guardrails
    from leuk.safety import ApprovalResult

    async def _confirm_tool_use(reason: str, tool_call) -> ApprovalResult:
        """Prompt the user for permission during agent execution.

        Keys:
          * ``y`` — allow once (default on Enter)
          * ``n`` — deny once
          * ``Y`` — always allow this tool+pattern (persisted)
          * ``N`` — always deny this tool+pattern (persisted)
        """
        # ``rich.Live`` (spinner) and ``prompt_toolkit`` collide on stdin,
        # which caused every keypress to be swallowed and Enter to resolve
        # as empty → default deny. Pause the renderer before prompting.
        try:
            stream_renderer.pause()
        except Exception:
            pass

        args_str = ", ".join(f"{k}={v!r}" for k, v in tool_call.arguments.items())
        console.print(
            Panel(
                f"[bold]{tool_call.name}[/bold]({args_str})\n\n[yellow]{reason}[/yellow]",
                title="[red]Permission Required[/red]",
                border_style="red",
                expand=False,
            )
        )
        response = await asyncio.to_thread(
            prompt_session_for_confirm.prompt,
            HTML(
                "<prompt>[y] allow  [n] deny  [Y] always allow  [N] always deny "
                "(default: n): </prompt>"
            ),
        )
        choice = response.strip()

        # Empty (Enter) → deny once. Safer default for a security prompt —
        # matches the convention of ``rm -i`` and most sudo-style gates.
        if choice == "":
            return ApprovalResult(approved=False)
        if choice == "y":
            return ApprovalResult(approved=True)
        if choice == "n":
            return ApprovalResult(approved=False)
        if choice == "Y":
            return ApprovalResult(approved=True, remember=True)
        if choice == "N":
            return ApprovalResult(approved=False, remember=True)
        # Anything else (typo, full word like "yes") — be lenient: treat
        # lowercase yes/no as one-shot decisions, anything truly unparseable
        # falls through to safe default (deny once).
        if choice.lower() in ("yes", "ok", "allow"):
            return ApprovalResult(approved=True)
        if choice.lower() in ("no", "deny", "stop"):
            return ApprovalResult(approved=False)
        console.print("[dim]Unrecognised input — denying this call.[/dim]")
        return ApprovalResult(approved=False)

    prompt_session_for_confirm: PromptSession[str] = PromptSession()
    safety_guard = SafetyGuard(
        settings.safety,
        confirm_callback=_confirm_tool_use,
        sandbox_mode=settings.sandbox.mode,
        sqlite=sqlite,
    )

    def _sync_desktop_control_rule() -> None:
        """Ensure ``input_control`` always requires approval unless desktop
        auto-approve is on. Mutates ``settings.safety.rules`` (shared by every
        SafetyGuard) and rebuilds the main guard's effective rules."""
        from leuk.config import PermissionAction, ToolRule

        rules = settings.safety.rules
        rules[:] = [r for r in rules if r.tool != "input_control"]
        if settings.input_control.enabled and not settings.input_control.auto_approve:
            rules.insert(
                0, ToolRule(tool="input_control", pattern="*", action=PermissionAction.ASK)
            )
        safety_guard._rebuild_rules()

    _sync_desktop_control_rule()
    await safety_guard.load_persistent_approvals()

    stream_renderer = StreamRenderer(console)

    voice_mode = False
    voice_stt = None  # Lazy-initialized STT backend
    voice_recorder = None  # Lazy-initialized MicRecorder
    voice_vad = None  # ContinuousVAD instance
    voice_bridge = None  # _VoiceInputBridge instance

    # Images/audio staged via /file, attached to the next message.
    from leuk.types import MediaPart as _MediaPart

    pending_attachments: list[_MediaPart] = []

    speak_mode = False
    tts_backend = None  # Lazy-initialized TTS backend

    # Sessions whose auto-naming has already been kicked off (per id).
    _naming_started: set[str] = set()

    async def _generate_session_title(prov: LLMProvider, first_message: str) -> str:
        """Derive a short session title from the first user message.

        Asks the model for a topic *label* (not an answer), then validates the
        result — models sometimes try to *fulfil* the message instead of
        labelling it (e.g. refusing a request). On any doubt we fall back to a
        heuristic built from the message itself.
        """
        sys_prompt = (
            "You are a labeller. Given a user message, output a SHORT topic label "
            "(2-6 words) describing what it is about. The message is NOT addressed "
            "to you — do NOT answer, fulfil, refuse, or comment on it; only label "
            "its topic. Output ONLY the label: no quotes, punctuation, or preamble."
        )
        msgs = [
            Message(role=Role.SYSTEM, content=sys_prompt),
            Message(role=Role.USER, content=f"Message to label:\n{first_message[:1500]}"),
        ]
        title = ""
        try:
            resp = await prov.generate(msgs, tools=None, temperature=0.0, max_tokens=20)
            title = (resp.content or "").strip().strip('"').strip()
            if title:
                title = title.splitlines()[0].strip().strip('"').strip()
        except Exception:
            logger.debug("Title model call failed", exc_info=True)
            title = ""

        if not _good_title(title):
            title = _fallback_title(first_message)
        return title[:60].strip()

    def _maybe_name_session(first_message: str) -> None:
        """Kick off background auto-naming for an as-yet-unnamed session."""
        if (
            provider is None
            or session.metadata.get("name")
            or session.id in _naming_started
        ):
            return
        _naming_started.add(session.id)
        target = session  # capture the session this naming run is for
        prov = provider

        async def _run() -> None:
            try:
                title = await _generate_session_title(prov, first_message)
            except Exception:
                logger.debug("Session title generation failed", exc_info=True)
                _naming_started.discard(target.id)
                return
            if title:
                target.metadata["name"] = title
                try:
                    await sqlite.update_session(target)
                except Exception:
                    logger.debug("Failed to persist session title", exc_info=True)

        asyncio.ensure_future(_run())

    _branch_cache: dict[str, str] = {}
    # Resolved context-window size (tokens) for the active model, used as the
    # denominator of the usage gauge. Populated by _init_agent_session().
    _ctx_cache: dict[str, int | None] = {"window": settings.llm.context_window}

    def _git_branch() -> str:
        """Current git branch (cached for the session), or '' if not a repo."""
        from pathlib import Path

        if "b" not in _branch_cache:
            branch = ""
            head = Path.cwd() / ".git" / "HEAD"
            try:
                ref = head.read_text(encoding="utf-8").strip()
                if ref.startswith("ref:"):
                    branch = ref.rsplit("/", 1)[-1]
                else:
                    branch = ref[:7]  # detached HEAD → short sha
            except (OSError, ValueError):
                branch = ""
            _branch_cache["b"] = branch
        return _branch_cache["b"]

    def _status_toolbar() -> HTML:
        """gemini-cli-style footer: location on the left, stats on the right.

        Reads live state on every render, so /policy, the voice/speak
        toggles, and context usage are reflected immediately. Segments are
        separated by '·' and styled via inline colours.
        """
        from leuk.agent.context import estimate_total_tokens
        from leuk.cli.banner import short_cwd
        from leuk.cli.theme import PALETTE

        BLUE, CYAN, GREY, YELLOW, GREEN, RED = (
            PALETTE["blue"],
            PALETTE["cyan"],
            PALETTE["grey"],
            PALETTE["yellow"],
            PALETTE["green"],
            PALETTE.get("red", PALETTE["yellow"]),
        )
        sep = f' <style fg="{GREY}">·</style> '

        # Left: cwd · branch
        left = [f'<style fg="{BLUE}">{short_cwd()}</style>']
        branch = _git_branch()
        if branch:
            left.append(f'<style fg="{CYAN}"> {branch}</style>')

        # Right: model · context% · policy · mode flags
        if agent_session is None:
            # Pending: no session started yet.
            left.append(f'<style fg="{GREY}">new session</style>')
        else:
            sess_name = session.metadata.get("name")
            if sess_name:
                from prompt_toolkit.formatted_text.html import html_escape

                left.append(f'<style fg="{PALETTE["purple"]}">{html_escape(sess_name)}</style>')

        right = [f'<style fg="{CYAN}">{settings.llm.model}</style>']
        window = _ctx_cache["window"]
        if agent is not None:
            # Context usage: estimated prompt tokens (~ marks it an estimate) out
            # of the model's context window, e.g. "ctx ~8.5k/200k 4%".
            used = estimate_total_tokens(agent._messages)

            def _k(n: int) -> str:
                return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

            if window:
                pct = int(used / window * 100)
                ctx_color = RED if pct >= 90 else YELLOW if pct >= 70 else GREY
                right.append(
                    f'<style fg="{ctx_color}">ctx ~{_k(used)}/{_k(window)} {pct}%</style>'
                )
            else:
                right.append(f'<style fg="{GREY}">ctx ~{_k(used)} tokens</style>')

        policy = settings.safety.review_policy.value
        right.append(f'<style fg="{YELLOW}">{policy}</style>')

        flags = [m for m, on in (("voice", voice_mode), ("speak", speak_mode)) if on]
        if flags:
            right.append(f'<style fg="{GREEN}">{" ".join(flags)}</style>')

        line = " " + sep.join(left) + "   " + sep.join(right) + " "
        return HTML(line)

    provider = None  # may be None until credentials are configured
    try:
        provider = create_provider(settings.llm)
    except NoCredentialsError:
        console.print(
            "[yellow]No credentials configured for "
            f"'{settings.llm.provider}'. Run /auth to set up.[/yellow]"
        )

    skills_loader = None
    if settings.skills.enabled:
        from leuk.skills import SkillLoader

        skills_loader = SkillLoader(
            settings.skills.directory,
            project_dir=".leuk/skills",
            trusted=set(settings.skills.trusted),
            disabled=set(settings.skills.disabled),
            max_index=settings.skills.max_index_skills,
        )

    tools = create_default_registry(
        browser_enabled=settings.browser.enabled,
        browser_headless=settings.browser.headless,
        sandbox=settings.sandbox if settings.sandbox.mode == "container" else None,
        local_llm=settings.local_llm if settings.local_llm.enabled else None,
        input_control=settings.input_control if settings.input_control.enabled else None,
        skills_loader=skills_loader,
    )

    # Connect to MCP servers **in the background**, concurrently, so startup
    # never blocks on them. ``mcp_status`` tracks each server's live state for
    # the /mcp manager; connected servers register their tools into the live
    # registry (the agent reads tool specs each turn, so they appear once ready).
    mcp_clients: list["MCPClient"] = []
    mcp_status: dict[str, str] = {}
    mcp_connect_task: "asyncio.Task[None] | None" = None
    enabled_servers = [s for s in settings.mcp_servers if s.enabled]
    if enabled_servers:
        from leuk.mcp.bridge import MCPToolBridge
        from leuk.mcp.client import MCPClient

        for s in enabled_servers:
            mcp_status[s.name] = "connecting"

        async def _connect_one(srv_cfg) -> None:  # noqa: ANN001 — MCPServerConfig
            client = None
            try:
                if srv_cfg.transport == "stdio" and srv_cfg.command:
                    client = MCPClient.stdio(
                        srv_cfg.command, srv_cfg.args, name=srv_cfg.name, env=srv_cfg.env
                    )
                elif srv_cfg.transport == "sse" and srv_cfg.url:
                    client = MCPClient.sse(srv_cfg.url, name=srv_cfg.name)
                else:
                    mcp_status[srv_cfg.name] = "failed: invalid config"
                    return
                await asyncio.wait_for(client.connect(), timeout=_MCP_CONNECT_TIMEOUT)
                MCPToolBridge(client).register_tools(tools)
                mcp_clients.append(client)
                mcp_status[srv_cfg.name] = "connected"
            except Exception as exc:  # noqa: BLE001 — incl. asyncio.TimeoutError
                reason = "timed out" if isinstance(exc, asyncio.TimeoutError) else str(exc)
                mcp_status[srv_cfg.name] = f"failed: {reason[:80]}"
                if client is not None:
                    try:
                        await client.close()
                    except Exception:  # noqa: BLE001 — best-effort cleanup
                        pass

        async def _connect_all() -> None:
            await asyncio.gather(*(_connect_one(s) for s in enabled_servers), return_exceptions=True)

        mcp_connect_task = asyncio.create_task(_connect_all())

    # Always begin with a fresh session. Previous sessions remain available
    # via /sessions and /switch; the new session is named from its first
    # message (see _maybe_name_session). It is NOT persisted/started until the
    # user picks a session or sends a first message (lazy creation).
    session = Session(system_prompt=settings.agent.system_prompt)

    # ── Channel registry (Telegram, Slack, Discord, …) ────────────
    from leuk.channels import ChannelRegistry

    async def _channel_session_factory(
        channel_name: str, chat_id: str, channel: object
    ) -> AgentSession:
        """Create an AgentSession for an incoming channel chat.

        Each channel session gets its own :class:`SafetyGuard` with a
        ``confirm_callback`` that routes approval requests to the channel
        as interactive buttons.
        """
        from leuk.agent.core import Agent as _Agent

        # Build a channel-specific confirm callback.
        if hasattr(channel, "request_approval"):

            async def _channel_confirm(
                reason: str, tool_call: object
            ) -> ApprovalResult:
                args_str = ", ".join(
                    f"{k}={v!r}" for k, v in tool_call.arguments.items()  # type: ignore[union-attr]
                )
                return await channel.request_approval(  # type: ignore[union-attr]
                    chat_id, tool_call.name, args_str, reason  # type: ignore[union-attr]
                )

        else:
            # Channel doesn't support interactive approval — auto-deny with warning.
            async def _channel_confirm(
                reason: str, tool_call: object
            ) -> ApprovalResult:
                if channel is not None and hasattr(channel, "send"):
                    await channel.send(  # type: ignore[union-attr]
                        chat_id,
                        f"⚠️ Tool `{tool_call.name}` requires approval but this "  # type: ignore[union-attr]
                        f"channel doesn't support interactive buttons. Auto-denying.",
                    )
                return ApprovalResult(approved=False)

        channel_guard = SafetyGuard(
            settings.safety,
            confirm_callback=_channel_confirm,
            sandbox_mode=settings.sandbox.mode,
            sqlite=sqlite,
        )
        await channel_guard.load_persistent_approvals()

        sess = Session(system_prompt=settings.agent.system_prompt)
        sess.metadata["channel"] = channel_name
        sess.metadata["chat_id"] = chat_id
        ag = _Agent(
            settings=settings,
            provider=provider,  # type: ignore[arg-type]
            tool_registry=tools,
            sqlite=sqlite,
            hot_store=hot_store,
            session=sess,
            safety_guard=channel_guard,
        )
        await ag.init()
        return AgentSession(ag)

    channel_registry: ChannelRegistry | None = None
    if provider is not None:
        # Disable the REPL channel — the interactive REPL already handles
        # stdin/stdout via prompt_toolkit; the ReplChannel would race on stdin.
        settings.channels.repl_enabled = False
        channel_registry = ChannelRegistry(_channel_session_factory, settings.channels)
        await channel_registry.start()

    from leuk.agent.core import Agent

    def _make_agent(sess: Session, prov: LLMProvider) -> Agent:
        """Create an agent and wire up the sub-agent tool."""
        ag = Agent(
            settings=settings,
            provider=prov,
            tool_registry=tools,
            sqlite=sqlite,
            hot_store=hot_store,
            session=sess,
            safety_guard=safety_guard,
        )
        # Wire up the sub-agent manager into the sub_agent tool
        sub_tool = tools.get("sub_agent")
        if isinstance(sub_tool, SubAgentTool):
            mgr = SubAgentManager(
                settings=settings,
                provider=prov,
                tool_registry=tools,
                sqlite=sqlite,
                hot_store=hot_store,
            )
            sub_tool.set_manager(mgr)
        return ag

    agent: Agent | None = None
    agent_session: AgentSession | None = None

    async def _init_agent_session(sess: Session, prov: LLMProvider) -> tuple[Agent, AgentSession]:
        """Create an Agent + AgentSession and start the background loop."""
        ag = _make_agent(sess, prov)
        await ag.init()
        asess = AgentSession(ag)
        asess.start()
        # Resolve the model's context window for the usage gauge (provider may
        # have just changed via /models or /auth).
        await _resolve_context_window_for(prov)
        return ag, asess

    async def _resolve_context_window_for(prov: LLMProvider | None) -> None:
        from leuk.providers.context_window import resolve_context_window

        try:
            _ctx_cache["window"] = await resolve_context_window(settings.llm, prov)
        except Exception:
            _ctx_cache["window"] = settings.llm.context_window

    async def _stop_agent_session() -> None:
        """Stop the current agent session if running."""
        nonlocal agent, agent_session
        if agent_session is not None:
            await agent_session.stop()
            agent_session = None
        if agent is not None:
            await agent.shutdown()
            agent = None

    # Resolve the model's context window up front (no session needed) so the
    # usage gauge is accurate even before a session is started.
    if provider is not None:
        await _resolve_context_window_for(provider)

    # ── Task scheduler ────────────────────────────────────────────
    task_scheduler = None
    if settings.scheduler.enabled and provider is not None:
        from leuk.scheduler.runner import TaskScheduler

        task_scheduler = TaskScheduler(
            settings=settings,
            sqlite=sqlite,
            hot_store=hot_store,
            provider=provider,
            tool_registry=tools,
            safety_guard=safety_guard,
        )
        await task_scheduler.start()

    def _provider_label() -> str:
        if provider is None:
            return "[red]none (run /auth)[/red]"
        return f"[cyan]{settings.llm.provider}[/cyan]"

    _extra_channels = [
        ch for ch in (channel_registry.active_channels if channel_registry else [])
        if ch != "repl"
    ]

    from leuk import __version__
    from leuk.cli.banner import render_banner, short_cwd

    render_banner(
        console,
        version=__version__,
        provider_label=_provider_label(),
        model=settings.llm.model,
        channels=_extra_channels,
        cwd=short_cwd(),
    )

    def _show_history(*, clear: bool = False) -> None:
        """Replay the current session's stored conversation, if any.

        When *clear* is set the terminal is wiped first (used by /switch so
        the restored history starts from a clean screen).
        """
        from leuk.cli.render import render_history

        if agent is not None and agent._messages:
            if clear:
                console.clear()
            render_history(console, agent._messages)

    async def _pick_session_modal(
        sessions: list[Session],
        *,
        prompt_text: str = "Select a session to continue:",
    ) -> str | None:
        """Show a full-screen modal to choose a session to continue with.

        Returns the chosen session id, the sentinel ``"__new__"`` to start a
        fresh session, or ``None`` if the user pressed Esc/Cancel.
        """
        from leuk.cli.settings_dialog import _radio

        values: list[tuple[str, str]] = []
        for s in sessions:
            nm = s.metadata.get("name", "") or "(unnamed)"
            ts = s.updated_at.strftime("%Y-%m-%d %H:%M")
            values.append((s.id, f"  {s.id[:8]}  {nm}  ·  {ts}"))
        values.append(("__new__", "  + Start a new session"))

        return await asyncio.to_thread(
            _radio, "Choose a session", prompt_text, values, None
        )

    def _go_pending() -> None:
        """Enter the 'no active session' state: hold an unpersisted draft
        session but don't start/persist an agent until the user sends a
        message (or picks/loads a session)."""
        nonlocal session, agent, agent_session
        session = Session(system_prompt=settings.agent.system_prompt)
        agent = None
        agent_session = None

    history_path = config_dir() / "repl_history"

    # Tab on an EMPTY line opens the history browser (when text is present, Tab
    # falls through to the default slash-command completion).
    from prompt_toolkit.application import get_app as _get_app
    from prompt_toolkit.filters import Condition as _Condition
    from prompt_toolkit.key_binding import KeyBindings as _KeyBindings

    _prompt_kb = _KeyBindings()

    @_prompt_kb.add("tab", filter=_Condition(lambda: _get_app().current_buffer.text == ""))
    def _open_history_kb(event) -> None:  # noqa: ANN001
        event.app.exit(result=_OPEN_HISTORY)

    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path)),
        style=pt_style,
        bottom_toolbar=_status_toolbar,
        completer=SlashCommandCompleter(COMMANDS),
        complete_while_typing=True,
        key_bindings=_prompt_kb,
    )

    async def _open_history_browser() -> None:
        """Open the interactive history browser over the current conversation."""
        msgs = agent._messages if agent is not None else []
        if not msgs:
            console.print("[dim]No conversation history yet.[/dim]")
            return
        from leuk.cli.history_browser import browse_history

        try:
            await browse_history(msgs, media_mode=settings.ui.media_render)
        except Exception:
            logger.debug("history browser error", exc_info=True)

    # Startup: no session is created or resumed. The user either selects one
    # manually (/sessions, /switch) or a session is created on the first
    # message. Just show the banner (above) and a one-line hint.
    console.print(
        "[dim]No active session — type a message to begin, or "
        "[bold]/sessions[/bold] to list past ones.[/dim]"
    )

    while True:
        # ── Dual input: keyboard OR voice (whichever fires first) ──
        try:
            if voice_mode and voice_bridge is not None:
                # Race: async keyboard prompt vs. voice transcription queue.
                # We use prompt_async() instead of threading prompt() so that
                # cancellation properly cleans up prompt_toolkit's Application
                # state (avoids "Application is already running" assertion).
                kb_task = asyncio.ensure_future(
                    prompt_session.prompt_async(
                        HTML("<prompt>leuk> </prompt>"),
                    )
                )
                voice_task = asyncio.ensure_future(
                    voice_bridge.text_queue.get()
                )
                done, pending = await asyncio.wait(
                    {kb_task, voice_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, EOFError, KeyboardInterrupt):
                        pass
                winner = done.pop()
                user_input = winner.result()
            else:
                user_input = await asyncio.to_thread(
                    prompt_session.prompt,
                    HTML("<prompt>leuk> </prompt>"),
                )
        except EOFError:
            # Ctrl-D → quit.
            break
        except KeyboardInterrupt:
            # Ctrl-C at the prompt should NOT quit — it clears the current line
            # (prompt_toolkit already discarded the buffer) and reminds the user
            # how to actually exit. Quitting on a stray Ctrl-C loses the session.
            console.print(
                "[dim]Press Ctrl-D or /exit to quit.[/dim]"
            )
            continue

        # Tab on an empty line → interactive history browser.
        if user_input is _OPEN_HISTORY:
            await _open_history_browser()
            continue

        text = user_input.strip()
        if not text:
            continue

        # Handle slash commands
        if text in ("/quit", "/exit"):
            break
        if text == "/help":
            _w = max(len(c) + (len(a) + 1 if a else 0) for c, a, _ in COMMANDS)
            _lines = []
            for _cmd, _args, _desc in COMMANDS:
                _name = f"{_cmd} [dim]{_args}[/dim]" if _args else _cmd
                _pad = " " * (_w - (len(_cmd) + (len(_args) + 1 if _args else 0)) + 2)
                _lines.append(f"[bold]{_name}[/bold]{_pad}— {_desc}")
            console.print(
                Panel(
                    "\n".join(_lines),
                    title="[bold]Commands[/bold]",
                    border_style="accent.blue",
                    expand=False,
                )
            )
            continue
        if text == "/new":
            await _stop_agent_session()
            # Lazy: don't persist/start until the first message.
            _go_pending()
            console.print(
                "[dim]New session — type a message to begin.[/dim]"
            )
            continue
        if text == "/sessions":
            sessions = await sqlite.list_sessions(limit=10, top_level_only=True)
            current_id = session.id
            if not sessions:
                console.print("[dim]No sessions yet.[/dim]")
            for s in sessions:
                marker = " [bold green]*[/bold green]" if s.id == current_id else ""
                name = s.metadata.get("name", "")
                label = f"[cyan]{name}[/cyan] " if name else ""
                state_label = s.status.value
                # Show if this is the currently-running agent session
                if agent_session is not None and s.id == current_id and agent_session.running:
                    state_label = "[green]running[/green]"
                console.print(
                    f"  {s.id[:8]}  {label}{state_label:<10} "
                    f"updated {s.updated_at.strftime('%Y-%m-%d %H:%M')}{marker}"
                )
            continue
        if text.startswith("/subagents"):
            arg = text[len("/subagents") :].strip()
            if arg and arg != "all":
                # View a specific sub-agent session's history (read-only).
                children = await sqlite.list_child_sessions(limit=200)
                matches = [s for s in children if s.id.startswith(arg)]
                if not matches:
                    console.print(f"[red]No sub-agent session matching '{arg}'[/red]")
                    continue
                if len(matches) > 1:
                    console.print(
                        f"[yellow]Ambiguous — {len(matches)} match '{arg}':[/yellow]"
                    )
                    for s in matches[:5]:
                        console.print(f"  {s.id[:8]}  {s.metadata.get('task', '')[:60]}")
                    continue
                sub = matches[0]
                msgs = await sqlite.get_messages(sub.id)
                from leuk.cli.render import render_history

                task_desc = sub.metadata.get("task", "")
                console.print(
                    f"[dim]Sub-agent [cyan]{sub.id[:8]}[/cyan] "
                    f"(parent {str(sub.parent_session_id)[:8]}): {task_desc}[/dim]"
                )
                if render_history(console, msgs) == 0:
                    console.print("[dim](no recorded messages)[/dim]")
                continue
            # List sub-agent sessions: for the current session, or all.
            parent_id = None if arg == "all" else (session.id if agent_session is not None else None)
            children = await sqlite.list_child_sessions(parent_id, limit=50)
            if not children:
                scope = "any session" if (arg == "all" or agent_session is None) else "this session"
                console.print(f"[dim]No sub-agent sessions for {scope}.[/dim]")
                continue
            console.print(
                "[dim]Sub-agent sessions"
                f"{' (all)' if arg == 'all' or agent_session is None else ''} "
                "— use [bold]/subagents <id>[/bold] to view one:[/dim]"
            )
            for s in children:
                task_desc = s.metadata.get("task", "")
                role = s.metadata.get("role", "")
                role_label = f"[cyan]{role}[/cyan] " if role else ""
                console.print(
                    f"  {s.id[:8]}  {s.status.value:<10} {role_label}"
                    f"{task_desc[:60]}"
                )
            continue
        if text.startswith("/switch"):
            arg = text[len("/switch") :].strip()
            if not arg:
                console.print("[yellow]Usage: /switch <session-id-prefix>[/yellow]")
                continue
            sessions = await sqlite.list_sessions(limit=50, top_level_only=True)
            matches = [s for s in sessions if s.id.startswith(arg)]
            if len(matches) == 0:
                console.print(f"[red]No session matching '{arg}'[/red]")
                continue
            if len(matches) > 1:
                console.print(
                    f"[yellow]Ambiguous — {len(matches)} sessions match '{arg}':[/yellow]"
                )
                for s in matches[:5]:
                    name = s.metadata.get("name", "")
                    label = f" ({name})" if name else ""
                    console.print(f"  {s.id[:8]}{label}")
                continue
            target = matches[0]
            if target.id == session.id:
                console.print("[dim]Already on that session.[/dim]")
                continue
            await _stop_agent_session()
            session = target
            if provider is not None:
                agent, agent_session = await _init_agent_session(session, provider)
            name = session.metadata.get("name", "")
            label = f" ({name})" if name else ""
            # Clear the screen, then restore the switched-to session's history.
            _show_history(clear=True)
            console.print(f"[green]Switched to session {session.id[:8]}{label}[/green]")
            continue
        if text.startswith("/rename"):
            new_name = text[len("/rename") :].strip()
            if not new_name:
                console.print("[yellow]Usage: /rename <name>[/yellow]")
                continue
            if agent_session is None:
                console.print(
                    "[yellow]No active session yet — send a message first.[/yellow]"
                )
                continue
            session.metadata["name"] = new_name
            await sqlite.update_session(session)
            console.print(f"[dim]Session {session.id[:8]} renamed to [cyan]{new_name}[/cyan][/dim]")
            continue
        if text.startswith("/delete"):
            arg = text[len("/delete") :].strip()
            if not arg and agent_session is None:
                console.print(
                    "[yellow]No active session to delete. Use "
                    "[bold]/delete <id>[/bold] to remove a stored session.[/yellow]"
                )
                continue
            if not arg:
                # Delete the current session (with confirmation)
                name = session.metadata.get("name", "")
                label = f" ({name})" if name else ""
                confirm = await asyncio.to_thread(
                    prompt_session.prompt,
                    HTML(
                        f"<prompt>Delete current session {session.id[:8]}{label}? [y/N] </prompt>"
                    ),
                )
                if confirm.strip().lower() not in ("y", "yes"):
                    console.print("[dim]Cancelled.[/dim]")
                    continue
                old_id = session.id
                # Stop agent and delete the current session's data.
                await _stop_agent_session()
                await sqlite.delete_session(old_id)
                await hot_store.delete_context(old_id)

                # Choose what to continue with. Never auto-create a session —
                # if none are left (or the user wants a new one) we go to the
                # pending state and wait for their first message.
                remaining = [
                    s
                    for s in await sqlite.list_sessions(limit=50, top_level_only=True)
                    if s.id != old_id
                ]
                if not remaining:
                    _go_pending()
                    console.print(
                        f"[dim]Deleted session {old_id[:8]}{label}. No sessions left "
                        f"— type a message to start a new one.[/dim]"
                    )
                    continue

                choice = await _pick_session_modal(remaining)
                if choice == "__new__":
                    _go_pending()
                    console.print(
                        f"[dim]Deleted session {old_id[:8]}{label}. "
                        f"Type a message to start a new session.[/dim]"
                    )
                    continue
                # A chosen id, or None (Esc) → fall back to the most recent.
                target = next((s for s in remaining if s.id == choice), remaining[0])
                session = target
                if provider is not None:
                    agent, agent_session = await _init_agent_session(session, provider)
                tname = session.metadata.get("name", "")
                tlabel = f" ({tname})" if tname else ""
                _show_history(clear=True)
                console.print(
                    f"[dim]Deleted session {old_id[:8]}{label}.[/dim] "
                    f"[green]Switched to {session.id[:8]}{tlabel}[/green]"
                )
                continue
            sessions = await sqlite.list_sessions(limit=50, top_level_only=True)
            matches = [s for s in sessions if s.id.startswith(arg)]
            if len(matches) == 0:
                console.print(f"[red]No session matching '{arg}'[/red]")
                continue
            if len(matches) > 1:
                console.print(
                    f"[yellow]Ambiguous — {len(matches)} sessions match '{arg}':[/yellow]"
                )
                for s in matches[:5]:
                    name = s.metadata.get("name", "")
                    label = f" ({name})" if name else ""
                    console.print(f"  {s.id[:8]}{label}")
                continue
            target = matches[0]
            if target.id == session.id:
                console.print(
                    "[yellow]That's the current session. "
                    "Run [bold]/delete[/bold] without arguments to delete it.[/yellow]"
                )
                continue
            await sqlite.delete_session(target.id)
            await hot_store.delete_context(target.id)
            name = target.metadata.get("name", "")
            label = f" ({name})" if name else ""
            console.print(f"[dim]Deleted session {target.id[:8]}{label}[/dim]")
            continue
        if text == "/detach":
            if agent_session is None or not agent_session.running:
                console.print("[dim]No running session to detach from.[/dim]")
            else:
                console.print(
                    f"[dim]Detached from session {session.id[:8]}. "
                    f"The agent continues in the background.\n"
                    f"Run leuk again to reattach, or /switch to this session.[/dim]"
                )
                # Don't stop the session — just exit the REPL
                # Drain the queue so events don't pile up without a consumer
                agent_session = None
                agent = None
            break
        if text == "/readonly":
            settings.safety.read_only = not settings.safety.read_only
            state = "[red]ON[/red]" if settings.safety.read_only else "[green]OFF[/green]"
            console.print(f"[dim]Read-only mode: {state}[/dim]")
            continue
        if text == "/safety":
            ro_state = "[red]ON[/red]" if settings.safety.read_only else "[green]OFF[/green]"
            sb_state = (
                "[green]container[/green]"
                if settings.sandbox.mode == "container"
                else "[dim]none[/dim]"
            )
            console.print(f"  Read-only mode:    {ro_state}")
            console.print(f"  Sandbox mode:      {sb_state}")
            console.print(f"  Project root:      {safety_guard.project_root}")
            deny_rules = [r for r in settings.safety.rules if r.action.value == "deny"]
            ask_rules = [r for r in settings.safety.rules if r.action.value == "ask"]
            allow_rules = [r for r in settings.safety.rules if r.action.value == "allow"]
            console.print(
                f"  Rules: {len(deny_rules)} deny, {len(ask_rules)} ask, {len(allow_rules)} allow"
            )
            continue
        if text == "/tasks":
            if task_scheduler is None:
                console.print(
                    "[yellow]Scheduler not enabled. "
                    "Set LEUK_SCHEDULER_ENABLED=true.[/yellow]"
                )
            else:
                tasks = await task_scheduler.store.list_tasks()
                if not tasks:
                    console.print("[dim]No scheduled tasks.[/dim]")
                for t in tasks:
                    state = "[green]enabled[/green]" if t.enabled else "[dim]disabled[/dim]"
                    next_str = t.next_run.strftime("%Y-%m-%d %H:%M") if t.next_run else "—"
                    last_str = t.last_run.strftime("%Y-%m-%d %H:%M") if t.last_run else "never"
                    console.print(
                        f"  [bold]{t.name}[/bold]  {t.schedule_type}:{t.schedule_expr}  "
                        f"{state}  next={next_str}  last={last_str}"
                    )
            continue
        if text.startswith("/policy"):
            from leuk.config import ReviewPolicy, save_persistent_config

            arg = text[len("/policy") :].strip()
            if not arg:
                console.print(
                    f"  Review policy: [cyan]{settings.safety.review_policy.value}[/cyan]\n"
                    f"  Options: {', '.join(p.value for p in ReviewPolicy)}"
                )
            else:
                try:
                    new_policy = ReviewPolicy(arg)
                except ValueError:
                    console.print(
                        f"[red]Unknown policy '{arg}'. "
                        f"Options: {', '.join(p.value for p in ReviewPolicy)}[/red]"
                    )
                    continue
                safety_guard.set_policy(new_policy)
                await safety_guard.load_persistent_approvals()
                save_persistent_config({"review_policy": new_policy.value})
                console.print(f"[dim]Review policy set to [cyan]{new_policy.value}[/cyan][/dim]")
            continue
        if text == "/desktop-auto":
            from leuk.config import save_persistent_config

            if not settings.input_control.enabled:
                console.print(
                    "[yellow]Desktop control is disabled. Enable it in "
                    "[bold]/settings[/bold] → General first.[/yellow]"
                )
                continue
            settings.input_control.auto_approve = not settings.input_control.auto_approve
            save_persistent_config(
                {"input_control_auto_approve": settings.input_control.auto_approve}
            )
            _sync_desktop_control_rule()
            await safety_guard.load_persistent_approvals()
            if settings.input_control.auto_approve:
                console.print(
                    "[bold red]⚠ Desktop auto-approve ON[/bold red] — the agent can "
                    "move the mouse, type, and trigger [bold]irreversible[/bold] desktop "
                    "actions WITHOUT asking. It self-verifies and should escalate risky "
                    "actions, but use with caution. Run [bold]/desktop-auto[/bold] again "
                    "to turn off."
                )
            else:
                console.print(
                    "[dim]Desktop auto-approve [bold]OFF[/bold] — desktop-control "
                    "actions require approval again.[/dim]"
                )
            continue
        if text.startswith("/approvals"):
            arg = text[len("/approvals") :].strip()
            if arg == "clear":
                count = await sqlite.clear_tool_approvals()
                safety_guard.set_policy(settings.safety.review_policy)
                console.print(f"[dim]Cleared {count} saved approval(s).[/dim]")
            else:
                approvals = await sqlite.list_tool_approvals()
                if not approvals:
                    console.print("[dim]No saved tool approvals.[/dim]")
                for a in approvals:
                    action_style = "green" if a["action"] == "allow" else "red"
                    console.print(
                        f"  [{action_style}]{a['action']}[/{action_style}]  "
                        f"{a['tool']}:{a['pattern']}  "
                        f"[dim](by {a['created_by'] or 'repl'}, {a['created_at'][:10]})[/dim]"
                    )
            continue
        if text == "/history":
            await _open_history_browser()
            continue
        if text.startswith("/file"):
            arg = text[len("/file"):].strip().strip("'\"")
            if not arg:
                console.print("[yellow]Usage: /file <path>[/yellow]")
                continue
            from leuk.media import load_media_file

            try:
                # Content type (image vs audio) is auto-detected from the file.
                part = load_media_file(arg)
            except (FileNotFoundError, ValueError) as exc:
                console.print(f"[red]{exc}[/red]")
                continue
            pending_attachments.append(part)
            kb = len(part.data) * 3 // 4 // 1024
            console.print(
                f"[dim]Attached {part.kind} ([cyan]{part.media_type}[/cyan], ~{kb} KB). "
                f"It will be sent with your next message "
                f"({len(pending_attachments)} pending).[/dim]"
            )
            continue
        if text == "/status":
            from datetime import datetime, timezone

            from leuk.agent.context import estimate_total_tokens

            _window = _ctx_cache["window"]
            if agent is not None:
                _msgs = len(agent._messages)
                _tokens = estimate_total_tokens(agent._messages)
            else:
                _msgs = 0
                _tokens = 0
            _elapsed = datetime.now(timezone.utc) - session.created_at
            _mins = int(_elapsed.total_seconds() // 60)
            _uptime = f"{_mins // 60}h {_mins % 60}m" if _mins >= 60 else f"{_mins}m"

            # "Context" = estimated tokens of the whole prompt we send the model
            # (system + conversation + tool results), vs. the model's window.
            _ctx_line = f"  Context:   ~{_tokens:,} est. prompt tokens"
            if _window:
                _pct = int(_tokens / _window * 100)
                _ctx_line += f" of {_window:,} window ({_pct}%)"
            else:
                _ctx_line += " (model window unknown — set LEUK_LLM_CONTEXT_WINDOW)"
            # The compaction budget (when leuk starts trimming) — derived from the
            # model window with headroom for the reply, or an explicit override.
            from leuk.agent.context import compaction_budget

            _budget = compaction_budget(
                _window,
                override=settings.agent.max_context_tokens,
                reserve=settings.llm.max_tokens,
            )
            if _budget != _window:
                _ctx_line += f"\n  Compact:   trims at ~{_budget:,} tokens"

            console.print(
                f"  Provider:  [cyan]{settings.llm.provider}[/cyan] / "
                f"[cyan]{settings.llm.model}[/cyan]\n"
                f"  Policy:    [cyan]{settings.safety.review_policy.value}[/cyan]\n"
                f"  Session:   [dim]{session.id[:8]}[/dim] "
                f"({session.status.value}, {_uptime})\n"
                f"  Messages:  {_msgs}\n"
                f"{_ctx_line}"
            )
            continue
        if text == "/doctor":
            from leuk.cli.doctor import render_report, build_report

            render_report(console, build_report(settings))
            continue
        if text == "/skills" or text.startswith("/skills "):
            from leuk.cli.extensions_manager import run_skills_manager

            await asyncio.to_thread(run_skills_manager, settings)
            if not settings.skills.enabled:
                console.print(
                    "[yellow]Skills are off — enable them in /settings to expose installed "
                    "skills to the model.[/yellow]"
                )
            console.print("[dim]Skill changes apply on the next session/restart.[/dim]")
            continue
        if text == "/mcp" or text.startswith("/mcp "):
            from leuk.cli.extensions_manager import run_mcp_manager

            await asyncio.to_thread(run_mcp_manager, settings, dict(mcp_status))
            console.print("[dim]Connector changes apply on the next restart.[/dim]")
            continue
        if text == "/speak":
            from leuk.voice import VOICE_AVAILABLE, _MISSING_REASON

            if not VOICE_AVAILABLE:
                console.print(f"[red]{_MISSING_REASON}[/red]")
                continue
            speak_mode = not speak_mode
            if speak_mode and tts_backend is None:
                from leuk.config import load_persistent_config as _load_pc
                from leuk.voice.tts import create_tts_backend

                pc = _load_pc()
                tts_backend = create_tts_backend(
                    pc.get("tts_backend", "local"),
                    voice=pc.get("tts_voice", "alloy"),
                    api_key=settings.llm.openai_api_key or None,
                    speaker=pc.get("tts_speaker"),
                    en_speaker=pc.get("tts_en_speaker"),
                    language=pc.get("tts_language"),
                )
            state = "[green]ON[/green]" if speak_mode else "[dim]OFF[/dim]"
            console.print(f"[dim]Text-to-speech: {state}[/dim]")
            continue
        if text == "/voice":
            from leuk.voice import VOICE_AVAILABLE, _MISSING_REASON

            if not VOICE_AVAILABLE:
                console.print(f"[red]{_MISSING_REASON}[/red]")
                continue
            voice_mode = not voice_mode
            if voice_mode:
                # ── Start voice input ────────────────────────────
                if voice_stt is None:
                    from leuk.config import load_persistent_config as _load_pc
                    from leuk.voice.recorder import ContinuousVAD, MicRecorder
                    from leuk.voice.stt import create_stt_backend

                    pc = _load_pc()
                    voice_stt = create_stt_backend(
                        pc.get("stt_backend", "local"),
                        model_size=pc.get("stt_model_size", "turbo"),
                        language=pc.get("stt_language"),
                        api_key=settings.llm.openai_api_key or None,
                    )
                    _mic_dev = pc.get("audio_input_device")
                    voice_recorder = MicRecorder(device=_mic_dev)

                    # Eagerly load the STT model
                    console.print("[dim]Loading STT model …[/dim]")
                    await _warmup_stt(voice_stt)

                    # Set up ContinuousVAD → transcription bridge. Voice input is
                    # half-duplex: the VAD is paused while the agent speaks (see
                    # _run_agent_turn) to avoid a speaker→mic feedback loop.
                    vad_sensitivity = float(pc.get("vad_sensitivity", 0.5))
                    silence_timeout = float(pc.get("vad_silence_timeout", 1.0))
                    min_speech = float(pc.get("vad_min_speech", 0.5))
                    voice_bridge = _VoiceInputBridge(voice_stt, console)
                    voice_vad = ContinuousVAD(
                        voice_recorder,
                        on_speech=voice_bridge.on_speech,
                        sensitivity=vad_sensitivity,
                        silence_timeout=silence_timeout,
                        min_duration=min_speech,
                    )
                    # Give the bridge a reference to the VAD for trim_silence
                    voice_bridge._vad = voice_vad

                # Start listening
                voice_vad.start()  # type: ignore[union-attr]
                console.print("[dim]Voice input: [green]ON[/green] (hands-free)[/dim]")

                # Auto-enable TTS
                if not speak_mode:
                    speak_mode = True
                    if tts_backend is None:
                        from leuk.config import load_persistent_config as _load_pc2
                        from leuk.voice.tts import create_tts_backend

                        pc2 = _load_pc2()
                        tts_backend = create_tts_backend(
                            pc2.get("tts_backend", "local"),
                            voice=pc2.get("tts_voice", "alloy"),
                            api_key=settings.llm.openai_api_key or None,
                            speaker=pc2.get("tts_speaker"),
                            en_speaker=pc2.get("tts_en_speaker"),
                            language=pc2.get("tts_language"),
                        )
                    console.print("[dim]Text-to-speech: [green]ON[/green] (auto)[/dim]")
            else:
                # ── Stop voice input ─────────────────────────────
                if voice_vad is not None:
                    await voice_vad.stop()  # type: ignore[union-attr]
                if speak_mode:
                    speak_mode = False
                    console.print("[dim]Text-to-speech: [dim]OFF[/dim] (auto)[/dim]")
                console.print("[dim]Voice input: [dim]OFF[/dim][/dim]")
            continue
        if text in ("/settings", "/voice-settings"):
            from leuk.cli.settings_dialog import run_settings
            from leuk.config import load_persistent_config as _load_pc_vs
            from leuk.config import save_persistent_config as _save_pc_vs

            cur_pc = _load_pc_vs()
            result = await asyncio.to_thread(run_settings, cur_pc)
            if result is not None:
                _save_pc_vs(result)
                # Apply a theme change live: re-colour the rich console and the
                # prompt's style/footer/completion menu immediately.
                if "theme" in result and result["theme"] != _theme.ACTIVE_THEME:
                    _theme.apply_theme(result["theme"])
                    console.push_theme(_theme.LEUK_THEME)
                    prompt_session.style = _build_pt_style(_theme.PALETTE)
                    console.print(
                        f"[dim]Theme set to [accent.cyan]"
                        f"{_theme.THEMES[_theme.ACTIVE_THEME]['label']}[/accent.cyan]."
                        f"[/dim]"
                    )
                # Force re-creation of backends on next /voice or /speak
                if voice_vad is not None:
                    await voice_vad.stop()
                    voice_vad = None
                    voice_bridge = None
                if voice_stt is not None:
                    await voice_stt.close()
                    voice_stt = None
                if tts_backend is not None:
                    await tts_backend.close()
                    tts_backend = None
                voice_recorder = None
                console.print("[dim]Settings saved. Backends will reload on next use.[/dim]")

                # Show summary
                pc = _load_pc_vs()
                stt_m = pc.get("stt_model_size", "turbo")
                lang = pc.get("tts_language") or pc.get("stt_language") or "(auto)"
                speaker = pc.get("tts_speaker") or "(default)"
                vad_s = pc.get("vad_sensitivity", "0.5")
                console.print(
                    f"  STT: [cyan]{stt_m}[/cyan]  Lang: [cyan]{lang}[/cyan]  "
                    f"Speaker: [cyan]{speaker}[/cyan]  VAD: [cyan]{vad_s}[/cyan]"
                )
            else:
                console.print("[dim]No changes.[/dim]")
            continue
        if text == "/models":
            from leuk.cli.models import run_model_selector
            from leuk.config import load_credentials
            from leuk.providers.catalog import fetch_all_available

            creds = load_credentials()

            console.print("[dim]Fetching available models...[/dim]")
            provider_models = await fetch_all_available(settings.llm, creds)

            if not provider_models:
                console.print(
                    "[yellow]No models available. Run /auth to configure a provider.[/yellow]"
                )
                continue

            selection = await asyncio.to_thread(
                run_model_selector,
                settings.llm.provider,
                settings.llm.model,
                provider_models,
            )

            if selection is not None:
                new_provider_key, new_model = selection

                if new_model != settings.llm.model or new_provider_key != settings.llm.provider:
                    settings.llm.model = new_model

                    if new_provider_key != settings.llm.provider:
                        # Switching provider too
                        settings.llm.provider = new_provider_key
                        try:
                            old_provider = provider
                            provider = create_provider(settings.llm)
                            if old_provider is not None:
                                await old_provider.close()
                            await _stop_agent_session()
                            agent, agent_session = await _init_agent_session(session, provider)
                        except NoCredentialsError:
                            provider = None
                            agent = None
                            agent_session = None
                            console.print(
                                f"[yellow]No credentials for '{new_provider_key}'. "
                                f"Run /auth to configure.[/yellow]"
                            )
                            continue
                    elif provider is not None:
                        # Same provider, just rebuild with new model
                        old_provider = provider
                        provider = create_provider(settings.llm)
                        await old_provider.close()
                        await _stop_agent_session()
                        agent, agent_session = await _init_agent_session(session, provider)

                    # Persist selection so it's restored on next launch.
                    from leuk.config import save_persistent_config

                    save_persistent_config(
                        {"last_provider": settings.llm.provider, "last_model": settings.llm.model}
                    )

                    console.print(
                        f"[dim]Model: {settings.llm.model} ({settings.llm.provider})[/dim]"
                    )
            else:
                console.print("[dim]Cancelled.[/dim]")
            continue

        if text == "/auth":
            from leuk.cli.auth import run_auth

            new_provider_key = await asyncio.to_thread(run_auth, settings.llm.provider)

            # Reload credentials regardless (user may have added/edited)
            settings = load_settings()

            if new_provider_key and new_provider_key != settings.llm.provider:
                settings.llm.provider = new_provider_key

            # Try to create the provider with updated settings
            try:
                old_provider = provider
                provider = create_provider(settings.llm)
                if old_provider is not None:
                    await old_provider.close()

                # Rebuild the agent with the new provider
                await _stop_agent_session()
                agent, agent_session = await _init_agent_session(session, provider)

                console.print(f"[dim]Provider: {settings.llm.provider} — ready.[/dim]")

                # Show the model selector immediately after switching providers
                from leuk.cli.models import run_model_selector
                from leuk.config import load_credentials as _load_creds_auth
                from leuk.providers.catalog import fetch_all_available

                creds = _load_creds_auth()
                console.print("[dim]Fetching available models...[/dim]")
                provider_models = await fetch_all_available(settings.llm, creds)
                if provider_models:
                    selection = await asyncio.to_thread(
                        run_model_selector,
                        settings.llm.provider,
                        settings.llm.model,
                        provider_models,
                    )
                    if selection is not None:
                        new_provider_key, new_model = selection
                        if new_model != settings.llm.model or new_provider_key != settings.llm.provider:
                            settings.llm.model = new_model
                            if new_provider_key != settings.llm.provider:
                                settings.llm.provider = new_provider_key
                            old_provider = provider
                            provider = create_provider(settings.llm)
                            if old_provider is not None:
                                await old_provider.close()
                            await _stop_agent_session()
                            agent, agent_session = await _init_agent_session(session, provider)

                            from leuk.config import save_persistent_config as _save_pc_auth

                            _save_pc_auth(
                                {"last_provider": settings.llm.provider, "last_model": settings.llm.model}
                            )
                            console.print(
                                f"[dim]Model: {settings.llm.model} ({settings.llm.provider})[/dim]"
                            )
                    else:
                        console.print("[dim]Keeping current model.[/dim]")
            except NoCredentialsError:
                provider = None
                agent = None
                agent_session = None
                console.print(
                    f"[yellow]No credentials for '{settings.llm.provider}'. "
                    f"Run /auth to configure.[/yellow]"
                )
            continue

        # /retry — re-send the last user message (typically after an error)
        if text == "/retry":
            last = agent_session.last_user_input if agent_session is not None else None
            if not last:
                console.print("[yellow]Nothing to retry — no previous message.[/yellow]")
                continue
            console.print(f"[dim]Retrying: {last[:80]}{'…' if len(last) > 80 else ''}[/dim]")
            text = last
            # fall through to the agent dispatch below

        # Guard: refuse to run without a provider
        if provider is None:
            console.print("[yellow]No provider configured. Run /auth to set up.[/yellow]")
            continue

        # Lazily materialise the session on the first message: this is the
        # point a draft session is actually persisted and the agent started.
        if agent_session is None:
            agent, agent_session = await _init_agent_session(session, provider)

        # Attach any staged images/audio to this turn's user message.
        if pending_attachments and agent is not None:
            agent.pending_attachments = list(pending_attachments)
            pending_attachments.clear()

        # Auto-name the session from its first real message (background).
        _maybe_name_session(text)

        # Pause VAD during agent turn to prevent feedback loops.
        # If speak_mode is active, _run_agent_turn handles the TTS-specific
        # VAD pause/resume itself so speech and VAD don't overlap.
        vad_paused_here = False
        if voice_vad is not None and not speak_mode:
            voice_vad.pause()
            vad_paused_here = True

        # Run the agent turn via the AgentSession, with graceful Ctrl-C:
        # during a turn the terminal is in cooked mode, so Ctrl-C raises
        # SIGINT. We catch it to cancel the turn — which interrupts the agent
        # *and* stops TTS playback at once (see _run_agent_turn) — and return to
        # the prompt, instead of letting KeyboardInterrupt tear down the REPL.
        turn_task = asyncio.ensure_future(
            _run_agent_turn(
                agent_session,
                text,
                stream_renderer,
                speak_mode=speak_mode,
                tts_backend=tts_backend,
                voice_vad=voice_vad,
            )
        )
        loop = asyncio.get_running_loop()

        def _on_sigint() -> None:
            # A single Ctrl-C cancels the turn: generation is interrupted and any
            # TTS playback halts immediately, then we drop back to the prompt.
            if not turn_task.done():
                turn_task.cancel()

        sigint_installed = False
        try:
            loop.add_signal_handler(signal.SIGINT, _on_sigint)
            sigint_installed = True
        except (NotImplementedError, RuntimeError):
            # No signal support (e.g. Windows) — fall back to default
            # KeyboardInterrupt propagation.
            pass

        try:
            await turn_task
        except asyncio.CancelledError:
            console.print("[dim]\nInterrupted.[/dim]")
        except Exception:
            console.print_exception()
        finally:
            if sigint_installed:
                try:
                    loop.remove_signal_handler(signal.SIGINT)
                except (NotImplementedError, ValueError, RuntimeError):
                    pass
                # remove_signal_handler resets SIGINT to SIG_DFL, which would
                # terminate the process. Restore Python's handler so the next
                # prompt's Ctrl-C → KeyboardInterrupt behaviour still works.
                try:
                    signal.signal(signal.SIGINT, signal.default_int_handler)
                except (ValueError, RuntimeError):
                    pass
            if vad_paused_here and voice_vad is not None:
                voice_vad.resume()

    # ── Shutdown ──────────────────────────────────────────────────
    if task_scheduler is not None:
        await task_scheduler.stop()
    if channel_registry is not None:
        await channel_registry.stop()
    await _stop_agent_session()
    if mcp_connect_task is not None and not mcp_connect_task.done():
        mcp_connect_task.cancel()
        try:
            await mcp_connect_task
        except (Exception, asyncio.CancelledError):  # noqa: BLE001 — best-effort
            pass
    for mc in mcp_clients:
        await mc.close()
    if provider is not None:
        await provider.close()
    await hot_store.close()
    await sqlite.close()
    if voice_vad is not None:
        await voice_vad.stop()
    if tts_backend is not None:
        await tts_backend.close()
    if voice_stt is not None:
        await voice_stt.close()
    console.print("[dim]Goodbye.[/dim]")


def main() -> None:
    """Entry point for `leuk` CLI command."""
    import argparse

    parser = argparse.ArgumentParser(prog="leuk", description="leuk AI agent")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG). -vv also shows suppressed warnings from ML/network libraries (aiogram, httpx, …).",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["doctor", "skills", "mcp"],
        help=(
            "doctor: check optional-feature setup; skills: manage agent skills; "
            "mcp: manage MCP connectors. With no command, start the REPL."
        ),
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments for 'skills'/'mcp' subcommands (e.g. `leuk mcp list`).",
    )
    args = parser.parse_args()

    if args.command == "doctor":
        from leuk.cli.doctor import run_doctor

        raise SystemExit(run_doctor())
    if args.command == "skills":
        from leuk.cli.extensions_manager import run_skills_cli

        raise SystemExit(run_skills_cli(args.args))
    if args.command == "mcp":
        from leuk.cli.extensions_manager import run_mcp_cli

        raise SystemExit(run_mcp_cli(args.args))

    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Quiet noisy third-party loggers unless the user explicitly asked for
    # debug output. Two mechanisms:
    #   * Raise generic chatter (INFO/WARNING) to ERROR via the logger level.
    #   * Drop aiogram's *transient long-polling reconnect* messages — which
    #     are logged at ERROR ("Failed to fetch updates …") followed by a
    #     WARNING retry ("Sleep for N seconds and try again …") — via a
    #     content filter, since aiogram recovers from them automatically.
    #     Genuine, non-transient errors still surface.
    if log_level > logging.DEBUG:
        for noisy in ("aiogram", "aiohttp", "httpx", "httpcore", "asyncio"):
            logging.getLogger(noisy).setLevel(logging.ERROR)

        noise_filter = _TransientPollingNoiseFilter()
        for handler in logging.getLogger().handlers:
            handler.addFilter(noise_filter)

    try:
        asyncio.run(_run_repl())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
