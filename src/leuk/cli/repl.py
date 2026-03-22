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

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from leuk.agent.session import AgentSession
from leuk.agent.sub_agent import SubAgentManager
from leuk.config import Settings, config_dir, load_settings
from leuk.safety import SafetyGuard
from leuk.persistence import create_hot_store
from leuk.persistence.base import HotStore
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


logger = logging.getLogger(__name__)

console = Console()

STYLE = Style.from_dict(
    {
        "prompt": "#00aa00 bold",
        "input": "#ffffff",
    }
)


async def _resume_or_create_session(
    sqlite: SQLiteStore, hot_store: HotStore, settings: Settings
) -> Session:
    """Try to resume the last active session, or create a new one."""
    active_id = await hot_store.get_active_session()
    if active_id:
        session = await sqlite.get_session(active_id)
        if session and session.status.value in ("active", "paused"):
            console.print(f"[dim]Resuming session {session.id[:8]}...[/dim]")
            return session

    # Check if there's a recent session in SQLite
    sessions = await sqlite.list_sessions(limit=1)
    if sessions and sessions[0].status.value in ("active", "paused"):
        session = sessions[0]
        console.print(f"[dim]Resuming session {session.id[:8]}...[/dim]")
        return session

    # Create fresh
    return Session(system_prompt=settings.agent.system_prompt)


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

        if text and text.strip():
            self.console.print(f"[cyan]🎤 {text.strip()}[/cyan]")
            await self.text_queue.put(text.strip())
        else:
            logger.debug("Empty transcription, ignoring")


# ── Agent turn (push + render) ───────────────────────────────────


async def _run_agent_turn(
    agent_session: AgentSession,
    text: str,
    renderer: StreamRenderer,
    *,
    speak_mode: bool = False,
    tts_backend: object | None = None,
) -> None:
    """Push a user message to the agent session and render the response.

    This blocks until the agent turn completes (TURN_COMPLETE event).
    While rendering, KeyboardInterrupt triggers an interrupt.
    """
    agent_session.push(text)

    # Render events until the turn finishes
    try:
        await renderer.render_queue(agent_session.event_queue)
    except asyncio.CancelledError:
        agent_session.interrupt()
        raise

    # TTS: speak the assistant's streamed text
    if speak_mode and tts_backend is not None and renderer._text_buffer:
        from leuk.voice.tts import clean_text_for_speech

        raw_text = "".join(renderer._text_buffer)
        spoken_text = clean_text_for_speech(raw_text)
        logger.debug("TTS input (cleaned): %r", spoken_text)
        if spoken_text.strip():
            try:
                await tts_backend.speak(spoken_text)  # type: ignore[union-attr]
            except Exception as tts_exc:
                logger.debug("TTS exception", exc_info=tts_exc)
                console.print(f"[red dim]TTS error: {type(tts_exc).__name__}: {tts_exc}[/red dim]")


async def _run_repl() -> None:
    """Main REPL loop."""
    settings = load_settings()

    # Initialise backends
    sqlite = SQLiteStore(settings.sqlite)
    await sqlite.init()
    hot_store = create_hot_store()

    # Safety guardrails
    async def _confirm_tool_use(reason: str, tool_call) -> bool:
        """Prompt the user for permission during agent execution."""

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
            HTML("<prompt>Allow? [y/N]: </prompt>"),
        )
        return response.strip().lower() in ("y", "yes")

    prompt_session_for_confirm: PromptSession[str] = PromptSession()
    safety_guard = SafetyGuard(
        settings.safety,
        confirm_callback=_confirm_tool_use,
    )

    verbose_mode = False
    stream_renderer = StreamRenderer(console, verbose=verbose_mode)

    voice_mode = False
    voice_stt = None  # Lazy-initialized STT backend
    voice_recorder = None  # Lazy-initialized MicRecorder
    voice_vad = None  # ContinuousVAD instance
    voice_bridge = None  # _VoiceInputBridge instance

    speak_mode = False
    tts_backend = None  # Lazy-initialized TTS backend

    provider = None  # may be None until credentials are configured
    try:
        provider = create_provider(settings.llm)
    except NoCredentialsError:
        console.print(
            "[yellow]No credentials configured for "
            f"'{settings.llm.provider}'. Run /auth to set up.[/yellow]"
        )

    tools = create_default_registry()

    # Connect to MCP servers
    mcp_clients: list["MCPClient"] = []
    if settings.mcp_servers:
        from leuk.mcp.client import MCPClient
        from leuk.mcp.bridge import MCPToolBridge

        for srv_cfg in settings.mcp_servers:
            try:
                if srv_cfg.transport == "stdio" and srv_cfg.command:
                    client = MCPClient.stdio(srv_cfg.command, srv_cfg.args, name=srv_cfg.name)
                elif srv_cfg.transport == "sse" and srv_cfg.url:
                    client = MCPClient.sse(srv_cfg.url, name=srv_cfg.name)
                else:
                    console.print(
                        f"[yellow]Skipping MCP server {srv_cfg.name}: invalid config[/yellow]"
                    )
                    continue
                await client.connect()
                bridge = MCPToolBridge(client)
                bridge.register_tools(tools)
                mcp_clients.append(client)
                console.print(f"[dim]MCP: connected to {client.name}[/dim]")
            except Exception as exc:
                console.print(f"[red]MCP: failed to connect to {srv_cfg.name}: {exc}[/red]")

    session = await _resume_or_create_session(sqlite, hot_store, settings)

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
        return ag, asess

    async def _stop_agent_session() -> None:
        """Stop the current agent session if running."""
        nonlocal agent, agent_session
        if agent_session is not None:
            await agent_session.stop()
            agent_session = None
        if agent is not None:
            await agent.shutdown()
            agent = None

    if provider is not None:
        agent, agent_session = await _init_agent_session(session, provider)

    def _provider_label() -> str:
        if provider is None:
            return "[red]none (run /auth)[/red]"
        return f"[cyan]{settings.llm.provider}[/cyan]"

    _sess_name = session.metadata.get("name", "")
    _sess_label = f"[dim]{session.id[:8]}[/dim]"
    if _sess_name:
        _sess_label += f" ([cyan]{_sess_name}[/cyan])"
    console.print(
        Panel(
            f"[bold]leuk[/bold] v0.1.0\n"
            f"Provider: {_provider_label()} / "
            f"Model: [cyan]{settings.llm.model}[/cyan]\n"
            f"Session: {_sess_label}\n\n"
            f"Type [bold]/help[/bold] for commands",
            border_style="bright_blue",
        )
    )

    history_path = config_dir() / "repl_history"
    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path)),
        style=STYLE,
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
                    except (asyncio.CancelledError, EOFError):
                        pass
                winner = done.pop()
                user_input = winner.result()
            else:
                user_input = await asyncio.to_thread(
                    prompt_session.prompt,
                    HTML("<prompt>leuk> </prompt>"),
                )
        except (EOFError, KeyboardInterrupt):
            break

        text = user_input.strip()
        if not text:
            continue

        # Handle slash commands
        if text == "/quit":
            break
        if text == "/help":
            console.print(
                Panel(
                    "[bold]/help[/bold]              — Show this help message\n"
                    "[bold]/models[/bold]            — Select model\n"
                    "[bold]/new[/bold]               — Start a new session\n"
                    "[bold]/sessions[/bold]          — List recent sessions\n"
                    "[bold]/switch[/bold] [dim]<id>[/dim]       — Switch to session by id prefix\n"
                    "[bold]/rename[/bold] [dim]<name>[/dim]     — Rename current session\n"
                    "[bold]/delete[/bold]             — Delete current session (with confirmation)\n"
                    "[bold]/delete[/bold] [dim]<id>[/dim]       — Delete another session by id prefix\n"
                    "[bold]/detach[/bold]            — Detach from session (keeps running)\n"
                    "[bold]/auth[/bold]              — Select provider / manage credentials\n"
                    "[bold]/sandbox[/bold]           — Toggle read-only sandbox mode\n"
                    "[bold]/safety[/bold]            — Show safety guardrail status\n"
                    "[bold]/verbose[/bold]           — Toggle verbose tool output\n"
                    "[bold]/voice[/bold]             — Toggle voice input\n"
                    "[bold]/speak[/bold]             — Toggle text-to-speech output\n"
                    "[bold]/voice-settings[/bold]    — Configure STT/TTS model, language, speaker\n"
                    "[bold]/quit[/bold]              — Exit leuk",
                    title="[bold]Commands[/bold]",
                    border_style="bright_blue",
                    expand=False,
                )
            )
            continue
        if text == "/new":
            await _stop_agent_session()
            session = Session(system_prompt=settings.agent.system_prompt)
            if provider is not None:
                agent, agent_session = await _init_agent_session(session, provider)
            console.print(f"[green]New session: {session.id[:8]}[/green]")
            continue
        if text == "/sessions":
            sessions = await sqlite.list_sessions(limit=10)
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
        if text.startswith("/switch"):
            arg = text[len("/switch") :].strip()
            if not arg:
                console.print("[yellow]Usage: /switch <session-id-prefix>[/yellow]")
                continue
            sessions = await sqlite.list_sessions(limit=50)
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
            console.print(f"[green]Switched to session {session.id[:8]}{label}[/green]")
            continue
        if text.startswith("/rename"):
            new_name = text[len("/rename") :].strip()
            if not new_name:
                console.print("[yellow]Usage: /rename <name>[/yellow]")
                continue
            session.metadata["name"] = new_name
            await sqlite.update_session(session)
            console.print(f"[dim]Session {session.id[:8]} renamed to [cyan]{new_name}[/cyan][/dim]")
            continue
        if text.startswith("/delete"):
            arg = text[len("/delete") :].strip()
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
                # Stop agent, delete session data, start fresh
                await _stop_agent_session()
                await sqlite.delete_session(old_id)
                await hot_store.delete_context(old_id)
                session = Session(system_prompt=settings.agent.system_prompt)
                if provider is not None:
                    agent, agent_session = await _init_agent_session(session, provider)
                console.print(
                    f"[dim]Deleted session {old_id[:8]}{label}.[/dim] "
                    f"[green]New session: {session.id[:8]}[/green]"
                )
                continue
            sessions = await sqlite.list_sessions(limit=50)
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
        if text == "/sandbox":
            settings.safety.read_only = not settings.safety.read_only
            state = "[red]ON[/red]" if settings.safety.read_only else "[green]OFF[/green]"
            console.print(f"[dim]Read-only sandbox: {state}[/dim]")
            continue
        if text == "/safety":
            state = "[red]ON[/red]" if settings.safety.read_only else "[green]OFF[/green]"
            console.print(f"  Read-only sandbox: {state}")
            console.print(f"  Project root: {safety_guard.project_root}")
            deny_rules = [r for r in settings.safety.rules if r.action.value == "deny"]
            ask_rules = [r for r in settings.safety.rules if r.action.value == "ask"]
            allow_rules = [r for r in settings.safety.rules if r.action.value == "allow"]
            console.print(
                f"  Rules: {len(deny_rules)} deny, {len(ask_rules)} ask, {len(allow_rules)} allow"
            )
            continue
        if text == "/verbose":
            verbose_mode = not verbose_mode
            stream_renderer.verbose = verbose_mode
            state = "[green]ON[/green]" if verbose_mode else "[dim]OFF[/dim]"
            console.print(f"[dim]Verbose tool output: {state}[/dim]")
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
                    voice_recorder = MicRecorder(device=pc.get("audio_input_device"))

                    # Eagerly load the STT model
                    console.print("[dim]Loading STT model …[/dim]")
                    await _warmup_stt(voice_stt)

                    # Set up ContinuousVAD → transcription bridge
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
        if text == "/voice-settings":
            from leuk.cli.voice_settings import run_voice_settings
            from leuk.config import load_persistent_config as _load_pc_vs
            from leuk.config import save_persistent_config as _save_pc_vs

            cur_pc = _load_pc_vs()
            result = await asyncio.to_thread(run_voice_settings, cur_pc)
            if result is not None:
                _save_pc_vs(result)
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
                console.print("[dim]Voice settings saved. Backends will reload on next use.[/dim]")

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
                console.print("[dim]Cancelled.[/dim]")
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

        # Guard: refuse to run without a provider
        if provider is None or agent_session is None:
            console.print("[yellow]No provider configured. Run /auth to set up.[/yellow]")
            continue

        # Pause VAD during agent turn + TTS to prevent feedback loops
        if voice_vad is not None:
            voice_vad.pause()

        # Run agent turn via the AgentSession
        try:
            await _run_agent_turn(
                agent_session,
                text,
                stream_renderer,
                speak_mode=speak_mode,
                tts_backend=tts_backend,
            )
        except asyncio.CancelledError:
            console.print("[dim]\nInterrupted.[/dim]")
        except Exception:
            console.print_exception()
        finally:
            # Resume VAD after agent turn + TTS complete
            if voice_vad is not None:
                voice_vad.resume()

    # ── Shutdown ──────────────────────────────────────────────────
    await _stop_agent_session()
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
        help="Increase verbosity (-v for INFO, -vv for DEBUG). Shows suppressed warnings from ML libraries.",
    )
    args = parser.parse_args()

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

    try:
        asyncio.run(_run_repl())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
