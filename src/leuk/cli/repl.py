"""Interactive REPL for the leuk agent."""

from __future__ import annotations

import asyncio
import logging
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from leuk.agent.sub_agent import SubAgentManager
from leuk.config import PermissionAction, Settings, config_dir, load_settings
from leuk.safety import SafetyGuard
from leuk.persistence import create_hot_store
from leuk.persistence.base import HotStore
from leuk.persistence.sqlite import SQLiteStore
from leuk.providers import create_provider
from leuk.providers.base import LLMProvider, NoCredentialsError
from leuk.tools import create_default_registry
from leuk.tools.sub_agent import SubAgentTool
from leuk.types import Message, Role, Session, StreamEvent, StreamEventType
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


async def _run_agent_streaming(
    agent: "Agent", text: str, *, renderer: StreamRenderer
) -> None:
    """Run the agent with streaming output via the StreamRenderer."""
    from leuk.agent.core import Agent

    await renderer.render_stream(agent.run_stream(text))


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
        from leuk.types import ToolCall as _TC
        args_str = ", ".join(f"{k}={v!r}" for k, v in tool_call.arguments.items())
        console.print(Panel(
            f"[bold]{tool_call.name}[/bold]({args_str})\n\n"
            f"[yellow]{reason}[/yellow]",
            title="[red]Permission Required[/red]",
            border_style="red",
            expand=False,
        ))
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
                    console.print(f"[yellow]Skipping MCP server {srv_cfg.name}: invalid config[/yellow]")
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
    if provider is not None:
        agent = _make_agent(session, provider)
        await agent.init()

    def _provider_label() -> str:
        if provider is None:
            return "[red]none (run /auth)[/red]"
        return f"[cyan]{settings.llm.provider}[/cyan]"

    console.print(
        Panel(
            f"[bold]leuk[/bold] v0.1.0\n"
            f"Provider: {_provider_label()} / "
            f"Model: [cyan]{settings.llm.model}[/cyan]\n"
            f"Session: [dim]{session.id[:8]}[/dim]\n\n"
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
        try:
            user_input = await asyncio.to_thread(
                prompt_session.prompt,
                HTML("<prompt>leuk> </prompt>"),
            )
        except (EOFError, KeyboardInterrupt):
            break

        text = user_input.strip()
        if not text and not voice_mode:
            continue

        # Handle slash commands
        if text == "/quit":
            break
        if text == "/help":
            console.print(
                Panel(
                    "[bold]/help[/bold]       — Show this help message\n"
                    "[bold]/models[/bold]     — Select model\n"
                    "[bold]/new[/bold]        — Start a new session\n"
                    "[bold]/sessions[/bold]   — List recent sessions\n"
                    "[bold]/auth[/bold]       — Select provider / manage credentials\n"
                    "[bold]/sandbox[/bold]    — Toggle read-only sandbox mode\n"
                    "[bold]/safety[/bold]     — Show safety guardrail status\n"
                    "[bold]/verbose[/bold]    — Toggle verbose tool output\n"
                    "[bold]/voice[/bold]      — Toggle voice input (push-to-talk)\n"
                    "[bold]/speak[/bold]      — Toggle text-to-speech output\n"
                    "[bold]/quit[/bold]       — Exit leuk",
                    title="[bold]Commands[/bold]",
                    border_style="bright_blue",
                    expand=False,
                )
            )
            continue
        if text == "/new":
            if agent is not None:
                await agent.shutdown()
            session = Session(system_prompt=settings.agent.system_prompt)
            if provider is not None:
                agent = _make_agent(session, provider)
                await agent.init()
            console.print(f"[green]New session: {session.id[:8]}[/green]")
            continue
        if text == "/sessions":
            sessions = await sqlite.list_sessions(limit=10)
            current_id = agent.session.id if agent else ""
            for s in sessions:
                marker = " [bold green]*[/bold green]" if s.id == current_id else ""
                console.print(
                    f"  {s.id[:8]}  {s.status.value:<10} "
                    f"updated {s.updated_at.strftime('%Y-%m-%d %H:%M')}{marker}"
                )
            continue
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
            console.print(f"  Rules: {len(deny_rules)} deny, {len(ask_rules)} ask, {len(allow_rules)} allow")
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
                from leuk.voice.tts import create_tts_backend
                tts_backend = create_tts_backend("local")
            state = "[green]ON[/green]" if speak_mode else "[dim]OFF[/dim]"
            console.print(f"[dim]Text-to-speech: {state}[/dim]")
            continue
        if text == "/voice":
            from leuk.voice import VOICE_AVAILABLE, _MISSING_REASON
            if not VOICE_AVAILABLE:
                console.print(f"[red]{_MISSING_REASON}[/red]")
                continue
            voice_mode = not voice_mode
            if voice_mode and voice_stt is None:
                from leuk.voice.stt import create_stt_backend
                from leuk.voice.recorder import MicRecorder
                voice_stt = create_stt_backend("local")
                voice_recorder = MicRecorder()
            state = "[green]ON[/green]" if voice_mode else "[dim]OFF[/dim]"
            console.print(f"[dim]Voice input: {state}[/dim]")
            if voice_mode:
                console.print("[dim]Press Enter to start recording, Enter again to stop[/dim]")
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
                            if agent is not None:
                                await agent.shutdown()
                            agent = _make_agent(session, provider)
                            await agent.init()
                        except NoCredentialsError:
                            provider = None
                            agent = None
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
                        if agent is not None:
                            await agent.shutdown()
                        agent = _make_agent(session, provider)
                        await agent.init()

                    console.print(
                        f"[dim]Model: {settings.llm.model} "
                        f"({settings.llm.provider})[/dim]"
                    )
            else:
                console.print("[dim]Cancelled.[/dim]")
            continue

        if text == "/auth":
            from leuk.cli.auth import run_auth

            new_provider_key = await asyncio.to_thread(
                run_auth, settings.llm.provider
            )

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
                if agent is not None:
                    await agent.shutdown()
                agent = _make_agent(session, provider)
                await agent.init()

                console.print(
                    f"[dim]Provider: {settings.llm.provider} — ready.[/dim]"
                )
            except NoCredentialsError:
                provider = None
                agent = None
                console.print(
                    f"[yellow]No credentials for '{settings.llm.provider}'. "
                    f"Run /auth to configure.[/yellow]"
                )
            continue

        # Guard: refuse to run without a provider
        if provider is None or agent is None:
            console.print(
                "[yellow]No provider configured. Run /auth to set up.[/yellow]"
            )
            continue

        # Voice input: if voice mode is on and input is empty-ish (just Enter),
        # start recording
        if voice_mode and text == "" and voice_recorder is not None and voice_stt is not None:
            try:
                voice_recorder.start()
            except RuntimeError as rec_err:
                console.print(f"[red]Mic error: {rec_err}[/red]")
                continue
            console.print("[yellow]Recording... (press Enter to stop)[/yellow]")
            try:
                await asyncio.to_thread(
                    prompt_session.prompt,
                    HTML("<prompt>[recording] </prompt>"),
                )
            except (EOFError, KeyboardInterrupt):
                voice_recorder.cancel()
                continue
            clip = voice_recorder.stop()
            if clip.duration < 0.3:
                console.print("[dim]Too short, skipping[/dim]")
                continue
            console.print(f"[dim]Transcribing {clip.duration:.1f}s of audio...[/dim]")
            text = await voice_stt.transcribe(clip)
            if not text.strip():
                console.print("[dim]No speech detected[/dim]")
                continue
            console.print(f"[cyan]> {text}[/cyan]")

        # Run agent with streaming
        try:
            await _run_agent_streaming(agent, text, renderer=stream_renderer)
            # TTS: speak the assistant's streamed text
            if speak_mode and tts_backend is not None and stream_renderer._text_buffer:
                spoken_text = "".join(stream_renderer._text_buffer)
                if spoken_text.strip():
                    try:
                        await tts_backend.speak(spoken_text)
                    except Exception as tts_exc:
                        console.print(f"[red dim]TTS error: {tts_exc}[/red dim]")
        except Exception:
            console.print_exception()

    if agent is not None:
        await agent.shutdown()
    for mc in mcp_clients:
        await mc.close()
    if provider is not None:
        await provider.close()
    await hot_store.close()
    await sqlite.close()
    if tts_backend is not None:
        await tts_backend.close()
    if voice_stt is not None:
        await voice_stt.close()
    console.print("[dim]Goodbye.[/dim]")


def main() -> None:
    """Entry point for `leuk` CLI command."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_run_repl())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
