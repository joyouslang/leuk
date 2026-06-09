[Home](../README.md) › [Development](../development.md) › File Layout

# File Layout

```
src/leuk/
  __init__.py            # package + __version__
  __main__.py            # python -m leuk
  config.py              # Settings, credentials, env loading, feature toggles
  types.py               # Message, ToolCall/Result, Session, MediaPart, StreamEvent
  safety.py              # SafetyGuard, policies, rules, approvals
  media.py               # multimodal tag parsing + media file loading
  billing.py             # Anthropic billing header
  agent/
    core.py              # Agent loop, tool dispatch, safety gate, orphan healing
    session.py           # AgentSession (background task + queues)
    sub_agent.py         # SubAgentManager (spawn/archive child agents)
    team.py              # named roles / team orchestration
    context.py           # tiered compaction pipeline
    archive.py           # archive dropped messages to disk
  cli/
    repl.py              # interactive REPL loop, commands, voice/multimodal, COMMANDS
    tui.py               # persistent-input full-screen TUI (default): TuiRenderer + ReplTUI
    blocks.py            # shared scrollback block model + rich→ANSI bridge
    render.py            # StreamRenderer (classic-prompt fallback), history replay
    history_browser.py   # interactive history view: navigate + expand blocks
    banner.py            # startup banner
    theme.py             # theme registry (gruvbox default)
    settings_dialog.py   # /settings full-screen dialog
    doctor.py            # `leuk doctor` / `/doctor` setup diagnostics
    models.py            # /model selector
    auth.py              # /auth, OAuth PKCE, credentials
  providers/
    base.py catalog.py   # protocol + factory
    anthropic.py openai.py google.py openrouter.py zen.py
    context_window.py    # resolve model window (live query → override → unknown)
    model_info.py        # queried model metadata: context window + vision/audio
  tools/
    base.py __init__.py  # Tool protocol + registry
    shell.py file_read.py file_edit.py web_fetch.py
    sub_agent.py memory_write.py local_llm.py
    browser.py input_control.py
  channels/
    base.py __init__.py  # Channel protocol + ChannelRegistry (auto-discovery)
    markdown.py          # Markdown → Telegram-HTML converter
    telegram.py slack.py discord.py pipe.py
  voice/
    __init__.py recorder.py stt.py tts.py
  persistence/
    base.py sqlite.py memory.py
  mcp/
    client.py bridge.py server.py
  scheduler/
    task.py store.py runner.py
  sandbox/
    container.py mount_policy.py
  memory/
    loader.py
docs/                    # this wiki (ground truth)
tests/                   # pytest suite
```

## See also

- [Development](../development.md) · [Architecture Overview](../architecture.md)
