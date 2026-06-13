# leuk wiki

> **Ground truth.** This wiki is the canonical documentation for leuk and is kept
> in sync with the codebase. When code changes, the relevant page is updated in
> the same change. If a page and the code disagree, the code wins — fix the page.

leuk is a persistent, terminal-based AI agent with sub-agent orchestration,
multi-provider LLM support, multimodal input, voice, desktop/browser control,
and environment access (shell, files, web, MCP).

## Index

### Guides
- [Getting Started](getting-started.md) — install, extras, first run, authentication
- [REPL & Commands](repl-commands.md) — the interactive prompt and every slash command
- [Configuration](configuration.md) — config files, precedence, feature toggles
- [Development](development.md) — tests, linting, adding tools, project layout

### Architecture
- [Architecture Overview](architecture.md) — the agent loop, streaming, sub-agents, teams
- [Providers](providers.md) — Anthropic, OpenAI, Google, OpenRouter, Zen, local
- [Context Management](context-management.md) — tiered compaction, archiving
- [Sessions & Persistence](sessions-and-persistence.md) — lifecycle, SQLite, hot store
- [Safety & Approvals](safety.md) — review policies, rules, persistent approvals
- [CLI & UI](cli-and-ui.md) — REPL UX, themes, banner, footer, settings dialog
- [MCP](mcp.md) — connecting external tool servers, importing connectors + exposing leuk as a server
- [Scheduler](scheduler.md) — background scheduled tasks

### Capabilities
- [Tools](tools.md) — the tool system and built-in tools
  - [Browser](tools/browser.md) — Playwright SPA/AJAX automation
  - [Input Control](tools/input_control.md) — desktop keyboard/mouse (X11 + Wayland)
- [Skills](skills.md) — install & use SKILL.md agent skills (OpenClaw/ClawHub-compatible)
- [Voice](voice.md) — speech-to-text, text-to-speech, VAD (half-duplex)
- [Multimodal](multimodal.md) — images & audio sent natively to the model
- [Channels](channels.md) — Telegram, Slack, Discord, allowlist, remote approvals

### Reference
- [Environment Variables](reference/environment.md) — full `LEUK_*` reference
- [System Dependencies](reference/system-dependencies.md) — OS/distro packages (ydotool, grim, …)
- [File Layout](reference/file-layout.md) — source tree map
- [Refinements Backlog](refinements-plan.md) — deferred work + future ideas

## Conventions

- Each page starts with a breadcrumb (`Home › …`) and ends with a **See also**
  list of related pages.
- Code references use `path:symbol` form, e.g. `src/leuk/agent/core.py:Agent`.
- Commits use [Conventional Commits](https://www.conventionalcommits.org/).

## Keeping the wiki in sync

Whenever a change touches behavior, configuration, commands, tools, providers,
or architecture, update the matching wiki page **in the same change**. The
root [`README.md`](../README.md) is a short landing page that points here;
[`CLAUDE.md`](../CLAUDE.md) is the agent-facing quickstart. Both defer to this
wiki for detail.
