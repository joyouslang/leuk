[Home](README.md) › REPL & Commands

# REPL & Commands

The REPL (`src/leuk/cli/repl.py`) is the interactive interface. Type a message to
talk to the agent, or a `/command`. The bottom status line shows cwd · branch ·
session · model · context% · policy.

## Persistent-input TUI (default)

By default the REPL runs as a **full-screen, persistent-input TUI**
(`src/leuk/cli/tui.py`, design in `docs/repl-tui-design.md`): the input box stays
typable **while the agent thinks and streams**, output accumulates in a
scrollable transcript above it, and **Tab** toggles focus between the input and a
navigable/expandable scrollback (the same block model as the history browser).

- **Ctrl-C** during a turn interrupts the agent. **Ctrl-D** quits.
- A submitted message streams in place; a message sent mid-turn is queued and
  answered after the current one.
- Tool **approvals** appear as an in-app overlay: **Enter** allow once, `a` always
  allow, `d` always deny, **Esc** deny once.
- A `/command` briefly drops back to the normal terminal to run (so dialogs and
  command output display as before), then returns to the TUI (press **Enter** at
  the `↵ back to leuk` prompt).
- The TUI is the default with **no opt-in flag**. If the full-screen app can't
  start on a terminal, leuk automatically falls back to the classic line prompt.
- **Voice mode** (`/voice`) uses the classic line prompt (keyboard/voice race);
  the TUI resumes when voice is turned off.

In the classic line-prompt fallback, **Tab** autocompletes commands as you type a
prefix (the dropdown shows each command's description), and **Tab** on an empty
line opens the history browser.

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show this help message |
| `/model` | Pick a model (themed dialog; ↑/↓, Enter to choose, Esc/q to cancel). Authorized providers only — redirects to `/auth` if the active provider has no credentials |
| `/cd <path>` | Change the current project directory (`~` expands; shell/file tools and project memory follow it) |
| `/new` | Start a new session (lazy — created on first message) |
| `/sessions` | List recent top-level sessions |
| `/subagents [<id>]` | List sub-agent sessions, or view one's history |
| `/switch <id>` | Switch to a session by id prefix (clears screen, replays history) |
| `/rename <name>` | Rename the current session |
| `/delete [<id>]` | Delete current (modal picker for what's next) or another session |
| `/detach` | Detach from the session (it keeps running in the background) |
| `/auth` | Manage provider authorization. Pick a provider by number to add / replace / delete its key (and switch to it) — no separate add/edit/delete commands. Separate from `/model` |
| `/readonly` | Toggle read-only mode (block all writes) |
| `/safety` | Show safety guardrail status |
| `/tasks` | List scheduled tasks |
| `/policy <mode>` | Show or set the [review policy](safety.md) |
| `/desktop-auto` | Toggle desktop-control auto-approval (**dangerous**) |
| `/approvals` | List saved tool approvals (`/approvals clear` to reset) |
| `/status` | Session stats + [context-window usage](context-management.md) |
| `/history` | Open the interactive history browser (in the TUI, **Tab** focuses the scrollback; in the classic prompt, **Tab** on an empty line opens it) |
| `/file <path>` | Attach an image/audio file (auto-detected) to your next message |
| `/doctor` | Check optional-feature setup (ydotool, screenshots, browser, voice…) and print fix steps |
| `/skills` | Manage [agent skills](skills.md) — add, trust, enable/disable, remove |
| `/mcp` | Manage [MCP connectors/plugins](mcp.md) — search, add, enable/disable, remove |
| `/voice` | Toggle hands-free [voice input](voice.md) |
| `/speak` | Toggle [text-to-speech](voice.md) output |
| `/settings` | Open the [settings dialog](cli-and-ui.md) (theme, STT/TTS/VAD, toggles) |
| `/retry` | Re-send the last message after an error |
| `/quit` | Exit leuk |

> The command list is generated from a single source of truth, `COMMANDS` in
> `src/leuk/cli/repl.py`, which also drives `/help` and Tab autocompletion. Keep
> this table in sync with that list.

## History browser

Tool and sub-agent results render **compact** in the live transcript — the full
output is one keypress away, so there's no `/verbose` toggle. In the TUI, press
**Tab** to focus the scrollback and expand blocks in place; in the classic
prompt, press **Tab** on an empty line (or run `/history`) to open the
standalone browser over the current conversation:

| Key | Action |
|-----|--------|
| `↑` / `↓` (or `k` / `j`) | Move the selection between blocks (view follows) |
| `Enter` / `Space` | Expand / collapse the selected tool / sub-agent block |
| **mouse wheel** / `PgUp` / `PgDn` | Scroll the view freely (independent of the selection) |
| mouse click | Select a block (click again to expand/collapse it) |
| `Home` / `End` | Select the first / latest block |
| `Tab` / `Esc` / `q` | Return to the REPL |

Implemented in `src/leuk/cli/history_browser.py`.

## Notable behaviors

- **Sessions are lazy.** Startup and `/new` create a *draft* that is only
  persisted when you send a message or pick an existing one. See
  [Sessions & Persistence](sessions-and-persistence.md).
- **`/delete`** of the current session opens a modal to pick the next session (or
  start fresh) — it never silently auto-creates one while others exist.
- **`/file`** stages an image/audio attachment for the next message; the model
  sees it natively. See [Multimodal](multimodal.md).
- **`/desktop-auto`** only matters when the [Input Control](tools/input_control.md)
  tool is enabled; it bypasses per-action approval (with a warning).

## See also

- [Configuration](configuration.md) · [Safety & Approvals](safety.md) · [CLI & UI](cli-and-ui.md)
