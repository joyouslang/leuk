[Home](README.md) › REPL & Commands

# REPL & Commands

The REPL (`src/leuk/cli/repl.py`) is the interactive interface. Type a message to
talk to the agent, or a `/command`. The themed bottom status line shows cwd ·
git branch · model · context usage (used/window + free tokens) · policy.
Keyboard shortcuts are listed in **`/help`** (the footer carries no hints).

## Persistent-input TUI (default)

By default the REPL runs as a **full-screen, persistent-input TUI**
(`src/leuk/cli/tui.py`, design in `docs/repl-tui-design.md`): the input box stays
typable **while the agent thinks and streams**, and output accumulates in a
scrollable transcript above it (with the startup banner at the top).

- **Tab** completes slash-commands as you type (the dropdown shows each
  command's description; **Shift-Tab** cycles backwards). After `/cd` and
  `/file` it completes **filesystem paths** (`~` expands; `/cd` offers
  directories only; dotfiles appear when you type the leading `.`). **Up/Down**
  recall previous inputs (REPL history).
- **Scroll** with the mouse wheel **or** PgUp/PgDn. When scrolled up, a **⤓ Jump
  to latest** button appears at the bottom (click it, or PgDn back down, to
  re-follow live output).
- **Select & copy**: drag with the mouse (or **Shift+↑/↓**) to select text — the
  selection is copied to the system clipboard automatically (via OSC52) as it's
  made. Dragging past the top/bottom edge auto-scrolls; **Esc** clears the
  selection. **Click** a tool block to expand/collapse its full output, or an
  image to open it in your viewer.
- **Ctrl-C** always interrupts the running turn (it never copies — selections
  copy automatically). **Ctrl-D** quits.
- **Ctrl-T** expands/collapses the model's **live reasoning** while it thinks
  (supported by default — Anthropic extended thinking, Gemini thought
  summaries, DeepSeek-style `reasoning_content`; models that don't support it
  are detected from the API's own response and skipped). When the answer
  starts, the trace is frozen into a collapsed `✦ thinking` block in the
  transcript — click it to expand. Stored with the conversation, so it
  survives `/switch`. Note: some endpoints (e.g. the Claude-subscription OAuth
  path) **withhold the reasoning text** — the model thinks but streams no
  content; `/status` tells you when that's the case
  ([details](providers.md#thinking--reasoning-stream)).
- A submitted message streams in place; a message sent mid-turn is queued and
  answered after the current one.
- Tool **approvals** appear as an in-app overlay: **Enter** allow once, `a` always
  allow, `d` always deny, **Esc** deny once.
- A `/command` runs in the normal terminal (so dialogs work); its output is
  captured and folded back into the transcript when the TUI returns — no manual
  step needed.
- The TUI is the default with **no opt-in flag**. If the full-screen app can't
  start on a terminal, leuk automatically falls back to the classic line prompt.
- **Voice mode** (`/voice`) uses the classic line prompt (keyboard/voice race);
  the TUI resumes when voice is turned off.

In the classic line-prompt fallback, **Tab** autocompletes commands as you type a
prefix.

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
| `/status` | Session stats, [context-window usage](context-management.md), and whether [thinking/reasoning](providers.md) is being requested (or why not) |
| `/file <path>` | Attach an image/audio file (auto-detected) to your next message |
| `/doctor` | Check optional-feature setup (ydotool, screenshots, browser, voice…) and print fix steps |
| `/skills` | Manage [agent skills](skills.md) — add, trust, enable/disable, remove |
| `/mcp` | Manage [MCP connectors/plugins](mcp.md) — search, add, enable/disable, remove |
| `/voice` | Toggle hands-free [voice input](voice.md) |
| `/speak` | Toggle [text-to-speech](voice.md) output |
| `/settings` | Open the [settings dialog](cli-and-ui.md) (theme, STT/TTS/VAD, toggles) |
| `/retry` | Re-send the last message after an error (or an unfinished turn — see crash recovery below) |
| `/undo` | Revert the last turn: restore files from a pre-turn **git snapshot** and drop the exchange from context (see below) |
| `/quit` | Exit leuk |

> The command list is generated from a single source of truth, `COMMANDS` in
> `src/leuk/cli/repl.py`, which also drives `/help` and Tab autocompletion. Keep
> this table in sync with that list.

## Reading past output

Tool and sub-agent results render **compact** in the transcript — there's no
`/verbose` toggle. The whole conversation is the scrollable transcript itself
(scroll with the wheel or PgUp/PgDn), so there's no separate history view:

| Action | Effect |
|--------|--------|
| mouse wheel / `PgUp` / `PgDn` | Scroll the transcript |
| click a tool / sub-agent block | Expand / collapse its full output in place |
| click an image | Open it in the external viewer |
| drag | Select text → copied to the system clipboard (auto-scrolls at edges) |
| **⤓ Jump to latest** (or `PgDn` to the end) | Re-follow live output |

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
- **Crash recovery.** A user message is persisted *before* its turn runs, so if
  leuk is killed mid-response, switching back to that session (`/switch`) detects
  the unanswered turn and offers **`/retry`** to re-send it.
- **`/undo` is a real undo, not just a context pop.** Before every turn the
  working tree (tracked + untracked, minus `.gitignore`d files) **and the
  `HEAD`/branch positions** are captured as a hidden git snapshot
  (`src/leuk/agent/undo.py`) — built through a temporary index, so your real
  index and stash are never touched. `/undo` restores files changed by the
  turn, deletes files it created, **unwinds commits the agent made** (branch
  reset to the pre-turn tip; the commits stay reachable via the reflog),
  re-attaches `HEAD` if the turn switched/created a branch, and removes the
  exchange from the conversation (memory + SQLite). Up to **5** turns can be
  undone (in-process; the stack doesn't survive a restart). Outside a git repo
  it degrades to context-only with a warning — `git init` enables full undo.
  Shell side-effects (processes, network) can't be undone, and the summary says
  so.

## See also

- [Configuration](configuration.md) · [Safety & Approvals](safety.md) · [CLI & UI](cli-and-ui.md)
