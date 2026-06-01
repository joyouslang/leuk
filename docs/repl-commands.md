[Home](README.md) › REPL & Commands

# REPL & Commands

The REPL (`src/leuk/cli/repl.py`) is the interactive interface. Type a message to
talk to the agent, or a `/command`. **Tab** autocompletes commands as you type a
prefix; the dropdown shows each command's description. The bottom status line
shows cwd · branch · session · model · context% · policy.

- **Ctrl-C** during a turn interrupts the agent (returns to the prompt); a second
  press force-stops. **Ctrl-D** quits.
- Input can come from the keyboard **or** voice when `/voice` is on.

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show this help message |
| `/models` | Select model (interactive dialog) |
| `/new` | Start a new session (lazy — created on first message) |
| `/sessions` | List recent top-level sessions |
| `/subagents [<id>]` | List sub-agent sessions, or view one's history |
| `/switch <id>` | Switch to a session by id prefix (clears screen, replays history) |
| `/rename <name>` | Rename the current session |
| `/delete [<id>]` | Delete current (modal picker for what's next) or another session |
| `/detach` | Detach from the session (it keeps running in the background) |
| `/auth` | Select provider / manage credentials |
| `/readonly` | Toggle read-only mode (block all writes) |
| `/safety` | Show safety guardrail status |
| `/tasks` | List scheduled tasks |
| `/policy <mode>` | Show or set the [review policy](safety.md) |
| `/desktop-auto` | Toggle desktop-control auto-approval (**dangerous**) |
| `/approvals` | List saved tool approvals (`/approvals clear` to reset) |
| `/status` | Session stats + [context-window usage](context-management.md) |
| `/history` | Open the interactive history browser (also **Tab** on an empty prompt) |
| `/file <path>` | Attach an image/audio file (auto-detected) to your next message |
| `/doctor` | Check optional-feature setup (ydotool, screenshots, browser, voice…) and print fix steps |
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
output is one keypress away, so there's no `/verbose` toggle. Press **Tab** on an
empty prompt (or run `/history`) to open the interactive browser over the current
conversation:

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
