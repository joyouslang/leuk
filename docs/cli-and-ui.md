[Home](README.md) › CLI & UI

# CLI & UI

The terminal UI is built with `rich` (output) and `prompt_toolkit` (input).

## Rendering — `src/leuk/cli/render.py`

`StreamRenderer` turns the agent event stream into:

- **Markdown** assistant output, streamed live (`rich.Live` + `Markdown`).
- **Tool blocks** — a header row `[✓/✗] tool  summary  time` + a rounded result
  panel, with syntax-highlighted diffs for `file_edit`.
- A braille **thinking spinner** with a Ctrl-C hint.
- **History replay** (`render_history`) used by `/switch`.

## Banner — `src/leuk/cli/banner.py`

Startup shows a block-letter "leuk" logo (theme-gradient), an info grid
(version/provider/model/cwd), and tips. No session line (sessions are
[lazy](sessions-and-persistence.md)).

## Status footer

A `prompt_toolkit` `bottom_toolbar` shows `cwd · branch · session · model ·
NN% ctx · policy · flags`. It renders while the prompt is active.

## Themes — `src/leuk/cli/theme.py`

A registry of palettes (each → a `rich.Theme` + a pygments code style + a `bg`
colour). Six themes: **Gruvbox** (default), Dracula, Nord, Tokyo Night,
Catppuccin Mocha, Solarized Dark. Pick in `/settings → General`;
`apply_theme()` switches live (console, prompt style, code blocks, banner
gradient) and persists to `config.json`. **No fixed colours anywhere in the
TUI**: the chrome (prompt, status footer, completion menu, approval overlay,
selection, jump button, frame borders) is styled by `theme.tui_style()` /
`theme.pt_style()`, resolved from the active palette **on every render** via a
`DynamicStyle` — a theme switch recolours the running interface immediately,
including the transcript content (the rich→ANSI bridge reads the theme
dynamically too).

**Every dialog and modal follows the active palette**, with no rich/prompt_toolkit
defaults bleeding through: the prompt_toolkit dialogs (`/settings`, `/model`,
`/skills`, `/mcp`, session pickers, the classic permission prompt) are built
through `settings_dialog.dialog_style()`, the in-app permission overlay through
`theme.tui_style()`, and the `/auth` flow (rich `Prompt`/`Confirm`) through a
console bound to `LEUK_THEME` with themed `prompt.*` styles — all read from the
active palette each time they're shown.

## Settings dialog — `src/leuk/cli/settings_dialog.py`

`/settings` is a full-screen `prompt_toolkit` dialog (arrow + **Enter** to
confirm, **Esc** to save & exit). Tabs: General (theme, browser, desktop-control
toggles), Speech-to-Text, Text-to-Speech, Voice Activity.

## Autocomplete

`SlashCommandCompleter` completes leading `/commands` from the `COMMANDS` table,
with a themed dropdown. See [REPL & Commands](repl-commands.md).

## See also

- [REPL & Commands](repl-commands.md) · [Voice](voice.md) · [Architecture Overview](architecture.md)
