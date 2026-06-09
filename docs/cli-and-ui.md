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

A registry of palettes (each → a `rich.Theme` + a pygments code style). Six
themes: **Gruvbox** (default), Dracula, Nord, Tokyo Night, Catppuccin Mocha,
Solarized Dark. Pick in `/settings → General`; `apply_theme()` switches live
(console, prompt style, code blocks, banner gradient) and persists to
`config.json`.

## Settings dialog — `src/leuk/cli/settings_dialog.py`

`/settings` is a full-screen `prompt_toolkit` dialog (arrow + **Enter** to
confirm, **Esc** to save & exit). Tabs: General (theme, browser, desktop-control
toggles), Speech-to-Text, Text-to-Speech, Voice Activity.

## Autocomplete

`SlashCommandCompleter` completes leading `/commands` from the `COMMANDS` table,
with a themed dropdown. See [REPL & Commands](repl-commands.md).

## See also

- [REPL & Commands](repl-commands.md) · [Voice](voice.md) · [Architecture Overview](architecture.md)
