"""Shared visual theme for the leuk REPL.

A registry of named colour palettes (gruvbox, dracula, nord, …) is turned
into a :class:`rich.theme.Theme` plus a pygments code-style name. The active
theme is held in module globals (:data:`PALETTE`, :data:`LEUK_THEME`,
:data:`CODE_THEME`) and can be switched at runtime via :func:`apply_theme`.

Everything that prints should use *named styles* (e.g. ``[accent]…[/accent]``,
``[comment]…[/comment]``) so a theme switch re-colours the whole UI.
"""

from __future__ import annotations

from rich.theme import Theme

DEFAULT_THEME = "gruvbox"

# ── Theme registry ─────────────────────────────────────────────────
# Each entry: a colour palette + a "code" pygments style + a display "label".
# Palettes share the same keys so any theme drops straight into the styles.

THEMES: dict[str, dict[str, str]] = {
    "gruvbox": {
        "label": "Gruvbox",
        "code": "gruvbox-dark",
        "fg": "#ebdbb2", "blue": "#83a598", "purple": "#d3869b", "cyan": "#8ec07c",
        "green": "#b8bb26", "yellow": "#fabd2f", "red": "#fb4934", "orange": "#fe8019",
        "grey": "#928374", "diff_add_bg": "#32361a", "diff_del_bg": "#3c1f1e",
    },
    "dracula": {
        "label": "Dracula",
        "code": "dracula",
        "fg": "#f8f8f2", "blue": "#bd93f9", "purple": "#ff79c6", "cyan": "#8be9fd",
        "green": "#50fa7b", "yellow": "#f1fa8c", "red": "#ff5555", "orange": "#ffb86c",
        "grey": "#6272a4", "diff_add_bg": "#1d3b27", "diff_del_bg": "#3b1d28",
    },
    "nord": {
        "label": "Nord",
        "code": "nord",
        "fg": "#d8dee9", "blue": "#81a1c1", "purple": "#b48ead", "cyan": "#88c0d0",
        "green": "#a3be8c", "yellow": "#ebcb8b", "red": "#bf616a", "orange": "#d08770",
        "grey": "#6b7488", "diff_add_bg": "#2b3a2e", "diff_del_bg": "#3a2b2e",
    },
    "tokyonight": {
        "label": "Tokyo Night",
        "code": "one-dark",
        "fg": "#c0caf5", "blue": "#7aa2f7", "purple": "#bb9af7", "cyan": "#7dcfff",
        "green": "#9ece6a", "yellow": "#e0af68", "red": "#f7768e", "orange": "#ff9e64",
        "grey": "#565f89", "diff_add_bg": "#1e3a2e", "diff_del_bg": "#3a1e2a",
    },
    "catppuccin": {
        "label": "Catppuccin Mocha",
        "code": "material",
        "fg": "#cdd6f4", "blue": "#89b4fa", "purple": "#cba6f7", "cyan": "#94e2d5",
        "green": "#a6e3a1", "yellow": "#f9e2af", "red": "#f38ba8", "orange": "#fab387",
        "grey": "#7f849c", "diff_add_bg": "#28351f", "diff_del_bg": "#352230",
    },
    "solarized": {
        "label": "Solarized Dark",
        "code": "solarized-dark",
        "fg": "#93a1a1", "blue": "#268bd2", "purple": "#6c71c4", "cyan": "#2aa198",
        "green": "#859900", "yellow": "#b58900", "red": "#dc322f", "orange": "#cb4b16",
        "grey": "#586e75", "diff_add_bg": "#12331a", "diff_del_bg": "#33161a",
    },
}


def theme_choices() -> list[tuple[str, str]]:
    """``[(key, label), …]`` for building a settings picker."""
    return [(key, t["label"]) for key, t in THEMES.items()]


def _build_rich_theme(p: dict[str, str]) -> Theme:
    """Construct the rich :class:`Theme` of named styles from a palette."""
    return Theme(
        {
            # General text roles
            "primary": p["fg"],
            "comment": p["grey"],
            "dim": p["grey"],
            "accent": f"bold {p['blue']}",
            "accent.blue": p["blue"],
            "accent.purple": p["purple"],
            "accent.cyan": p["cyan"],
            "accent.green": p["green"],
            "accent.yellow": p["yellow"],
            "accent.red": p["red"],
            "accent.orange": p["orange"],
            # Roles in the conversation
            "user.label": f"bold {p['cyan']}",
            "assistant.label": f"bold {p['purple']}",
            # Tool-call lifecycle
            "tool.name": f"bold {p['fg']}",
            "tool.desc": p["grey"],
            "tool.pending": p["yellow"],
            "tool.running": p["yellow"],
            "tool.success": p["green"],
            "tool.failed": f"bold {p['red']}",
            "tool.border": p["grey"],
            # Diffs
            "diff.add": f"{p['green']} on {p['diff_add_bg']}",
            "diff.del": f"{p['red']} on {p['diff_del_bg']}",
            "diff.hunk": p["cyan"],
            # Status / notices
            "status.ok": p["green"],
            "status.warn": p["yellow"],
            "status.error": f"bold {p['red']}",
            "status.info": p["blue"],
            # Footer / banner
            "banner": f"bold {p['blue']}",
            "banner.sub": p["purple"],
            "footer": p["grey"],
            "footer.key": p["cyan"],
            "footer.value": p["fg"],
            "footer.sep": p["grey"],
            "rule.line": p["grey"],
        }
    )


def _palette_of(name: str) -> dict[str, str]:
    entry = THEMES.get(name) or THEMES[DEFAULT_THEME]
    return {k: v for k, v in entry.items() if k not in ("label", "code")}


# ── Active theme (rebindable via apply_theme) ──────────────────────

ACTIVE_THEME = DEFAULT_THEME
PALETTE = _palette_of(DEFAULT_THEME)
LEUK_THEME = _build_rich_theme(PALETTE)
CODE_THEME = THEMES[DEFAULT_THEME]["code"]


def apply_theme(name: str) -> Theme:
    """Switch the active theme, rebinding the module-level palette/theme.

    Unknown names fall back to the default. Returns the new rich Theme so
    callers can ``console.push_theme(...)`` it.
    """
    global ACTIVE_THEME, PALETTE, LEUK_THEME, CODE_THEME
    name = name if name in THEMES else DEFAULT_THEME
    ACTIVE_THEME = name
    PALETTE = _palette_of(name)
    LEUK_THEME = _build_rich_theme(PALETTE)
    CODE_THEME = THEMES[name]["code"]
    return LEUK_THEME
