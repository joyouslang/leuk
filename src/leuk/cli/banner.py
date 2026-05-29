"""Startup banner for the leuk REPL.

Renders a gemini-cli-style header: a block-letter ASCII logo in a
blue→purple gradient, a compact info grid (version / provider / model /
session / cwd), and a short "tips" block. A narrow-terminal fallback
shows a single-line wordmark instead of the full logo.
"""

from __future__ import annotations

from pathlib import Path

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── ASCII logo ─────────────────────────────────────────────────────
# Block-letter "leuk", assembled from per-glyph rows so alignment is
# guaranteed. Each glyph is 5 rows tall.

_GLYPHS: dict[str, list[str]] = {
    "l": ["█▌    ", "█▌    ", "█▌    ", "█▌    ", "█████▌"],
    "e": ["█████▌", "█▌    ", "████▌ ", "█▌    ", "█████▌"],
    "u": ["█▌  █▌", "█▌  █▌", "█▌  █▌", "█▌  █▌", "█████▌"],
    "k": ["█▌  █▌", "█▌ █▌ ", "███▌  ", "█▌ █▌ ", "█▌  █▌"],
}

_LOGO_WORD = "leuk"


def _logo_lines() -> list[str]:
    """Assemble the multi-line block logo for ``leuk``."""
    rows: list[str] = []
    for r in range(5):
        rows.append("  ".join(_GLYPHS[ch][r] for ch in _LOGO_WORD))
    return rows


def _gradient_logo() -> Text:
    """Block logo coloured with a vertical blue→purple gradient."""
    lines = _logo_lines()
    n = max(len(lines) - 1, 1)
    # Interpolate between the active theme's blue and purple per row.
    from leuk.cli.theme import PALETTE

    def _hex(c: str) -> tuple[int, int, int]:
        c = c.lstrip("#")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

    blue = _hex(PALETTE["blue"])
    purple = _hex(PALETTE["purple"])
    text = Text()
    for i, line in enumerate(lines):
        t = i / n
        r = int(blue[0] + (purple[0] - blue[0]) * t)
        g = int(blue[1] + (purple[1] - blue[1]) * t)
        b = int(blue[2] + (purple[2] - blue[2]) * t)
        text.append(line + "\n", style=f"#{r:02x}{g:02x}{b:02x} bold")
    return text


# ── Public API ─────────────────────────────────────────────────────


def render_banner(
    console: Console,
    *,
    version: str,
    provider_label: str,
    model: str,
    channels: list[str] | None = None,
    cwd: str | None = None,
    show_logo: bool = True,
) -> None:
    """Print the startup banner to *console*.

    ``provider_label`` may contain Rich markup. No session line is shown:
    startup always begins a fresh session (named later from the first
    message), so there is nothing meaningful to display yet.
    """
    width = console.size.width
    blocks: list[object] = []

    if show_logo and width >= 44:
        blocks.append(Align.left(_gradient_logo()))
    else:
        wordmark = Text()
        wordmark.append("leuk", style="banner")
        wordmark.append(f"  v{version}", style="comment")
        blocks.append(wordmark)

    # Info grid — two columns (key / value), dim keys.
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style="comment", no_wrap=True)
    grid.add_column(style="footer.value")

    def _row(key: str, value: str) -> None:
        grid.add_row(key, value)

    if show_logo and width >= 44:
        _row("version", f"v{version}")
    _row("provider", f"{provider_label}  ·  [accent.cyan]{model}[/accent.cyan]")
    if cwd:
        _row("cwd", f"[accent.blue]{cwd}[/accent.blue]")
    if channels:
        _row("channels", f"[accent.cyan]{', '.join(channels)}[/accent.cyan]")

    blocks.append(grid)

    # Tips block. Commands are highlighted; surrounding prose stays muted.
    tips = Text()
    tips.append("Tips\n", style="accent.purple")
    for tip in (
        "Ask anything, or describe a task to get started.",
        "Use [footer.key]/help[/footer.key] for commands, "
        "[footer.key]/settings[/footer.key] to configure.",
        "Press [footer.key]Ctrl-C[/footer.key] to stop the agent, "
        "[footer.key]Ctrl-D[/footer.key] to quit.",
    ):
        tips.append("  • ", style="comment")
        tips.append_text(Text.from_markup(tip + "\n"))

    blocks.append(tips)

    console.print(
        Panel(
            Group(*blocks),
            border_style="rule.line",
            padding=(1, 2),
            expand=False,
        )
    )


def short_cwd(path: str | Path | None = None) -> str:
    """Return *path* with ``$HOME`` collapsed to ``~``."""
    p = Path(path) if path else Path.cwd()
    try:
        home = Path.home()
        if p == home or home in p.parents:
            return "~/" + str(p.relative_to(home)) if p != home else "~"
    except (ValueError, RuntimeError):
        pass
    return str(p)
