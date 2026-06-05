"""``leuk doctor`` — diagnose optional features and print exact setup steps.

Every optional capability (desktop control, screenshots, browser, voice, local
LLM) needs some mix of a Python extra, a system binary, a running daemon, or a
permission. This module checks each requirement and, for anything missing, emits
a copy-pasteable fix tailored to the detected distro and display server — plus
how to *enable* the feature once its dependencies are met.

The checks are read-only; nothing is installed or changed. Run via ``leuk
doctor`` or the ``/doctor`` REPL command.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console


@dataclass
class Check:
    """One requirement: satisfied or not, with how to fix it."""

    label: str
    ok: bool
    detail: str = ""
    fix: list[str] = field(default_factory=list)


@dataclass
class Section:
    """A feature: its checks and how to turn it on once they pass."""

    title: str
    summary: str
    checks: list[Check]
    enable: list[str]
    enabled_now: bool | None = None  # None = not a config-gated feature


# ── environment detection ──────────────────────────────────────────


def _os_release() -> dict[str, str]:
    data: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                data[k.strip()] = v.strip().strip('"')
    except OSError:
        pass
    return data


def _distro_family() -> set[str]:
    """Lower-cased {ID} ∪ ID_LIKE tokens, e.g. {'neon','ubuntu','debian'}."""
    rel = _os_release()
    fam = {rel.get("ID", "").lower()}
    fam |= {t for t in rel.get("ID_LIKE", "").lower().split() if t}
    return {f for f in fam if f}


def _pkg_install(*pkgs: str) -> list[str]:
    """A package-install command line for the detected distro (best effort)."""
    fam = _distro_family()
    joined = " ".join(pkgs)
    if {"debian", "ubuntu"} & fam:
        return [f"sudo apt install -y {joined}"]
    if {"fedora", "rhel", "centos"} & fam:
        return [f"sudo dnf install -y {joined}"]
    if {"arch", "manjaro"} & fam:
        return [f"sudo pacman -S --needed {joined}"]
    if {"opensuse", "suse", "sles"} & fam:
        return [f"sudo zypper install -y {joined}"]
    return [f"Install '{joined}' with your distribution's package manager."]


def _session() -> str:
    return os.environ.get("XDG_SESSION_TYPE", "").lower() or "unknown"


def _has(binary: str) -> str | None:
    return shutil.which(binary)


def _module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _ydotool_socket() -> str | None:
    """First existing ydotoold socket, mirroring InputControlTool's lookup."""
    cands = []
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        cands.append(os.path.join(runtime, ".ydotool_socket"))
    try:
        cands.append(f"/run/user/{os.getuid()}/.ydotool_socket")
    except AttributeError:
        pass
    cands.append("/tmp/.ydotool_socket")
    env = os.environ.get("LEUK_INPUT_CONTROL_YDOTOOL_SOCKET")
    if env:
        cands.insert(0, env)
    return next((c for c in cands if c and os.path.exists(c)), None)


_UINPUT_FIX = [
    'sudo usermod -aG input "$USER"      # then log out and back in',
    "echo 'KERNEL==\"uinput\", GROUP=\"input\", MODE=\"0660\"' | "
    "sudo tee /etc/udev/rules.d/99-uinput.rules",
    "sudo udevadm control --reload-rules && sudo udevadm trigger",
    "sudo modprobe uinput",
]

# Debian/Ubuntu apt only ships ydotool 0.1.x (incompatible CLI), so v1.x must be
# built from source there; other distros carry v1.x in their repos.
_YDOTOOL_SOURCE_BUILD = [
    "sudo apt install -y build-essential cmake scdoc git",
    "git clone https://github.com/ReimuNotMoe/ydotool /tmp/ydotool",
    "cmake -B /tmp/ydotool/build /tmp/ydotool && make -C /tmp/ydotool/build",
    "sudo make -C /tmp/ydotool/build install",
]


def _install_modern_ydotool() -> list[str]:
    """How to get ydotool ≥ 1.0 (build from source on Debian/Ubuntu)."""
    if {"debian", "ubuntu"} & _distro_family():
        return _YDOTOOL_SOURCE_BUILD
    return _pkg_install("ydotool")  # Fedora/Arch/openSUSE ship v1.x


# ── per-feature checks ─────────────────────────────────────────────


def _screenshot_check() -> Check:
    """Screenshots back input_control verification and browser/desktop capture."""
    if _session() == "wayland":
        for b in ("grim", "gnome-screenshot", "spectacle"):
            if _has(b):
                return Check("screenshot backend (Wayland)", True, b)
        return Check(
            "screenshot backend (Wayland)", False, "none found", _pkg_install("grim")
        )
    if _module("mss"):
        return Check("screenshot backend (X11)", True, "mss (Python)")
    for b in ("scrot", "maim", "import"):
        if _has(b):
            return Check("screenshot backend (X11)", True, b)
    return Check(
        "screenshot backend (X11)",
        False,
        "none found",
        ["uv sync --extra input-control   # installs mss", *_pkg_install("scrot")],
    )


def _input_control_section(enabled: bool) -> Section:
    from leuk.tools.input_control import ydotool_supports_absolute

    checks: list[Check] = []
    yd = _has("ydotool")
    checks.append(
        Check(
            "ydotool installed",
            bool(yd),
            yd or "not found",
            [] if yd else _install_modern_ydotool(),
        )
    )
    if yd:
        modern = ydotool_supports_absolute(yd)
        checks.append(
            Check(
                "ydotool ≥ 1.0 (absolute mouse positioning)",
                modern,
                "ok" if modern else "v0.1.x — CLI too old; actions silently no-op",
                [] if modern else _install_modern_ydotool(),
            )
        )
        if modern:
            ydd = _has("ydotoold")
            checks.append(
                Check(
                    "ydotoold installed (v1.x needs it)",
                    bool(ydd),
                    ydd or "not found",
                    [] if ydd else _install_modern_ydotool(),
                )
            )
            if ydd:
                sock = _ydotool_socket()
                checks.append(
                    Check(
                        "ydotoold running",
                        sock is not None,
                        sock or "no socket — daemon not started",
                        []
                        if sock
                        else [
                            "ydotoold &     # quick foreground test",
                            "bash scripts/setup-input-control.sh   # persistent user service",
                        ],
                    )
                )
        writable = os.access("/dev/uinput", os.W_OK)
        checks.append(
            Check(
                "/dev/uinput accessible",
                writable,
                "ok" if writable else "no access (whoever runs ydotoold needs it)",
                [] if writable else _UINPUT_FIX,
            )
        )
    checks.append(_screenshot_check())
    return Section(
        "Desktop control — input_control tool",
        "Let the agent move the mouse, click, and type on your real desktop.",
        checks,
        enable=[
            "Toggle it on in /settings → General (saved to config.json)",
            "DANGEROUS, optional: turn on desktop-control auto-approve in /settings"
            " (skips per-action approval)",
        ],
        enabled_now=enabled,
    )


def _playwright_chromium_present() -> bool:
    base = Path(
        os.environ.get("PLAYWRIGHT_BROWSERS_PATH", str(Path.home() / ".cache" / "ms-playwright"))
    )
    try:
        return any(p.name.startswith("chromium") for p in base.iterdir())
    except OSError:
        return False


def _browser_section(enabled: bool) -> Section:
    checks: list[Check] = []
    pw = _module("playwright")
    checks.append(
        Check(
            "playwright (Python)",
            pw,
            "installed" if pw else "missing",
            [] if pw else ["uv sync --extra browser"],
        )
    )
    if pw:
        chromium = _playwright_chromium_present()
        checks.append(
            Check(
                "chromium browser",
                chromium,
                "installed" if chromium else "not downloaded",
                [] if chromium else ["python -m playwright install chromium"],
            )
        )
    return Section(
        "Browser automation — browser tool",
        "Drive a real Chromium browser (SPA/AJAX-aware) for web tasks.",
        checks,
        enable=["Toggle it on in /settings → General (saved to config.json)"],
        enabled_now=enabled,
    )


def _voice_section() -> Section:
    try:
        from leuk.voice import VOICE_AVAILABLE, _MISSING_REASON
    except Exception:  # noqa: BLE001
        VOICE_AVAILABLE, _MISSING_REASON = False, "import failed"
    checks = [
        Check(
            "voice extra (torch, sounddevice, transformers)",
            VOICE_AVAILABLE,
            "ready" if VOICE_AVAILABLE else (_MISSING_REASON or "missing"),
            [] if VOICE_AVAILABLE else ["uv sync --extra voice"],
        ),
        Check(
            "num2words (spoken numbers)",
            _module("num2words"),
            "installed" if _module("num2words") else "missing",
            [] if _module("num2words") else ["uv sync --extra voice"],
        ),
    ]
    if not VOICE_AVAILABLE:
        # sounddevice needs the PortAudio system library.
        checks.append(
            Check(
                "PortAudio system library",
                False,
                "needed by sounddevice if 'voice ok' fails to import",
                _pkg_install("libportaudio2"),
            )
        )
    return Section(
        "Voice — speech in / out",
        "Hands-free voice input (STT) and spoken replies (TTS), fully offline.",
        checks,
        enable=["In the REPL: /voice for input, /speak for text-to-speech"],
    )


def _local_llm_section(enabled: bool) -> Section:
    ol = _has("ollama")
    checks = [
        Check(
            "ollama installed (optional)",
            bool(ol),
            ol or "not found",
            [] if ol else ["Install from https://ollama.com"],
        )
    ]
    return Section(
        "Local LLM — local_llm tool (Ollama)",
        "Delegate sub-tasks to a local Ollama model.",
        checks,
        enable=[
            'Set {"local_llm": {"enabled": true}} in ~/.config/leuk/config.json',
            "Run `ollama serve` and pull a model, e.g. `ollama pull llama3.2`",
        ],
        enabled_now=enabled,
    )


def _skills_mcp_section(skills_enabled: bool) -> Section:
    clawhub = _has("clawhub")
    git = _has("git")
    try:
        import PIL  # noqa: F401, PLC0415

        pillow = True
    except ImportError:
        pillow = False
    checks = [
        Check(
            "clawhub CLI (optional)",
            bool(clawhub),
            clawhub or "not found",
            [] if clawhub else [
                "Install ClawHub's CLI to import skills/plugins by slug "
                "(https://github.com/openclaw/clawhub). Git URLs / local paths work without it."
            ],
        ),
        Check(
            "git (git-URL skill import)",
            bool(git),
            git or "not found",
            [] if git else _pkg_install("git"),
        ),
        Check(
            "Pillow (inline media thumbnails)",
            pillow,
            "installed" if pillow else "not installed",
            [] if pillow else ["uv sync --extra input-control"],
        ),
    ]
    return Section(
        "Skills & MCP connectors",
        "Import SKILL.md skills and MCP connectors. Skills are instructions-only and inert until trusted.",
        checks,
        enable=[
            'Enable skills: set {"skills": {"enabled": true}} in config.json (or /settings)',
            "Manage with /skills and /mcp (or `leuk skills …` / `leuk mcp …`)",
        ],
        enabled_now=skills_enabled,
    )


def build_report(settings: object | None = None) -> list[Section]:
    """Assemble the diagnostic sections (read-only)."""
    if settings is None:
        from leuk.config import load_settings

        settings = load_settings()
    ic = getattr(settings, "input_control", None)
    br = getattr(settings, "browser", None)
    ll = getattr(settings, "local_llm", None)
    sk = getattr(settings, "skills", None)
    return [
        _input_control_section(bool(getattr(ic, "enabled", False))),
        _browser_section(bool(getattr(br, "enabled", False))),
        _voice_section(),
        _local_llm_section(bool(getattr(ll, "enabled", False))),
        _skills_mcp_section(bool(getattr(sk, "enabled", False))),
    ]


# ── rendering ──────────────────────────────────────────────────────


def render_report(console: Console, sections: list[Section]) -> None:
    rel = _os_release()
    distro = rel.get("PRETTY_NAME") or rel.get("NAME") or "unknown OS"
    console.print()
    console.print("[bold]leuk doctor[/bold] — optional feature setup")
    console.print(f"[dim]System: {distro} · session: {_session()}[/dim]\n")

    for sec in sections:
        state = ""
        if sec.enabled_now is not None:
            state = (
                " [green](enabled)[/green]"
                if sec.enabled_now
                else " [dim](disabled)[/dim]"
            )
        console.print(f"[bold]{sec.title}[/bold]{state}")
        console.print(f"  [dim]{sec.summary}[/dim]")
        all_ok = True
        for chk in sec.checks:
            glyph = "[green]✓[/green]" if chk.ok else "[red]✗[/red]"
            detail = f" [dim]— {chk.detail}[/dim]" if chk.detail else ""
            console.print(f"  {glyph} {chk.label}{detail}")
            if not chk.ok:
                all_ok = False
                for step in chk.fix:
                    console.print(f"      [accent.blue]$[/accent.blue] {step}")
        if all_ok:
            console.print("  [dim]Enable:[/dim]")
        else:
            console.print("  [dim]Once the above is fixed, enable:[/dim]")
        for step in sec.enable:
            console.print(f"      [accent.blue]›[/accent.blue] {step}")
        console.print()

    console.print(
        "[dim]One-shot desktop-control setup: "
        "[bold]bash scripts/setup-input-control.sh[/bold]. "
        "Full reference: docs/reference/system-dependencies.md[/dim]\n"
    )


def run_doctor(console: Console | None = None) -> int:
    """Print the diagnostic report. Returns a process exit code (always 0)."""
    if console is None:
        from leuk.cli.theme import LEUK_THEME

        console = Console(theme=LEUK_THEME)
    render_report(console, build_report())
    return 0
