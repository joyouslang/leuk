"""Read-only host observation: screen capture (+ HiDPI scaling) and system info.

Nothing here writes to or controls the machine — it only *gathers data*. Shared by
the low-risk ``monitoring`` tool and the high-risk ``input_control`` tool (which
reuses the screenshot helpers to verify its actions).

Screenshots are tried across several backends so it works on common X11 and
Wayland setups; on failure a precise reason says exactly what to install (see
docs/reference/system-dependencies.md#screenshots).
"""

from __future__ import annotations

import base64
import logging
import os
import platform
import shutil
import socket
import subprocess
from datetime import datetime

from leuk.media import png_size

logger = logging.getLogger(__name__)

_TIMEOUT = 10.0
DOC_HINT = "see docs/reference/system-dependencies.md#screenshots"

# HiDPI scaling: vision APIs downscale large screenshots, so a 4K capture is
# resized so its long edge ≈ TARGET_LONG_EDGE (WXGA) — both for sane image sizes
# and so input_control can map model coordinates back to real pixels.
TARGET_LONG_EDGE = 1366


def compute_scale(width: int, height: int, target: int = TARGET_LONG_EDGE) -> float:
    """Downscale factor so the screenshot's long edge ≈ *target*. Never upscales."""
    long_edge = max(width, height)
    if long_edge <= 0 or long_edge <= target:
        return 1.0
    return target / long_edge


def to_physical(value: float, scale: float) -> int:
    """Map a logical (model/screenshot-space) coordinate back to a real pixel."""
    if scale <= 0:
        return int(round(value))
    return int(round(value / scale))


def pil_available() -> bool:
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


def downscale_png(png: bytes, scale: float) -> bytes:
    """Resize a PNG by *scale* (≤1.0). Returns the original bytes if not scaling."""
    if scale >= 1.0:
        return png
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(png)) as im:
            w, h = im.size
            new = (max(1, round(w * scale)), max(1, round(h * scale)))
            resized = im.convert("RGB").resize(new, Image.Resampling.LANCZOS)
            out = io.BytesIO()
            resized.save(out, format="PNG")
            return out.getvalue()
    except ImportError:
        return png
    except Exception:  # noqa: BLE001 — never let a resize failure break a capture
        logger.debug("screenshot downscale failed; sending native resolution", exc_info=True)
        return png


# ── Screen capture ─────────────────────────────────────────────────


def _run_capture(cmd: list[str], *, to_stdout: bool) -> tuple[bytes | None, str]:
    """Run a screenshot command; return (png_bytes, error_reason)."""
    try:
        if to_stdout:
            proc = subprocess.run(cmd, capture_output=True, timeout=_TIMEOUT, check=False)
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout, ""
            return None, (proc.stderr.decode(errors="replace").strip() or f"exit {proc.returncode}")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            path = tf.name
        try:
            proc = subprocess.run([*cmd, path], capture_output=True, timeout=_TIMEOUT, check=False)
            data = b""
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError:
                pass
            if data:
                return data, ""
            return None, (proc.stderr.decode(errors="replace").strip() or f"exit {proc.returncode}")
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)


def capture_png() -> tuple[bytes | None, str]:
    """Capture the screen as PNG. Returns (png_bytes, "") or (None, reason)."""
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    reasons: list[str] = []

    def _try(name: str, cmd: list[str] | None, *, stdout: bool) -> bytes | None:
        if cmd is None or not shutil.which(cmd[0]):
            reasons.append(f"{name}: not installed")
            return None
        png, err = _run_capture(cmd, to_stdout=stdout)
        if png:
            return png
        reasons.append(f"{name}: {err}")
        return None

    is_wayland = session == "wayland"

    if not is_wayland:
        try:
            from mss import mss as MSS
            from mss.tools import to_png

            with MSS() as sct:
                mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                raw = sct.grab(mon)
                return to_png(raw.rgb, raw.size), ""
        except ImportError:
            reasons.append("mss: not installed (pip install 'leuk[monitoring]')")
        except Exception as exc:  # noqa: BLE001 — no DISPLAY etc.
            reasons.append(f"mss: {exc}")

    if is_wayland:
        for name, cmd, stdout in (
            ("grim", ["grim", "-"], True),
            ("gnome-screenshot", ["gnome-screenshot", "-f"], False),
            ("spectacle", ["spectacle", "-b", "-n", "-o"], False),
        ):
            png = _try(name, cmd, stdout=stdout)
            if png:
                return png, ""
    else:
        for name, cmd, stdout in (
            ("scrot", ["scrot", "-o"], False),
            ("maim", ["maim"], False),
            ("import", ["import", "-window", "root"], False),
        ):
            png = _try(name, cmd, stdout=stdout)
            if png:
                return png, ""

    reason = "; ".join(reasons) if reasons else "no screenshot backend available"
    return None, f"{reason} ({DOC_HINT})"


def screenshot_tag(scale: float = 1.0) -> tuple[str | None, str]:
    """Return (``[screenshot:…]`` tag, "") on success or (None, reason)."""
    png, reason = capture_png()
    if not png:
        return None, reason
    png = downscale_png(png, scale)
    return f"[screenshot:image/png;base64,{base64.b64encode(png).decode()}]", ""


def screen_size() -> tuple[tuple[int, int] | None, str]:
    """Return the physical screen size ((w, h), "") or (None, reason)."""
    try:
        from mss import mss as MSS

        with MSS() as sct:
            mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            return (int(mon["width"]), int(mon["height"])), ""
    except ImportError:
        pass
    except Exception:  # noqa: BLE001
        pass
    png, reason = capture_png()
    if png:
        size = png_size(png)
        if size:
            return size, ""
        return None, "captured image had no readable dimensions"
    return None, reason


# ── System info ────────────────────────────────────────────────────


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024 or unit == "TB":
            return f"{int(size)}B" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _linux_meminfo() -> str | None:
    try:
        fields: dict[str, int] = {}
        for line in open("/proc/meminfo", encoding="utf-8"):
            key, _, rest = line.partition(":")
            kb = rest.strip().split()
            if kb and kb[0].isdigit():
                fields[key] = int(kb[0]) * 1024
    except OSError:
        return None
    total = fields.get("MemTotal")
    avail = fields.get("MemAvailable")
    if total is None:
        return None
    used = total - (avail if avail is not None else 0)
    return f"Memory: {_human_bytes(used)}/{_human_bytes(total)} used ({_human_bytes(avail or 0)} available)"


def _linux_uptime() -> str | None:
    try:
        secs = float(open("/proc/uptime", encoding="utf-8").read().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    d, rem = divmod(int(secs), 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    parts = [f"{d}d" for _ in (1,) if d] + [f"{h}h", f"{m}m"]
    return "Uptime: " + " ".join(parts)


def system_info() -> str:
    """A read-only summary of host facts (OS, CPU, memory, disk, uptime)."""
    lines = [
        f"OS: {platform.system()} {platform.release()} ({platform.machine()})",
        f"Hostname: {socket.gethostname()}",
        f"Python: {platform.python_version()}",
        f"CPU cores: {os.cpu_count()}",
    ]
    try:
        la = os.getloadavg()
        lines.append(f"Load average (1/5/15m): {la[0]:.2f} {la[1]:.2f} {la[2]:.2f}")
    except (OSError, AttributeError):
        pass
    mem = _linux_meminfo()
    if mem:
        lines.append(mem)
    try:
        du = shutil.disk_usage("/")
        lines.append(
            f"Disk /: {_human_bytes(du.used)}/{_human_bytes(du.total)} used "
            f"({_human_bytes(du.free)} free)"
        )
    except OSError:
        pass
    up = _linux_uptime()
    if up:
        lines.append(up)
    lines.append(f"Time: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    return "\n".join(lines)
