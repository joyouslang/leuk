"""Desktop keyboard/mouse control tool for Linux (X11 + Wayland).

Injects input via **ydotool** (kernel ``uinput``), so it works the same under
X11 and Wayland. Screenshots are a core capability used to verify the effect of
actions (mandatory after a failure or timeout): X11 via ``mss``, Wayland via
``grim``.

This tool is HIGH RISK — it drives the real desktop — and is gated behind
``input_control.enabled`` plus a dedicated approval path (see SafetyGuard).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import shutil
import subprocess
from typing import Any

from leuk import host
from leuk.host import compute_scale, to_physical  # noqa: F401 — re-exported for callers
from leuk.types import ToolSpec

logger = logging.getLogger(__name__)

_TIMEOUT = 10.0
# Long edge (px) of the magnified, coordinate-labelled image 'zoom' returns.
_ZOOM_OUT_LONG_EDGE = 1024

# Read-only screen capture + HiDPI coordinate scaling live in ``leuk.host`` (shared
# with the low-risk ``monitoring`` tool). input_control reuses them to take
# verification screenshots and to map the model's downscaled click coordinates
# back to real pixels — see ``leuk.host`` for the rationale.

# ── evdev keycodes (linux/input-event-codes.h) for `ydotool key` ────
KEYCODES: dict[str, int] = {
    # modifiers
    "ctrl": 29, "control": 29, "lctrl": 29, "rctrl": 97,
    "shift": 42, "lshift": 42, "rshift": 54,
    "alt": 56, "lalt": 56, "ralt": 100, "altgr": 100,
    "super": 125, "meta": 125, "win": 125, "cmd": 125,
    # letters
    **{c: code for c, code in zip(
        "abcdefghijklmnopqrstuvwxyz",
        [30, 48, 46, 32, 18, 33, 34, 35, 23, 36, 37, 38, 50, 49,
         24, 25, 16, 19, 31, 20, 22, 47, 17, 45, 21, 44],
    )},
    # digits
    "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9, "9": 10, "0": 11,
    # whitespace / control
    "enter": 28, "return": 28, "esc": 1, "escape": 1, "backspace": 14,
    "tab": 15, "space": 57, "delete": 111, "del": 111, "insert": 110,
    # navigation
    "home": 102, "end": 107, "pageup": 104, "pgup": 104,
    "pagedown": 109, "pgdn": 109,
    "up": 103, "down": 108, "left": 105, "right": 106,
    # punctuation
    "minus": 12, "equal": 13, "leftbrace": 26, "rightbrace": 27,
    "semicolon": 39, "apostrophe": 40, "grave": 41, "backslash": 43,
    "comma": 51, "dot": 52, "period": 52, "slash": 53,
    # function keys
    **{f"f{i}": code for i, code in zip(range(1, 13),
        [59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 87, 88])},
}

# Mouse button codes for `ydotool click` (down|up = 0xC0+button).
_CLICK = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
_DOWN = {"left": "0x40", "right": "0x41", "middle": "0x42"}
_UP = {"left": "0x80", "right": "0x81", "middle": "0x82"}


def ydotool_supports_absolute(ydotool: str | None = None) -> bool:
    """Whether ydotool is the modern v1.x CLI (daemon-based, with absolute mouse
    positioning) rather than the legacy v0.1.x (Debian/Ubuntu apt), whose CLI
    silently rejects ``mousemove --absolute`` while exiting 0.

    Detection is **daemon-independent**: we can't probe ``mousemove --help`` —
    v1.x tries to reach ydotoold first and prints a connection error instead of
    help when the daemon is down. Instead: v1.x ships ``ydotoold``, and its
    top-level usage advertises ``YDOTOOL_SOCKET`` and lists commands (``stdin``)
    absent from v0.1.x (which lists ``recorder`` instead).
    """
    yd = ydotool or shutil.which("ydotool")
    if not yd:
        return False
    # Only the modern daemon-based release ships ydotoold alongside the client.
    if shutil.which("ydotoold"):
        return True
    try:
        proc = subprocess.run([yd], capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    usage = (proc.stdout + proc.stderr).decode(errors="replace").lower()
    return "ydotool_socket" in usage or "stdin" in usage


class InputControlTool:
    """Control the real keyboard and mouse via ydotool (X11 + Wayland)."""

    name = "input_control"

    def __init__(
        self,
        *,
        verify: str = "on_failure",
        ydotool_socket: str | None = None,
    ) -> None:
        self._verify = verify
        self._socket = ydotool_socket
        self._modern: bool | None = None  # cached ydotool-version probe
        self._scale: float | None = None  # cached logical→physical scale factor

    # ── spec ───────────────────────────────────────────────────────
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "Click, type, move the mouse, scroll, and press keys to drive the "
                "REAL desktop — the physical keyboard and mouse — on Linux (X11 and "
                "Wayland) via ydotool. Operate any on-screen GUI application. HIGH "
                "RISK: actions affect the user's actual session.\n\n"
                "COORDINATES are FULL-SCREEN PIXELS (origin top-left, x right, y "
                "down). Call action='geometry' for the exact width×height. A target "
                "is 'x','y' (absolute pixels) or 'xpct','ypct' (percent of the "
                "screen, 0–100) for a rough location.\n"
                "PRECISION: the plain 'screenshot' is a DOWNSCALED thumbnail — you "
                "cannot read exact pixels off it. To click a precise point, 'zoom' "
                "into that area first: zoom returns a magnified view with a grid "
                "LABELLED in real screen coordinates. Read the exact x,y off the "
                "grid, then click with x:<value> y:<value>. Raise the 'zoom' factor "
                "for finer reading; this is how you reach any single pixel.\n\n"
                "VERIFICATION: after a failure or a timeout a screenshot of the "
                "current desktop is attached automatically so you can see the real "
                "state and recover. You may also pass verify=true on any action to "
                "force a post-action screenshot. In auto-approve mode you should "
                "verify each step and escalate risky/irreversible actions for "
                "approval.\n\n"
                "ACTIONS: 'geometry' (full screen size); 'screenshot' (downscaled "
                "overview); 'zoom' (magnified, coordinate-labelled view of an area — "
                "centre with xpct,ypct or x,y, plus 'zoom' factor); 'move' (x,y or "
                "xpct,ypct); 'move_rel' (x,y pixel deltas); 'click'/'right_click'/"
                "'middle_click'/'double_click' (optional x,y or xpct,ypct to move "
                "first); 'mouse_down'/'mouse_up' (button); 'scroll' (direction "
                "up|down, amount); 'type' (text, optional key_delay ms); 'key' (combo "
                "like 'ctrl+c', 'alt+Tab', 'super+l'); 'key_down'/'key_up' (single "
                "key).\n"
                "KEY NAMES: letters a-z, digits 0-9, modifiers ctrl/shift/alt/super, "
                "enter, esc, tab, space, backspace, delete, home, end, pageup, "
                "pagedown, up/down/left/right, f1-f12, and common punctuation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "geometry", "screenshot", "zoom", "move", "move_rel",
                            "click", "right_click", "middle_click", "double_click",
                            "mouse_down", "mouse_up", "scroll",
                            "type", "key", "key_down", "key_up",
                        ],
                    },
                    "x": {"type": "integer", "description": "X in full-screen pixels."},
                    "y": {"type": "integer", "description": "Y in full-screen pixels."},
                    "zoom": {"type": "number", "description": "Zoom magnification (default 8)."},
                    "xpct": {
                        "type": "number",
                        "description": "X as a percent of screen width, 0–100 "
                        "(rough location / zoom centre; use x for an exact pixel).",
                    },
                    "ypct": {
                        "type": "number",
                        "description": "Y as a percent of screen height, 0–100 "
                        "(rough location / zoom centre; use y for an exact pixel).",
                    },
                    "button": {
                        "type": "string", "enum": ["left", "right", "middle"],
                        "description": "Mouse button (default left).",
                    },
                    "text": {"type": "string", "description": "Text to type."},
                    "key": {"type": "string", "description": "Key or combo, e.g. 'ctrl+c'."},
                    "direction": {"type": "string", "enum": ["up", "down"]},
                    "amount": {"type": "integer", "description": "Scroll steps (default 3)."},
                    "key_delay": {"type": "integer", "description": "ms between keystrokes when typing."},
                    "verify": {"type": "boolean", "description": "Attach a screenshot of the result."},
                },
                "required": ["action"],
            },
        )

    # ── execution ──────────────────────────────────────────────────
    async def execute(self, arguments: dict[str, Any]) -> str:
        action = arguments.get("action")

        guard = self._guard(action)
        if guard is not None:
            return guard

        want = bool(arguments.get("verify")) or self._verify == "each_action"
        # Capture a "before" frame for actionable steps when verification is on,
        # so the agent can compare before/after and direct its next move.
        before = ""
        if want and action not in ("screenshot", "geometry"):
            tag, _r = host.screenshot_tag(self._get_scale())
            before = tag or ""

        try:
            result, failed = await self._dispatch(action, arguments)
        except asyncio.TimeoutError:
            return self._verified(f"[ERROR] {action} timed out", forced=True, before=before)
        except Exception as exc:  # noqa: BLE001
            logger.debug("input_control %s failed", action, exc_info=True)
            return self._verified(f"[ERROR] {action} failed: {exc}", forced=True, before=before)

        return self._verified(result, forced=failed, requested=want, before=before)

    def _candidate_sockets(self) -> list[str]:
        """Possible ydotoold socket paths, most-specific first."""
        if self._socket:
            return [self._socket]
        cands: list[str] = []
        runtime = os.environ.get("XDG_RUNTIME_DIR")
        if runtime:
            cands.append(os.path.join(runtime, ".ydotool_socket"))
        try:
            cands.append(f"/run/user/{os.getuid()}/.ydotool_socket")
        except AttributeError:  # no os.getuid() (non-POSIX) — ignore
            pass
        cands.append("/tmp/.ydotool_socket")
        # De-duplicate while preserving order.
        seen: set[str] = set()
        return [c for c in cands if not (c in seen or seen.add(c))]

    def _resolve_socket(self) -> str | None:
        """The first existing ydotoold socket, or None if the daemon isn't up."""
        for path in self._candidate_sockets():
            if path and os.path.exists(path):
                return path
        return None

    def _modern_ydotool(self) -> bool:
        """True if ydotool is the v1.x CLI (supports absolute positioning)."""
        if self._modern is None:
            self._modern = ydotool_supports_absolute()
        return self._modern

    def _get_scale(self) -> float:
        """The logical→physical scale factor (cached). 1.0 = no scaling."""
        if self._scale is not None:
            return self._scale
        if not host.pil_available():
            self._scale = 1.0  # can't resize the image → keep coords at native res
            return self._scale
        size, _reason = self._screen_size()
        if size is None:
            return 1.0  # don't cache a transient failure — retry next time
        self._scale = compute_scale(size[0], size[1])
        return self._scale

    def _abs_target(self, a: dict[str, Any]) -> tuple[int, int] | None:
        """Resolve an absolute pointer target to real hardware pixels.

        ``x``/``y`` are FULL-RESOLUTION screen pixels (1:1 — every pixel is
        addressable; read exact values off a ``zoom`` view). ``xpct``/``ypct``
        (percent of the screen, 0–100) give a rough, resolution-independent
        location. Returns ``None`` when no usable target is present.
        """
        if a.get("x") is not None and a.get("y") is not None:
            return (int(a["x"]), int(a["y"]))
        xp, yp = a.get("xpct"), a.get("ypct")
        if xp is not None and yp is not None:
            size, _reason = self._screen_size()
            if size is None:
                return None
            w, h = size
            return (
                max(0, min(w - 1, round(float(xp) / 100.0 * w))),
                max(0, min(h - 1, round(float(yp) / 100.0 * h))),
            )
        return None

    @staticmethod
    def _target_label(a: dict[str, Any]) -> str:
        """How the resolved target reads back to the model (its own units)."""
        if a.get("x") is not None and a.get("y") is not None:
            return f"({a.get('x')}, {a.get('y')})"
        if a.get("xpct") is not None and a.get("ypct") is not None:
            return f"x={a['xpct']}%, y={a['ypct']}%"
        return "(?, ?)"

    async def _zoom(self, a: dict[str, Any]) -> tuple[str, bool]:
        """A magnified, coordinate-labelled view of a screen region.

        Lets the model read off an exact pixel it cannot resolve in the
        downscaled overview, then click that pixel at full resolution.
        """
        if not host.pil_available():
            return "[ERROR] zoom needs Pillow (install the input-control extra)", True
        size, reason = self._screen_size()
        if size is None:
            return f"[ERROR] could not determine screen geometry: {reason}", True
        w, h = size
        factor = float(a.get("zoom") or 8)
        if a.get("x") is not None and a.get("y") is not None:
            cx, cy = int(a["x"]), int(a["y"])
        elif a.get("xpct") is not None and a.get("ypct") is not None:
            cx, cy = round(float(a["xpct"]) / 100.0 * w), round(float(a["ypct"]) / 100.0 * h)
        else:
            cx, cy = w // 2, h // 2
        x0, y0, rw, rh = host.zoom_region_box(w, h, cx, cy, factor)
        png, capture_reason = host.capture_png()
        if not png:
            return f"[ERROR] screenshot unavailable: {capture_reason}", True
        try:
            crop = host.crop_region(png, x0, y0, rw, rh)
            annotated = host.annotate_zoom(
                crop, x0=x0, y0=y0, magnify=_ZOOM_OUT_LONG_EDGE / max(rw, rh)
            )
        except Exception as exc:  # noqa: BLE001 — PIL/crop failure shouldn't crash the turn
            return f"[ERROR] zoom failed: {exc}", True
        tag = f"[screenshot:image/png;base64,{base64.b64encode(annotated).decode()}]"
        return (
            f"zoom of x[{x0}–{x0 + rw}] y[{y0}–{y0 + rh}] (full screen {w}x{h}). "
            "Read the labelled grid for the exact pixel, then click with x/y.\n"
            f"{tag}",
            False,
        )

    def _guard(self, action: str | None) -> str | None:
        # Read-only observation actions don't need ydotool.
        if action in ("screenshot", "geometry", "zoom"):
            return None
        if not shutil.which("ydotool"):
            sess = os.environ.get("XDG_SESSION_TYPE", "unknown")
            return (
                "[ERROR] ydotool not found. Run `leuk doctor` for exact per-distro "
                f"setup steps. Session type: {sess}."
            )
        if not self._modern_ydotool():
            # Legacy v0.1.x (Debian/Ubuntu apt): incompatible CLI — `--absolute`,
            # hex click codes and keycode syntax are all rejected while exiting 0,
            # so actions silently do nothing.
            return (
                "[ERROR] ydotool is too old (v0.1.x): it can't position the mouse "
                "absolutely, so actions silently do nothing. Install ydotool ≥ 1.0 — "
                "run `leuk doctor` or `bash scripts/setup-input-control.sh`."
            )
        # ydotool v1.x injects through the ydotoold daemon.
        if shutil.which("ydotoold") is None:
            return (
                "[ERROR] ydotoold (the ydotool daemon) isn't installed, but v1.x needs "
                "it. Install and start it — run `leuk doctor`."
            )
        if self._resolve_socket() is None:
            cands = ", ".join(self._candidate_sockets())
            return (
                "[ERROR] ydotoold is not running — no socket found (looked at: "
                f"{cands}). Start it (a user service or `ydotoold`), or run `leuk "
                "doctor`. Set LEUK_INPUT_CONTROL_YDOTOOL_SOCKET if it lives elsewhere."
            )
        return None

    async def _dispatch(self, action: str | None, a: dict[str, Any]) -> tuple[str, bool]:
        match action:
            case "geometry":
                size, reason = self._screen_size()
                if size is None:
                    return f"[ERROR] could not determine screen geometry: {reason}", True
                return (
                    f"screen: {size[0]}x{size[1]} px (full resolution, origin "
                    "top-left). Coordinates are absolute pixels in this space; "
                    "'zoom' into an area to read exact pixel values.",
                    False,
                )
            case "screenshot":
                tag, reason = host.screenshot_tag(self._get_scale())
                if tag is None:
                    return f"[ERROR] screenshot unavailable: {reason}", True
                return tag, False
            case "zoom":
                return await self._zoom(a)
            case "move":
                t = self._abs_target(a)
                if t is None:
                    return (
                        "[ERROR] move needs x,y (full-screen pixels) or "
                        "xpct,ypct (percent of screen)",
                        True,
                    )
                await self._yd("mousemove", "--absolute", "-x", str(t[0]), "-y", str(t[1]))
                return f"moved to {self._target_label(a)}", False
            case "move_rel":
                await self._yd("mousemove", "-x", str(int(a.get("x", 0))),
                               "-y", str(int(a.get("y", 0))))
                return f"moved by ({a.get('x')}, {a.get('y')})", False
            case "click" | "right_click" | "middle_click" | "double_click":
                btn = {"click": "left", "right_click": "right",
                       "middle_click": "middle", "double_click": "left"}[action]
                t = self._abs_target(a)
                if t is not None:
                    await self._yd("mousemove", "--absolute", "-x", str(t[0]), "-y", str(t[1]))
                await self._yd("click", _CLICK[btn])
                if action == "double_click":
                    await self._yd("click", _CLICK[btn])
                return f"{action}", False
            case "mouse_down":
                btn = a.get("button", "left")
                await self._yd("click", _DOWN.get(btn, _DOWN["left"]))
                return f"{btn} down", False
            case "mouse_up":
                btn = a.get("button", "left")
                await self._yd("click", _UP.get(btn, _UP["left"]))
                return f"{btn} up", False
            case "scroll":
                direction = a.get("direction", "down")
                amount = int(a.get("amount", 3))
                key = "pagedown" if direction == "down" else "pageup"
                for _ in range(max(1, amount)):
                    await self._key_combo(key)
                return f"scrolled {direction} x{amount}", False
            case "type":
                text = str(a.get("text", ""))
                args = ["type"]
                if a.get("key_delay"):
                    args += ["--key-delay", str(int(a["key_delay"]))]
                args.append(text)
                await self._yd(*args)
                return f"typed {len(text)} chars", False
            case "key":
                combo = str(a.get("key", ""))
                err = await self._key_combo(combo)
                return (err, True) if err else (f"key {combo}", False)
            case "key_down" | "key_up":
                code = KEYCODES.get(str(a.get("key", "")).lower())
                if code is None:
                    return f"[ERROR] unknown key {a.get('key')!r}", True
                await self._yd("key", f"{code}:{1 if action == 'key_down' else 0}")
                return f"{action} {a.get('key')}", False
            case _:
                return f"[ERROR] unknown action {action!r}", True

    async def _key_combo(self, combo: str) -> str | None:
        """Press a key/combo like 'ctrl+c'. Returns an error string or None."""
        parts = [p for p in combo.replace(" ", "").split("+") if p]
        codes: list[int] = []
        for p in parts:
            code = KEYCODES.get(p.lower())
            if code is None:
                return f"[ERROR] unknown key {p!r} in {combo!r}"
            codes.append(code)
        if not codes:
            return f"[ERROR] empty key combo {combo!r}"
        seq = [f"{c}:1" for c in codes] + [f"{c}:0" for c in reversed(codes)]
        await self._yd("key", *seq)
        return None

    async def _yd(self, *args: str) -> None:
        env = dict(os.environ)
        sock = self._resolve_socket()
        if sock:
            env["YDOTOOL_SOCKET"] = sock

        def _run() -> subprocess.CompletedProcess[bytes]:
            return subprocess.run(
                ["ydotool", *args], capture_output=True, timeout=_TIMEOUT, env=env, check=False
            )

        proc = await asyncio.to_thread(_run)
        stderr = proc.stderr.decode(errors="replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(stderr or "ydotool error")
        # ydotool frequently exits 0 even when it couldn't reach ydotoold (the
        # event is silently dropped). Surface those warnings as real failures so
        # the action doesn't report success while doing nothing.
        if stderr and any(
            marker in stderr.lower()
            for marker in ("failed to connect", "no such file", "connection refused", "socket")
        ):
            raise RuntimeError(stderr)

    def _screen_size(self) -> tuple[tuple[int, int] | None, str]:
        """Return the physical screen size ((w, h), "") or (None, reason)."""
        return host.screen_size()

    def _verified(
        self, result: str, *, forced: bool = False, requested: bool = False, before: str = ""
    ) -> str:
        """Append before/after verification screenshots to *result*.

        *before* (captured prior to the action) is shown labelled alongside the
        post-action capture, so the agent sees the effect of what it just did.
        """
        if self._verify == "never" and not requested:
            return result
        if not (forced or requested):
            return result
        after, _reason = host.screenshot_tag(self._get_scale())
        parts = [result]
        if before:
            parts.append(f"Before:\n{before}")
        if after and not result.startswith("[screenshot:"):
            parts.append(f"After:\n{after}" if before else after)
        return "\n".join(parts)
