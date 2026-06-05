"""Tests for the input_control (desktop keyboard/mouse) tool."""

from __future__ import annotations

import struct
import zlib

import pytest

from leuk import host
from leuk.tools.input_control import KEYCODES, InputControlTool


def _fake_png(width: int, height: int) -> bytes:
    """Minimal valid PNG header (IHDR) so png_size can read dimensions."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + struct.pack(
        ">I", zlib.crc32(b"IHDR" + ihdr)
    )
    return sig + chunk


class TestKeymap:
    def test_modifiers_and_letters(self):
        assert KEYCODES["ctrl"] == 29
        assert KEYCODES["c"] == 46
        assert KEYCODES["enter"] == 28
        assert KEYCODES["super"] == 125

    def test_function_keys(self):
        assert KEYCODES["f1"] == 59
        assert KEYCODES["f12"] == 88


class TestSpec:
    def test_spec_actions(self):
        spec = InputControlTool().spec
        assert spec.name == "input_control"
        enum = spec.parameters["properties"]["action"]["enum"]
        for a in ("move", "click", "type", "key", "geometry", "screenshot", "scroll"):
            assert a in enum


class TestKeyCombo:
    @pytest.mark.asyncio
    async def test_combo_emits_press_then_release_reversed(self, monkeypatch):
        tool = InputControlTool()
        captured: list[tuple[str, ...]] = []

        async def _fake_yd(*args):
            captured.append(args)

        monkeypatch.setattr(tool, "_yd", _fake_yd)
        err = await tool._key_combo("ctrl+c")
        assert err is None
        # 29:1 46:1 46:0 29:0
        assert captured == [("key", "29:1", "46:1", "46:0", "29:0")]

    @pytest.mark.asyncio
    async def test_unknown_key(self, monkeypatch):
        tool = InputControlTool()
        monkeypatch.setattr(tool, "_yd", lambda *a: None)
        err = await tool._key_combo("ctrl+@")
        assert err is not None and "unknown key" in err


class TestGuards:
    @pytest.mark.asyncio
    async def test_missing_ydotool(self, monkeypatch):
        import leuk.tools.input_control as ic

        monkeypatch.setattr(ic.shutil, "which", lambda name: None)
        out = await InputControlTool().execute({"action": "move", "x": 1, "y": 1})
        assert "[ERROR]" in out and "ydotool" in out

    @pytest.mark.asyncio
    async def test_screenshot_not_guarded_by_ydotool(self, monkeypatch):
        import leuk.tools.input_control as ic

        monkeypatch.setattr(ic.shutil, "which", lambda name: None)
        monkeypatch.setattr(host, "capture_png", lambda: (_fake_png(800, 600), ""))
        out = await InputControlTool().execute({"action": "screenshot"})
        assert out.startswith("[screenshot:image/png;base64,")


class TestActions:
    @pytest.mark.asyncio
    async def test_geometry(self, monkeypatch):
        import sys

        import leuk.tools.input_control as ic

        monkeypatch.setattr(ic.shutil, "which", lambda name: "/usr/bin/ydotool")
        # Force the screenshot-dimension fallback by making `mss` unavailable, so
        # the test is independent of the real monitor's resolution. Disable
        # downscaling here so it asserts the raw read (scaling has its own test).
        monkeypatch.setitem(sys.modules, "mss", None)
        monkeypatch.setattr(host, "pil_available", lambda: False)
        monkeypatch.setattr(host, "capture_png", lambda: (_fake_png(1920, 1080), ""))
        out = await InputControlTool().execute({"action": "geometry"})
        assert "1920x1080" in out

    @pytest.mark.asyncio
    async def test_move_invokes_ydotool(self, monkeypatch):
        import leuk.tools.input_control as ic

        monkeypatch.setattr(ic.shutil, "which", lambda name: "/usr/bin/ydotool")
        tool = InputControlTool()
        tool._modern = True  # ydotool >= 1.0 (skip the version probe)
        tool._scale = 1.0  # pin scaling off so the coords are exact
        # Pretend ydotoold is up so the daemon guard passes.
        monkeypatch.setattr(tool, "_resolve_socket", lambda: "/run/user/1000/.ydotool_socket")
        calls: list[tuple[str, ...]] = []

        async def _fake_yd(*args):
            calls.append(args)

        monkeypatch.setattr(tool, "_yd", _fake_yd)
        out = await tool.execute({"action": "move", "x": 100, "y": 200})
        assert "moved to (100, 200)" in out
        assert calls == [("mousemove", "--absolute", "-x", "100", "-y", "200")]

    @pytest.mark.asyncio
    async def test_move_scales_coordinates_to_pixels(self, monkeypatch):
        """On a HiDPI screen the model's logical coords are scaled up to real
        pixels before reaching ydotool (scale 0.5 → double the coordinate)."""
        import leuk.tools.input_control as ic

        monkeypatch.setattr(ic.shutil, "which", lambda name: "/usr/bin/ydotool")
        tool = InputControlTool()
        tool._modern = True
        tool._scale = 0.5  # logical→physical = /0.5 = ×2
        monkeypatch.setattr(tool, "_resolve_socket", lambda: "/run/user/1000/.ydotool_socket")
        calls: list[tuple[str, ...]] = []

        async def _fake_yd(*args):
            calls.append(args)

        monkeypatch.setattr(tool, "_yd", _fake_yd)
        await tool.execute({"action": "click", "x": 100, "y": 200})
        # mousemove to the SCALED pixel (200, 400), then a left click.
        assert calls[0] == ("mousemove", "--absolute", "-x", "200", "-y", "400")
        assert calls[1] == ("click", "0xC0")

    @pytest.mark.asyncio
    async def test_geometry_reports_scaled_resolution(self, monkeypatch):
        """With Pillow present, geometry reports the downscaled (model-facing)
        size, not the raw 4K resolution."""
        import leuk.tools.input_control as ic

        monkeypatch.setattr(ic.shutil, "which", lambda name: "/usr/bin/ydotool")
        monkeypatch.setattr(host, "pil_available", lambda: True)
        tool = InputControlTool()
        monkeypatch.setattr(tool, "_screen_size", lambda: ((3840, 2160), ""))
        out = await tool.execute({"action": "geometry"})
        assert "1366x768" in out  # 3840→1366 long edge, 2160→768
        assert tool._scale == 1366 / 3840  # cached for subsequent coordinate calls

    @pytest.mark.asyncio
    async def test_verify_attaches_screenshot_on_failure(self, monkeypatch):
        import leuk.tools.input_control as ic

        monkeypatch.setattr(ic.shutil, "which", lambda name: "/usr/bin/ydotool")
        monkeypatch.setattr(host, "capture_png", lambda: (_fake_png(640, 480), ""))
        tool = InputControlTool(verify="on_failure")
        tool._modern = True
        tool._scale = 1.0
        monkeypatch.setattr(tool, "_resolve_socket", lambda: "/run/user/1000/.ydotool_socket")

        async def _boom(*args):
            raise RuntimeError("ydotool blew up")

        monkeypatch.setattr(tool, "_yd", _boom)
        out = await tool.execute({"action": "click"})
        assert "[ERROR]" in out
        assert "[screenshot:image/png;base64," in out

    @pytest.mark.asyncio
    async def test_old_ydotool_is_error(self, monkeypatch):
        """Legacy ydotool 0.1.x (no absolute positioning) → a clear error."""
        import leuk.tools.input_control as ic

        monkeypatch.setattr(ic.shutil, "which", lambda name: "/usr/bin/ydotool")
        monkeypatch.setattr(ic, "ydotool_supports_absolute", lambda path=None: False)
        out = await InputControlTool().execute({"action": "move", "x": 1, "y": 1})
        assert "[ERROR]" in out
        assert "too old" in out

    @pytest.mark.asyncio
    async def test_ydotoold_missing_is_error(self, monkeypatch):
        """Modern ydotool but no ydotoold installed → a clear error."""
        import leuk.tools.input_control as ic

        monkeypatch.setattr(
            ic.shutil, "which", lambda name: "/usr/bin/ydotool" if name == "ydotool" else None
        )
        tool = InputControlTool()
        tool._modern = True
        out = await tool.execute({"action": "move", "x": 1, "y": 1})
        assert "[ERROR]" in out
        assert "ydotoold" in out

    @pytest.mark.asyncio
    async def test_daemon_not_running_is_error(self, monkeypatch):
        """ydotoold installed but not running → a clear error, not silent success."""
        import leuk.tools.input_control as ic

        monkeypatch.setattr(ic.shutil, "which", lambda name: "/usr/bin/ydotool")
        tool = InputControlTool()
        tool._modern = True
        monkeypatch.setattr(tool, "_resolve_socket", lambda: None)  # no socket
        out = await tool.execute({"action": "move", "x": 1, "y": 1})
        assert "[ERROR]" in out
        assert "ydotoold is not running" in out

    @pytest.mark.asyncio
    async def test_exit_zero_connection_failure_is_error(self, monkeypatch):
        """ydotool exiting 0 while warning it couldn't reach ydotoold must be
        treated as a failure (the event was silently dropped)."""
        import subprocess

        import leuk.tools.input_control as ic

        monkeypatch.setattr(ic.shutil, "which", lambda name: "/usr/bin/ydotool")
        monkeypatch.setattr(host, "capture_png", lambda: (_fake_png(640, 480), ""))
        tool = InputControlTool()
        tool._modern = True
        tool._scale = 1.0
        monkeypatch.setattr(tool, "_resolve_socket", lambda: "/run/user/1000/.ydotool_socket")

        def _fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(
                args, 0, stdout=b"", stderr=b"failed to connect socket: No such file"
            )

        monkeypatch.setattr(ic.subprocess, "run", _fake_run)
        out = await tool.execute({"action": "move", "x": 5, "y": 5})
        assert "[ERROR]" in out
        assert "failed to connect" in out

    def test_supports_absolute_via_ydotoold_present(self, monkeypatch):
        """ydotoold on PATH is the modern-release signal (daemon-independent)."""
        import leuk.tools.input_control as ic

        monkeypatch.setattr(ic.shutil, "which", lambda name: "/usr/local/bin/" + name)
        assert ic.ydotool_supports_absolute("/usr/local/bin/ydotool") is True

    def test_supports_absolute_via_usage_probe(self, monkeypatch):
        """Without ydotoold, fall back to the top-level usage text — v1.x mentions
        YDOTOOL_SOCKET / has a `stdin` command; v0.1.x lists `recorder` instead."""
        import subprocess

        import leuk.tools.input_control as ic

        # ydotool present, ydotoold absent → usage probe decides.
        monkeypatch.setattr(
            ic.shutil, "which", lambda name: "/usr/bin/ydotool" if name == "ydotool" else None
        )

        def _modern(*args, **kwargs):
            return subprocess.CompletedProcess(
                args, 0, b"Available commands:\n click\n stdin\nUse YDOTOOL_SOCKET", b""
            )

        monkeypatch.setattr(ic.subprocess, "run", _modern)
        assert ic.ydotool_supports_absolute("/usr/bin/ydotool") is True

        def _legacy(*args, **kwargs):
            return subprocess.CompletedProcess(
                args, 0, b"Available commands:\n type\n recorder\n mousemove\n key\n click", b""
            )

        monkeypatch.setattr(ic.subprocess, "run", _legacy)
        assert ic.ydotool_supports_absolute("/usr/bin/ydotool") is False
