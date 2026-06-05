"""Tests for leuk.host: screen-capture scaling, geometry, and system info."""

from __future__ import annotations

import pytest

from leuk import host


class TestCoordinateScaling:
    def test_compute_scale_downscales_4k(self):
        assert host.compute_scale(3840, 2160) == 1366 / 3840

    def test_compute_scale_no_upscale_for_small_screens(self):
        assert host.compute_scale(1366, 768) == 1.0
        assert host.compute_scale(1024, 768) == 1.0

    def test_compute_scale_uses_long_edge(self):
        assert host.compute_scale(1080, 1920) == 1366 / 1920

    def test_compute_scale_handles_zero(self):
        assert host.compute_scale(0, 0) == 1.0

    def test_to_physical_roundtrips_logical_coords(self):
        assert host.to_physical(100, 0.5) == 200
        assert host.to_physical(100, 1.0) == 100
        assert host.to_physical(486, 1366 / 3840) == 1366


class TestDownscale:
    def test_resizes_when_scaling(self):
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        from leuk.media import png_size

        buf = io.BytesIO()
        Image.new("RGB", (3840, 2160), "blue").save(buf, format="PNG")
        out = host.downscale_png(buf.getvalue(), 1366 / 3840)
        assert png_size(out) == (1366, 768)

    def test_noop_at_scale_one(self):
        png = b"original-bytes"
        assert host.downscale_png(png, 1.0) is png

    def test_survives_bad_image(self):
        junk = b"\x89PNG\r\n\x1a\nnot-a-real-image"
        assert host.downscale_png(junk, 0.5) == junk


class TestScreenSizeAndTag:
    def test_screen_size_from_capture_fallback(self, monkeypatch):
        import struct
        import zlib

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", 1920, 1080, 8, 2, 0, 0, 0)
        png = sig + struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + struct.pack(
            ">I", zlib.crc32(b"IHDR" + ihdr)
        )
        import sys

        monkeypatch.setitem(sys.modules, "mss", None)  # force the capture fallback
        monkeypatch.setattr(host, "capture_png", lambda: (png, ""))
        size, reason = host.screen_size()
        assert size == (1920, 1080)

    def test_screenshot_tag_reports_failure(self, monkeypatch):
        monkeypatch.setattr(host, "capture_png", lambda: (None, "no backend"))
        tag, reason = host.screenshot_tag()
        assert tag is None and "no backend" in reason


class TestSystemInfo:
    def test_reports_core_facts(self):
        info = host.system_info()
        assert "OS:" in info
        assert "Hostname:" in info
        assert "CPU cores:" in info
        assert "Disk /:" in info
