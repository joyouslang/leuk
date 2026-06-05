"""Tests for the read-only monitoring tool."""

from __future__ import annotations

import pytest

from leuk import host
from leuk.tools.monitoring import MonitoringTool


class TestSpec:
    def test_actions(self):
        spec = MonitoringTool().spec
        assert spec.name == "monitoring"
        enum = spec.parameters["properties"]["action"]["enum"]
        assert set(enum) == {"screenshot", "geometry", "system_info"}


class TestExecute:
    @pytest.mark.asyncio
    async def test_system_info(self):
        out = await MonitoringTool().execute({"action": "system_info"})
        assert "OS:" in out and "CPU cores:" in out

    @pytest.mark.asyncio
    async def test_geometry(self, monkeypatch):
        monkeypatch.setattr(host, "pil_available", lambda: False)
        monkeypatch.setattr(host, "screen_size", lambda: ((1920, 1080), ""))
        out = await MonitoringTool().execute({"action": "geometry"})
        assert "1920x1080" in out

    @pytest.mark.asyncio
    async def test_screenshot(self, monkeypatch):
        monkeypatch.setattr(host, "screen_size", lambda: ((800, 600), ""))
        monkeypatch.setattr(host, "pil_available", lambda: False)
        monkeypatch.setattr(host, "screenshot_tag", lambda scale=1.0: ("[screenshot:image/png;base64,AA]", ""))
        out = await MonitoringTool().execute({"action": "screenshot"})
        assert out.startswith("[screenshot:")

    @pytest.mark.asyncio
    async def test_screenshot_failure(self, monkeypatch):
        monkeypatch.setattr(host, "screen_size", lambda: (None, "no backend"))
        monkeypatch.setattr(host, "screenshot_tag", lambda scale=1.0: (None, "no backend"))
        out = await MonitoringTool().execute({"action": "screenshot"})
        assert "[ERROR]" in out

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        out = await MonitoringTool().execute({"action": "nope"})
        assert "[ERROR]" in out
