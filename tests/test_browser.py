"""Tests for the browser tool's coordinate-click fallback (no Playwright needed).

A vision-driven model that can't produce a DOM selector can still act on what it
sees by clicking a percentage of the viewport; these tests inject a fake page so
no real browser is launched.
"""

from __future__ import annotations

import pytest

from leuk.tools.browser import BrowserTool


class _FakeMouse:
    def __init__(self) -> None:
        self.clicks: list[tuple[float, float]] = []
        self.moves: list[tuple[float, float]] = []

    async def click(self, x: float, y: float) -> None:
        self.clicks.append((x, y))

    async def move(self, x: float, y: float) -> None:
        self.moves.append((x, y))


class _FakePage:
    def __init__(self, w: int = 1280, h: int = 720) -> None:
        self.viewport_size = {"width": w, "height": h}
        self.mouse = _FakeMouse()
        self.url = "https://example.com"

    async def wait_for_load_state(self, *_a, **_k) -> None:  # _settle no-op
        pass


def _tool_with_page(w: int = 1280, h: int = 720) -> tuple[BrowserTool, _FakePage]:
    tool = BrowserTool()
    page = _FakePage(w, h)
    tool._page = page  # skip the real _ensure_page / Playwright launch
    return tool, page


@pytest.mark.asyncio
async def test_click_by_percent_maps_to_viewport():
    tool, page = _tool_with_page(1000, 800)
    out = await tool.execute({"action": "click", "xpct": 10, "ypct": 10})
    # 10% of 1000×800 → (100, 80) CSS pixels.
    assert page.mouse.clicks == [(100.0, 80.0)]
    assert "click at" in out


@pytest.mark.asyncio
async def test_click_by_pixels():
    tool, page = _tool_with_page()
    await tool.execute({"action": "click", "x": 42, "y": 99})
    assert page.mouse.clicks == [(42.0, 99.0)]


@pytest.mark.asyncio
async def test_hover_by_percent_moves_mouse():
    tool, page = _tool_with_page(2000, 1000)
    await tool.execute({"action": "hover", "xpct": 50, "ypct": 50})
    assert page.mouse.moves == [(1000.0, 500.0)]
    assert page.mouse.clicks == []


@pytest.mark.asyncio
async def test_click_without_target_errors_with_coord_hint():
    tool, page = _tool_with_page()
    out = await tool.execute({"action": "click"})
    assert "[ERROR]" in out
    assert "xpct" in out  # the error now points at the coordinate fallback
    assert page.mouse.clicks == []


@pytest.mark.asyncio
async def test_selector_path_still_used_when_given(monkeypatch):
    # A selector takes precedence over coordinates (existing behaviour preserved).
    tool, page = _tool_with_page()
    clicked: list[str] = []

    class _Loc:
        @property
        def first(self):
            return self

        async def click(self, **_k):
            clicked.append("css")

    page.locator = lambda sel: _Loc()  # type: ignore[attr-defined]
    out = await tool.execute({"action": "click", "selector": "#go", "xpct": 50, "ypct": 50})
    assert clicked == ["css"]  # used the selector, not the coordinates
    assert page.mouse.clicks == []
    assert out == "click ok"


def test_spec_documents_coordinate_params():
    spec = BrowserTool().spec
    for p in ("xpct", "ypct", "x", "y"):
        assert p in spec.parameters["properties"]
    assert "xpct" in spec.description
