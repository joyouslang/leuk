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


# ── fast hot path: targeting, short timeouts, capped settle ─────────────────


class _RecLoc:
    def __init__(self, page: "_RecPage", kind: str, count: int = 1) -> None:
        self._page, self._kind, self._count = page, kind, count

    @property
    def first(self) -> "_RecLoc":
        return self

    def filter(self, has_text=None) -> "_RecLoc":  # noqa: ANN001
        return self

    async def count(self) -> int:
        return self._count

    async def click(self, **k) -> None:
        if self._page.raise_on_action:
            raise RuntimeError("Locator.click: Timeout 6000ms exceeded.\nCall log:\n  - ...")
        self._page.actions.append(("click", self._kind, k.get("timeout")))

    async def hover(self, **k) -> None:
        self._page.actions.append(("hover", self._kind, k.get("timeout")))


class _RecPage:
    def __init__(self, clickable_count: int = 1, raise_on_action: bool = False) -> None:
        self.actions: list[tuple] = []
        self.settle_timeout: int | None = None
        self.url = "https://example.com"
        self._cc = clickable_count
        self.raise_on_action = raise_on_action

    def get_by_text(self, text: str, **_k) -> _RecLoc:
        return _RecLoc(self, "text")

    def locator(self, _sel: str) -> _RecLoc:
        return _RecLoc(self, "clickable", self._cc)

    async def wait_for_load_state(self, _state: str, timeout: int | None = None) -> None:
        self.settle_timeout = timeout


def _tool_rec(**kw) -> tuple[BrowserTool, _RecPage]:
    page = _RecPage(**{k: v for k, v in kw.items() if k in ("clickable_count", "raise_on_action")})
    tool = BrowserTool(timeout_ms=kw.get("timeout_ms", 6000), settle_ms=kw.get("settle_ms", 2500))
    tool._page = page
    return tool, page


@pytest.mark.asyncio
async def test_text_click_prefers_interactive_element():
    tool, page = _tool_rec(clickable_count=1)
    await tool.execute({"action": "click", "text": "Beginner"})
    assert page.actions == [("click", "clickable", 6000)]  # not the bare text match


@pytest.mark.asyncio
async def test_text_click_falls_back_when_no_interactive():
    tool, page = _tool_rec(clickable_count=0)
    await tool.execute({"action": "click", "text": "Beginner"})
    assert page.actions == [("click", "text", 6000)]


@pytest.mark.asyncio
async def test_action_uses_short_default_timeout_and_override():
    tool, page = _tool_rec(clickable_count=0)
    await tool.execute({"action": "click", "text": "x"})
    assert page.actions[-1][2] == 6000  # short default, not 15000
    await tool.execute({"action": "click", "text": "x", "timeout": 12000})
    assert page.actions[-1][2] == 12000  # per-call override still honoured


@pytest.mark.asyncio
async def test_settle_is_capped_short():
    tool, page = _tool_rec(clickable_count=0, settle_ms=1234)
    await tool.execute({"action": "click", "text": "x"})
    assert page.settle_timeout == 1234  # not the old 8000ms


@pytest.mark.asyncio
async def test_click_timeout_returns_fast_hint():
    tool, page = _tool_rec(clickable_count=0, raise_on_action=True)
    out = await tool.execute({"action": "click", "text": "Beginner"})
    assert out.startswith("[ERROR] click failed:")
    assert "xpct/ypct" in out  # points the model at the fast coordinate path
    assert "Call log" not in out  # verbose Playwright dump is trimmed
