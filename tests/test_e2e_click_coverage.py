"""End-to-end: the agent (leuk harness) can click *every pixel* at full
resolution (3840×2160), in the browser and via input_control.

Coordinates are full-resolution and map 1:1, so covering every x (at one y) and
every y (at one x) independently proves all 3840×2160 pixels are addressable
without an 8.3M-pixel loop. We also exercise the precise path the model uses:
``zoom`` into an area, then click an exact pixel.

* Browser — a real headless Chromium at a 3840×2160 viewport; clicks land on a
  recorder page at the exact CSS pixel.
* input_control — real hardware injection can't run in CI, so coordinates are
  captured at the ydotool boundary; x/y are absolute full-resolution pixels.

Both go through the real ``Agent`` so model→agent→tool→actuation is exercised.
"""

from __future__ import annotations

import io
import urllib.parse
from pathlib import Path

import pytest

from leuk import host
from leuk.agent.core import Agent
from leuk.config import AgentConfig, Settings, SQLiteConfig
from leuk.persistence.memory import MemoryStore
from leuk.persistence.sqlite import SQLiteStore
from leuk.tools.base import ToolRegistry
from leuk.types import Message, Role, ToolCall

from tests.conftest import MockProvider

W, H = 3840, 2160  # full 4K


async def _agent_with(
    tool, tmp_path: Path, *, responses: list[Message]
) -> tuple[Agent, SQLiteStore]:
    settings = Settings(
        sqlite=SQLiteConfig(path=str(tmp_path / f"e2e_{id(tool)}.db")),
        agent=AgentConfig(max_tool_rounds=4),
    )
    registry = ToolRegistry()
    registry.register(tool)
    sqlite = SQLiteStore(settings.sqlite)
    agent = Agent(
        settings=settings,
        provider=MockProvider(responses),
        tool_registry=registry,
        sqlite=sqlite,
        hot_store=MemoryStore(),
    )
    await agent.init()
    return agent, sqlite


def _click_call(tool_name: str, **args) -> list[Message]:
    return [
        Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="c", name=tool_name, arguments={"action": "click", **args})],
        ),
        Message(role=Role.ASSISTANT, content="done"),
    ]


# ── Browser: every pixel of a 4K viewport (real Chromium) ───────────────────


@pytest.mark.asyncio
async def test_e2e_browser_agent_clicks_every_pixel_at_4k(tmp_path: Path):
    from leuk.tools.browser import BrowserTool

    tool = BrowserTool(headless=True)
    try:
        try:
            page = await tool._ensure_page()
        except RuntimeError as exc:  # playwright/chromium not installed
            pytest.skip(f"browser unavailable: {exc}")

        await page.set_viewport_size({"width": W, "height": H})
        recorder = (
            "<!doctype html><body style='margin:0'>"
            "<script>window.__hits=[];"
            "document.addEventListener('click',e=>window.__hits.push([e.clientX,e.clientY]));"
            "</script>"
        )
        await page.goto("data:text/html," + urllib.parse.quote(recorder))

        # leuk-as-harness: the Agent drives a click on a precise 4K pixel.
        px, py = 3007, 1873
        agent, sqlite = await _agent_with(
            tool, tmp_path, responses=_click_call("browser", x=px, y=py)
        )
        try:
            async for _ in agent.run("click that pixel"):
                pass
        finally:
            await sqlite.close()  # not agent.shutdown(): that would close the browser
        assert [px, py] in await page.evaluate("window.__hits")

        # The precise path: zoom into the area, then click the exact pixel read off it.
        await page.evaluate("window.__hits=[]")
        await tool.execute({"action": "zoom", "x": px, "y": py, "zoom": 12})
        await tool.execute({"action": "click", "x": px, "y": py})
        assert await page.evaluate("window.__hits") == [[px, py]]

        # Every pixel: exhaustive per-axis over the full 4K range (1:1 mapping →
        # every x and every y independently ⇒ all W×H pixels addressable).
        await page.evaluate("window.__hits=[]")
        for x in range(W):
            await tool.execute({"action": "click", "x": x, "y": H // 2})
        for y in range(H):
            await tool.execute({"action": "click", "x": 0, "y": y})
        hits = await page.evaluate("window.__hits")
        assert hits[:W] == [[x, H // 2] for x in range(W)]
        assert hits[W:] == [[0, y] for y in range(H)]
    finally:
        await tool.close()


# ── input_control: every physical pixel of a 4K screen (ydotool boundary) ───


@pytest.mark.asyncio
async def test_e2e_input_control_agent_clicks_every_pixel_at_4k(tmp_path: Path, monkeypatch):
    from leuk.tools.input_control import InputControlTool

    tool = InputControlTool()
    tool._modern = True
    calls: list[tuple[str, ...]] = []

    async def _fake_yd(*args: str) -> None:
        calls.append(args)

    tool._yd = _fake_yd  # type: ignore[assignment]  # capture at the ydotool boundary
    tool._guard = lambda action: None  # type: ignore[assignment]
    tool._screen_size = lambda: ((W, H), "")  # type: ignore[assignment]

    # leuk-as-harness: the Agent drives a click on a precise 4K pixel via full-res x/y.
    px, py = 3007, 1873
    agent, sqlite = await _agent_with(
        tool, tmp_path, responses=_click_call("input_control", x=px, y=py)
    )
    try:
        async for _ in agent.run("click that pixel"):
            pass
    finally:
        await sqlite.close()
    assert ("mousemove", "--absolute", "-x", str(px), "-y", str(py)) in calls

    # The precise path: zoom into the area, then click the exact pixel read off it.
    monkeypatch.setattr(host, "pil_available", lambda: True)
    buf = io.BytesIO()
    from PIL import Image

    Image.new("RGB", (W, H), (0, 0, 0)).save(buf, "PNG")
    monkeypatch.setattr(host, "capture_png", lambda: (buf.getvalue(), ""))
    zoomed = await tool.execute({"action": "zoom", "x": px, "y": py, "zoom": 12})
    assert "[screenshot:image/png;base64," in zoomed
    calls.clear()
    await tool.execute({"action": "click", "x": px, "y": py})
    assert calls[0] == ("mousemove", "--absolute", "-x", str(px), "-y", str(py))

    # Every physical pixel: exhaustive per-axis over the full 4K range, x/y 1:1.
    for x in range(W):
        calls.clear()
        await tool.execute({"action": "click", "x": x, "y": 0})
        assert calls[0] == ("mousemove", "--absolute", "-x", str(x), "-y", "0")
    for y in range(H):
        calls.clear()
        await tool.execute({"action": "click", "x": 0, "y": y})
        assert calls[0] == ("mousemove", "--absolute", "-x", "0", "-y", str(y))
