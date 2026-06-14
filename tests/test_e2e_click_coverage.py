"""End-to-end: the agent (leuk harness) can click *every pixel*.

Two surfaces:

* Browser — a real headless Chromium. The agent drives a click through the
  tool, and every pixel of a viewport is clicked and lands exactly (CSS pixels
  are 1:1; percentages reach them too).
* input_control (any GUI app) — real hardware injection can't run in CI, so we
  capture the coordinates at the ydotool boundary. Because the desktop
  screenshot is downscaled, only *percentage* targets can address every physical
  pixel; we prove every one resolves exactly, at the test resolution and at real
  screen widths.

Both go through the real ``Agent`` so the model→agent→tool→actuation path is
exercised, not just the tool in isolation.
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path

import pytest

from leuk.agent.core import Agent
from leuk.config import AgentConfig, Settings, SQLiteConfig
from leuk.persistence.memory import MemoryStore
from leuk.persistence.sqlite import SQLiteStore
from leuk.tools.base import ToolRegistry
from leuk.types import Message, Role, ToolCall

from tests.conftest import MockProvider


async def _agent_with(tool, tmp_path: Path, *, responses: list[Message]) -> tuple[Agent, SQLiteStore]:
    """A real Agent (leuk harness) whose only tool is *tool*."""
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


# ── Browser: every viewport pixel (real Chromium) ───────────────────────────


@pytest.mark.asyncio
async def test_e2e_browser_agent_clicks_every_viewport_pixel(tmp_path: Path):
    from leuk.tools.browser import BrowserTool

    tool = BrowserTool(headless=True)
    try:
        try:
            page = await tool._ensure_page()
        except RuntimeError as exc:  # playwright/chromium not installed
            pytest.skip(f"browser unavailable: {exc}")

        W, H = 40, 30
        await page.set_viewport_size({"width": W, "height": H})
        recorder = (
            "<!doctype html><body style='margin:0'>"
            "<script>window.__hits=[];"
            "document.addEventListener('click',e=>window.__hits.push([e.clientX,e.clientY]));"
            "</script>"
        )
        await page.goto("data:text/html," + urllib.parse.quote(recorder))

        # leuk-as-harness: the Agent drives one real click through the tool.
        cx, cy = W // 2, H // 2
        agent, sqlite = await _agent_with(tool, tmp_path, responses=_click_call("browser", x=cx, y=cy))
        try:
            async for _ in agent.run("click the centre"):
                pass
        finally:
            await sqlite.close()  # not agent.shutdown(): that would close the browser
        assert [cx, cy] in await page.evaluate("window.__hits")

        # Every single pixel of the viewport, clicked exactly (CSS pixels, 1:1).
        await page.evaluate("window.__hits=[]")
        expected = [[x, y] for x in range(W) for y in range(H)]
        for x, y in expected:
            await tool.execute({"action": "click", "x": x, "y": y})
        assert await page.evaluate("window.__hits") == expected

        # Percentages reach the extremes and centre too (resolution-independent).
        await page.evaluate("window.__hits=[]")
        corners = [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1), (W // 2, H // 2)]
        for x, y in corners:
            await tool.execute(
                {"action": "click", "xpct": 100 * x / W, "ypct": 100 * y / H}
            )
        assert await page.evaluate("window.__hits") == [[x, y] for x, y in corners]
    finally:
        await tool.close()


# ── input_control: every physical screen pixel (ydotool boundary) ───────────


@pytest.mark.asyncio
async def test_e2e_input_control_agent_reaches_every_screen_pixel(tmp_path: Path):
    from leuk.tools.input_control import InputControlTool

    tool = InputControlTool()
    tool._modern = True
    calls: list[tuple[str, ...]] = []

    async def _fake_yd(*args: str) -> None:
        calls.append(args)

    tool._yd = _fake_yd  # type: ignore[assignment]  # capture at the ydotool boundary
    tool._guard = lambda action: None  # type: ignore[assignment]  # skip ydotool presence checks

    W, H = 64, 48
    tool._screen_size = lambda: ((W, H), "")  # type: ignore[assignment]

    # leuk-as-harness: the Agent drives a percentage click to the physical centre.
    agent, sqlite = await _agent_with(
        tool, tmp_path, responses=_click_call("input_control", xpct=50, ypct=50)
    )
    try:
        async for _ in agent.run("click the centre"):
            pass
    finally:
        await sqlite.close()
    assert ("mousemove", "--absolute", "-x", str(W // 2), "-y", str(H // 2)) in calls

    # Every single physical pixel is addressable via a percentage target.
    for px in range(W):
        for py in range(H):
            calls.clear()
            await tool.execute({"action": "click", "xpct": 100 * px / W, "ypct": 100 * py / H})
            assert calls[0] == ("mousemove", "--absolute", "-x", str(px), "-y", str(py))

    # And at real screen widths: every x-pixel round-trips exactly through the
    # percentage→physical mapping (the desktop screenshot is downscaled, so the
    # raw x/y path could not reach all of these — percentages can).
    for sw in (1366, 1920, 2560, 3840):
        tool._screen_size = lambda sw=sw: ((sw, 4), "")  # type: ignore[assignment]
        for px in range(sw):
            target = tool._abs_target({"xpct": 100 * px / sw, "ypct": 0})
            assert target is not None and target[0] == px
