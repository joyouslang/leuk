# Skill: /add-browser

Add Playwright-based browser automation to leuk, giving the agent interactive control
over a headless Chromium browser (navigate, click, type, screenshot, evaluate JS).

---

## Prerequisites

1. Playwright installed: `uv add playwright` then `playwright install chromium`.
2. No other code changes are required before this skill.

---

## Step 1 — Add optional dependency

Edit `pyproject.toml`: add a `[browser]` optional dependency group.

```toml
[project.optional-dependencies]
browser = [
    "playwright>=1.40",
]
```

Then run: `uv sync --extra browser`

---

## Step 2 — Add config section

File: `src/leuk/config.py`

Add a new settings model after `MCPServerConfig`:

```python
class BrowserConfig(BaseSettings):
    """Playwright browser tool configuration."""

    model_config = SettingsConfigDict(env_prefix="LEUK_BROWSER_", extra="ignore")

    enabled: bool = Field(default=False, description="Enable the browser tool")
    headless: bool = Field(default=True, description="Run browser headlessly")
```

Then add it to `Settings`:

```python
browser: BrowserConfig = Field(default_factory=BrowserConfig)
```

---

## Step 3 — Create the tool

Create `src/leuk/tools/browser.py`:

```python
"""Playwright browser automation tool."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

from leuk.types import ToolSpec

logger = logging.getLogger(__name__)

_ACTIONS = ["navigate", "click", "type", "screenshot", "extract", "evaluate", "close"]


class BrowserTool:
    """Interactive browser control via Playwright."""

    def __init__(self, headless: bool = True) -> None:
        self._headless = headless
        self._playwright = None
        self._browser = None
        self._page = None

    # ------------------------------------------------------------------
    # Tool protocol
    # ------------------------------------------------------------------

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser",
            description=(
                "Control a headless browser. Actions: navigate(url), click(selector), "
                "type(selector, text), screenshot(), extract(selector), "
                "evaluate(js), close(). Returns text or base64 PNG."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": _ACTIONS,
                        "description": "Browser action to perform",
                    },
                    "url": {"type": "string", "description": "URL for navigate action"},
                    "selector": {"type": "string", "description": "CSS selector"},
                    "text": {"type": "string", "description": "Text to type"},
                    "js": {"type": "string", "description": "JavaScript to evaluate"},
                },
                "required": ["action"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        action = arguments["action"]
        await self._ensure_browser()

        if action == "navigate":
            url = arguments.get("url", "")
            await self._page.goto(url, wait_until="domcontentloaded")
            return await self._page.title()

        if action == "click":
            await self._page.click(arguments["selector"])
            return "clicked"

        if action == "type":
            await self._page.fill(arguments["selector"], arguments.get("text", ""))
            return "typed"

        if action == "screenshot":
            png = await self._page.screenshot()
            return base64.b64encode(png).decode()

        if action == "extract":
            elements = await self._page.query_selector_all(arguments["selector"])
            texts = [await el.inner_text() for el in elements]
            return "\n".join(texts)

        if action == "evaluate":
            result = await self._page.evaluate(arguments["js"])
            return str(result)

        if action == "close":
            await self._close()
            return "browser closed"

        return f"unknown action: {action}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_browser(self) -> None:
        if self._page is not None:
            return
        from playwright.async_api import async_playwright
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        context = await self._browser.new_context()
        self._page = await context.new_page()

    async def _close(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._browser = None
        self._page = None
        self._playwright = None

    async def shutdown(self) -> None:
        """Call on agent shutdown to release browser resources."""
        await self._close()
```

---

## Step 4 — Register the tool

File: `src/leuk/tools/__init__.py`

In `create_default_registry()`, add conditional registration after loading settings:

```python
from leuk.config import load_settings

def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ShellTool())
    registry.register(FileReadTool())
    registry.register(FileEditTool())
    registry.register(SubAgentTool())
    registry.register(WebFetchTool())

    settings = load_settings()
    if settings.browser.enabled:
        from leuk.tools.browser import BrowserTool
        registry.register(BrowserTool(headless=settings.browser.headless))

    return registry
```

---

## Step 5 — Enable the tool

```bash
# Option A: environment variable
export LEUK_BROWSER_ENABLED=true

# Option B: config.env
echo "LEUK_BROWSER_ENABLED=true" >> ~/.config/leuk/config.env
```

---

## Step 6 — Verification

```bash
# Confirm Playwright browsers are installed
playwright install chromium --dry-run

# Start leuk and ask it to use the browser
leuk
# > navigate to https://example.com and screenshot it
```

The agent should call `browser(action="navigate", url="https://example.com")` then
`browser(action="screenshot")`.

---

## Notes

- The browser is lazy-initialized on first use — startup is not affected.
- A single browser instance is shared for the session. If isolation is needed, the
  `new_context()` call in `_ensure_browser` can be moved to per-request.
- Call `BrowserTool.shutdown()` in `AgentSession` cleanup (see
  `src/leuk/agent/session.py`) to ensure the Chromium process is terminated.
- `screenshot()` returns base64 PNG; providers with vision support (Anthropic, GPT-4o,
  Gemini) can reason about it directly.
