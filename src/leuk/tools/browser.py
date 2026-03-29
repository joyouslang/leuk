"""Browser automation tool using Playwright."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any

from leuk.types import ToolSpec

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 64_000


class BrowserTool:
    """Control a headless Chromium browser for web automation tasks."""

    def __init__(self, *, headless: bool = True) -> None:
        self._headless = headless
        self._playwright: Any = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def _ensure_page(self) -> Page:
        if self._page is not None:
            return self._page

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "playwright is not installed. Install it with: pip install 'leuk[browser]'"
            )

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()
        logger.debug("Browser launched (headless=%s)", self._headless)
        return self._page

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser",
            description=(
                "Control a headless Chromium browser for web automation. "
                "Supports navigation, clicking, typing, screenshots, content extraction, "
                "and JavaScript evaluation. The browser persists across calls within a session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["navigate", "click", "type", "screenshot", "extract", "evaluate"],
                        "description": (
                            "Action to perform: "
                            "'navigate' — go to a URL; "
                            "'click' — click an element by CSS selector; "
                            "'type' — type text into an element by CSS selector; "
                            "'screenshot' — capture the current page as a base64 PNG; "
                            "'extract' — extract text from elements matching a CSS selector; "
                            "'evaluate' — run arbitrary JavaScript and return the result."
                        ),
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to (required for 'navigate').",
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector (required for 'click', 'type', 'extract').",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type (required for 'type').",
                    },
                    "js": {
                        "type": "string",
                        "description": "JavaScript expression to evaluate (required for 'evaluate').",
                    },
                },
                "required": ["action"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        action = arguments.get("action")

        if action == "navigate":
            return await self._navigate(arguments.get("url", ""))
        elif action == "click":
            return await self._click(arguments.get("selector", ""))
        elif action == "type":
            return await self._type(arguments.get("selector", ""), arguments.get("text", ""))
        elif action == "screenshot":
            return await self._screenshot()
        elif action == "extract":
            return await self._extract(arguments.get("selector", ""))
        elif action == "evaluate":
            return await self._evaluate(arguments.get("js", ""))
        else:
            return f"[ERROR] Unknown action: {action!r}"

    async def _navigate(self, url: str) -> str:
        if not url:
            return "[ERROR] 'url' is required for navigate"
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            page = await self._ensure_page()
            response = await page.goto(url, wait_until="domcontentloaded")
            status = response.status if response else "?"
            title = await page.title()
            return f"Navigated to {url} (status={status}, title={title!r})"
        except Exception as exc:
            return f"[ERROR] navigate failed: {exc}"

    async def _click(self, selector: str) -> str:
        if not selector:
            return "[ERROR] 'selector' is required for click"
        try:
            page = await self._ensure_page()
            await page.click(selector)
            return f"Clicked '{selector}'"
        except Exception as exc:
            return f"[ERROR] click failed: {exc}"

    async def _type(self, selector: str, text: str) -> str:
        if not selector:
            return "[ERROR] 'selector' is required for type"
        try:
            page = await self._ensure_page()
            await page.fill(selector, text)
            return f"Typed into '{selector}'"
        except Exception as exc:
            return f"[ERROR] type failed: {exc}"

    async def _screenshot(self) -> str:
        try:
            page = await self._ensure_page()
            png_bytes = await page.screenshot(type="png")
            b64 = base64.b64encode(png_bytes).decode()
            return f"[screenshot:image/png;base64,{b64}]"
        except Exception as exc:
            return f"[ERROR] screenshot failed: {exc}"

    async def _extract(self, selector: str) -> str:
        if not selector:
            return "[ERROR] 'selector' is required for extract"
        try:
            page = await self._ensure_page()
            elements = await page.query_selector_all(selector)
            if not elements:
                return f"[No elements matched selector: {selector!r}]"
            texts = []
            for el in elements:
                t = await el.inner_text()
                texts.append(t.strip())
            result = "\n\n".join(t for t in texts if t)
            if len(result) > _MAX_OUTPUT_CHARS:
                result = result[:_MAX_OUTPUT_CHARS] + "\n... [truncated]"
            return result
        except Exception as exc:
            return f"[ERROR] extract failed: {exc}"

    async def _evaluate(self, js: str) -> str:
        if not js:
            return "[ERROR] 'js' is required for evaluate"
        try:
            page = await self._ensure_page()
            result = await page.evaluate(js)
            text = str(result)
            if len(text) > _MAX_OUTPUT_CHARS:
                text = text[:_MAX_OUTPUT_CHARS] + "\n... [truncated]"
            return text
        except Exception as exc:
            return f"[ERROR] evaluate failed: {exc}"

    async def close(self) -> None:
        """Release browser resources."""
        if self._page is not None:
            try:
                await self._page.close()
            except Exception:
                pass
            self._page = None
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:
                pass
            self._context = None
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        logger.debug("Browser closed")
