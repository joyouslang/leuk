"""Browser automation tool using Playwright."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any

from leuk import host
from leuk.types import ToolSpec

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 64_000
# Long edge (px) of the magnified, coordinate-labelled image 'zoom' returns.
_ZOOM_OUT_LONG_EDGE = 1024

# Interactive elements a click/hover-by-text should prefer, so a bare word like
# "Beginner" lands on the actual button/link rather than a heading that merely
# contains it.
_CLICKABLE = (
    "a, button, [role=button], [role=link], [role=menuitem], [role=tab], "
    "[role=option], [role=checkbox], [role=radio], input[type=button], "
    "input[type=submit], input[type=checkbox], input[type=radio], [onclick]"
)


class BrowserTool:
    """Control a Chromium browser for web automation tasks.

    The browser window is **visible** by default so the user can watch what the
    agent does; pass ``headless=True`` for headless servers / explicit opt-in.
    """

    def __init__(
        self, *, headless: bool = False, timeout_ms: int = 6000, settle_ms: int = 2500
    ) -> None:
        self._headless = headless
        self._timeout = timeout_ms
        self._settle_ms = settle_ms
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
                "Drive a Chromium browser for complex, dynamic web apps (SPAs, AJAX, "
                "client-side routing). The page persists across calls. Targeting is "
                "flexible: pass a CSS 'selector' OR a robust descriptor — 'role'+'name' "
                "(accessibility), 'text', 'label', or 'placeholder' — which survive "
                "re-renders and hashed class names. For 'click'/'hover', when no good "
                "selector exists, target by POSITION instead: 'x'/'y' (CSS pixels, "
                "full viewport resolution) or 'xpct'/'ypct' (percent of the viewport, "
                "0–100) for a rough spot. To pick an exact pixel you can't resolve in "
                "the screenshot, 'zoom' into the area first: it returns a magnified "
                "view with a grid LABELLED in CSS coordinates — read the exact x,y off "
                "it, then click x:<value> y:<value>. Every action auto-waits for the "
                "element and the page to settle (network idle), so you rarely need "
                "explicit waits. Use 'read_page' to get a structured accessibility "
                "snapshot of the current state, and 'find' to discover targets.\n"
                "Actions: navigate, read_page, find, click, fill, type, press, hover, "
                "select, check, uncheck, scroll, wait_for, wait_for_network_idle, "
                "go_back, go_forward, reload, get_url, get_title, screenshot, zoom, "
                "extract, evaluate, upload."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "navigate", "read_page", "find", "click", "fill", "type",
                            "press", "hover", "select", "check", "uncheck", "scroll",
                            "wait_for", "wait_for_network_idle", "go_back", "go_forward",
                            "reload", "get_url", "get_title", "screenshot", "zoom",
                            "extract", "evaluate", "upload",
                        ],
                    },
                    "url": {"type": "string", "description": "URL (for 'navigate')."},
                    "selector": {"type": "string", "description": "CSS selector (target)."},
                    "role": {"type": "string", "description": "ARIA role, e.g. 'button' (target)."},
                    "name": {"type": "string", "description": "Accessible name for role (target)."},
                    "text": {"type": "string", "description": "Visible text (target) or text to type."},
                    "label": {"type": "string", "description": "Form label (target)."},
                    "placeholder": {"type": "string", "description": "Input placeholder (target)."},
                    "value": {"type": "string", "description": "Value to fill/select."},
                    "xpct": {"type": "number", "description": "X as % of viewport (0-100)."},
                    "ypct": {"type": "number", "description": "Y as % of viewport (0-100)."},
                    "x": {"type": "number", "description": "X in CSS pixels (alt to xpct)."},
                    "y": {"type": "number", "description": "Y in CSS pixels (alt to ypct)."},
                    "zoom": {"type": "number", "description": "Zoom magnification (default 8)."},
                    "key": {"type": "string", "description": "Key for 'press', e.g. 'Enter', 'Control+a'."},
                    "state": {
                        "type": "string",
                        "description": "wait_for state: 'visible'|'hidden'|'attached'|'load'|'domcontentloaded'|'networkidle'.",
                    },
                    "timeout": {"type": "integer", "description": "Per-action timeout in ms (default ~6000); raise it for a slow element."},
                    "path": {"type": "string", "description": "File path for 'upload'."},
                    "js": {"type": "string", "description": "JavaScript for 'evaluate'."},
                },
                "required": ["action"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        action = arguments.get("action")
        try:
            match action:
                case "navigate":
                    return await self._navigate(arguments.get("url", ""))
                case "read_page":
                    return await self._read_page()
                case "find":
                    return await self._find(arguments)
                case "click":
                    return await self._do(arguments, "click")
                case "hover":
                    return await self._do(arguments, "hover")
                case "check":
                    return await self._do(arguments, "check")
                case "uncheck":
                    return await self._do(arguments, "uncheck")
                case "fill":
                    return await self._fill(arguments)
                case "type":
                    return await self._fill(arguments)  # alias: clear + type
                case "press":
                    return await self._press(arguments)
                case "select":
                    return await self._select(arguments)
                case "upload":
                    return await self._upload(arguments)
                case "scroll":
                    return await self._scroll(arguments)
                case "wait_for":
                    return await self._wait_for(arguments)
                case "wait_for_network_idle":
                    return await self._wait_network_idle(arguments)
                case "go_back":
                    return await self._history("back")
                case "go_forward":
                    return await self._history("forward")
                case "reload":
                    return await self._history("reload")
                case "get_url":
                    page = await self._ensure_page()
                    return page.url
                case "get_title":
                    page = await self._ensure_page()
                    return await page.title()
                case "screenshot":
                    return await self._screenshot()
                case "zoom":
                    return await self._zoom(arguments)
                case "extract":
                    return await self._extract(arguments.get("selector", ""))
                case "evaluate":
                    return await self._evaluate(arguments.get("js", ""))
                case _:
                    return f"[ERROR] Unknown action: {action!r}"
        except Exception as exc:  # noqa: BLE001
            return f"[ERROR] {action} failed: {exc}"

    # ── targeting + settle helpers ────────────────────────────────
    def _locator(self, page: "Page", a: dict[str, Any]):  # noqa: ANN202
        """Resolve a Playwright Locator from a CSS selector or a descriptor."""
        if a.get("selector"):
            return page.locator(a["selector"])
        if a.get("role"):
            return page.get_by_role(a["role"], name=a.get("name"))
        if a.get("label"):
            return page.get_by_label(a["label"])
        if a.get("placeholder"):
            return page.get_by_placeholder(a["placeholder"])
        if a.get("text"):
            return page.get_by_text(a["text"])
        return None

    def _coord_target(self, page: "Page", a: dict[str, Any]) -> tuple[float, float] | None:
        """Resolve a click/hover point from xpct/ypct (percent of viewport) or x/y
        (CSS pixels). Percentages are resolution-independent, so they survive the
        screenshot being scaled down for the model. Returns None if absent."""
        xp, yp = a.get("xpct"), a.get("ypct")
        if xp is not None and yp is not None:
            vp = page.viewport_size or {"width": 1280, "height": 720}
            return (float(xp) / 100.0 * vp["width"], float(yp) / 100.0 * vp["height"])
        if a.get("x") is not None and a.get("y") is not None:
            return (float(a["x"]), float(a["y"]))
        return None

    async def _settle(self, page: "Page", timeout: int | None = None) -> None:
        """Best-effort wait for AJAX/SPA updates to quiesce.

        Capped short: many pages (ads, websockets, polling) never reach
        networkidle, so a long wait here would stall every single action.
        """
        try:
            await page.wait_for_load_state(
                "networkidle", timeout=self._settle_ms if timeout is None else timeout
            )
        except Exception:  # noqa: BLE001 — networkidle may never fire on some apps
            pass

    async def _do(self, a: dict[str, Any], op: str) -> str:
        page = await self._ensure_page()
        loc = self._locator(page, a)
        if loc is None:
            # Coordinate fallback for click/hover: lets a vision-driven model that
            # can't produce a selector still act on what it sees in the screenshot.
            if op in ("click", "hover"):
                pt = self._coord_target(page, a)
                if pt is not None:
                    if op == "click":
                        await page.mouse.click(pt[0], pt[1])
                    else:
                        await page.mouse.move(pt[0], pt[1])
                    await self._settle(page)
                    return f"{op} at ({round(pt[0])}, {round(pt[1])})"
                return (
                    "[ERROR] provide selector|role+name|text|label|placeholder, "
                    "or xpct,ypct (percent of viewport) / x,y (CSS pixels)"
                )
            return "[ERROR] provide selector|role+name|text|label|placeholder"
        timeout = int(a.get("timeout", self._timeout))
        target = loc.first
        # Click/hover by bare text: prefer an actually-interactive element so we
        # don't land on (and time out waiting for) a heading/paragraph that merely
        # contains the word.
        if op in ("click", "hover") and a.get("text") and not (
            a.get("selector") or a.get("role") or a.get("label") or a.get("placeholder")
        ):
            try:
                clickable = page.locator(_CLICKABLE).filter(has_text=a["text"])
                if await clickable.count() > 0:
                    target = clickable.first
            except Exception:  # noqa: BLE001 — fall back to the plain text match
                pass
        try:
            await getattr(target, op)(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — turn a slow timeout into fast guidance
            hint = ""
            if op in ("click", "hover"):
                hint = (
                    " Try a more specific target (role+name or exact text), or "
                    "click by xpct/ypct read off a screenshot."
                )
            return f"[ERROR] {op} failed: {str(exc).splitlines()[0]}{hint}"
        await self._settle(page)
        return f"{op} ok"

    async def _navigate(self, url: str) -> str:
        if not url:
            return "[ERROR] 'url' is required for navigate"
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        page = await self._ensure_page()
        response = await page.goto(url, wait_until="domcontentloaded")
        status = response.status if response else "?"
        await self._settle(page)
        title = await page.title()
        return f"Navigated to {url} (status={status}, title={title!r})"

    async def _read_page(self) -> str:
        """Structured accessibility snapshot of interactive/labelled nodes."""
        page = await self._ensure_page()
        snap = await page.accessibility.snapshot(interesting_only=True)
        lines: list[str] = []

        def _walk(node: dict[str, Any], depth: int = 0) -> None:
            role = node.get("role", "")
            name = (node.get("name") or "").strip()
            value = node.get("value")
            if role and role not in ("WebArea", "generic", "none"):
                line = f"{'  ' * depth}- {role}"
                if name:
                    line += f' "{name[:80]}"'
                if value:
                    line += f" = {str(value)[:40]!r}"
                lines.append(line)
                depth += 1
            for child in node.get("children", []) or []:
                _walk(child, depth)

        if snap:
            _walk(snap)
        header = f"url: {page.url}\ntitle: {await page.title()!r}\n"
        body = "\n".join(lines) if lines else "(no interactive nodes)"
        out = header + body
        if len(out) > _MAX_OUTPUT_CHARS:
            out = out[:_MAX_OUTPUT_CHARS] + "\n... [truncated]"
        return out

    async def _find(self, a: dict[str, Any]) -> str:
        """List elements matching a descriptor (role/text/label/placeholder/CSS)."""
        page = await self._ensure_page()
        loc = self._locator(page, a)
        if loc is None:
            return "[ERROR] provide selector|role+name|text|label|placeholder"
        count = await loc.count()
        if count == 0:
            return "[no matches]"
        out: list[str] = [f"{count} match(es):"]
        for i in range(min(count, 15)):
            el = loc.nth(i)
            try:
                txt = (await el.inner_text())[:80].replace("\n", " ").strip()
            except Exception:  # noqa: BLE001
                txt = ""
            out.append(f"  [{i}] {txt!r}")
        return "\n".join(out)

    async def _fill(self, a: dict[str, Any]) -> str:
        page = await self._ensure_page()
        loc = self._locator(page, a)
        if loc is None:
            return "[ERROR] provide selector|role+name|text|label|placeholder"
        value = a.get("value", a.get("text", ""))
        await loc.first.fill(str(value), timeout=int(a.get("timeout", self._timeout)))
        await self._settle(page)
        return "filled"

    async def _press(self, a: dict[str, Any]) -> str:
        page = await self._ensure_page()
        key = a.get("key", "")
        if not key:
            return "[ERROR] 'key' is required for press"
        loc = self._locator(page, a)
        if loc is not None:
            await loc.first.press(key, timeout=int(a.get("timeout", self._timeout)))
        else:
            await page.keyboard.press(key)
        await self._settle(page)
        return f"pressed {key}"

    async def _select(self, a: dict[str, Any]) -> str:
        page = await self._ensure_page()
        loc = self._locator(page, a)
        if loc is None:
            return "[ERROR] provide a target for select"
        await loc.first.select_option(str(a.get("value", "")))
        await self._settle(page)
        return "selected"

    async def _upload(self, a: dict[str, Any]) -> str:
        page = await self._ensure_page()
        loc = self._locator(page, a)
        if loc is None or not a.get("path"):
            return "[ERROR] upload needs a file input target and 'path'"
        await loc.first.set_input_files(a["path"])
        return "uploaded"

    async def _scroll(self, a: dict[str, Any]) -> str:
        page = await self._ensure_page()
        loc = self._locator(page, a)
        if loc is not None:
            await loc.first.scroll_into_view_if_needed()
            return "scrolled into view"
        await page.mouse.wheel(0, int(a.get("value", 600) or 600))
        return "scrolled"

    async def _wait_for(self, a: dict[str, Any]) -> str:
        page = await self._ensure_page()
        timeout = int(a.get("timeout", self._timeout))
        state = a.get("state")
        if state in ("load", "domcontentloaded", "networkidle"):
            await page.wait_for_load_state(state, timeout=timeout)
            return f"load state {state}"
        loc = self._locator(page, a)
        if loc is not None:
            await loc.first.wait_for(state=state or "visible", timeout=timeout)
            return f"element {state or 'visible'}"
        if a.get("text"):
            await page.get_by_text(a["text"]).first.wait_for(timeout=timeout)
            return "text visible"
        return "[ERROR] wait_for needs a target or load state"

    async def _wait_network_idle(self, a: dict[str, Any]) -> str:
        page = await self._ensure_page()
        await page.wait_for_load_state("networkidle", timeout=int(a.get("timeout", self._timeout)))
        return "network idle"

    async def _history(self, op: str) -> str:
        page = await self._ensure_page()
        if op == "back":
            await page.go_back()
        elif op == "forward":
            await page.go_forward()
        else:
            await page.reload()
        await self._settle(page)
        return f"{op} ok ({page.url})"

    async def _screenshot(self) -> str:
        try:
            page = await self._ensure_page()
            url = page.url or ""
            if not url or url == "about:blank":
                # A fresh browser page is blank — screenshotting it yields a white
                # image the model can't interpret ("I can't see the screenshot").
                # Tell it what to do instead of returning a useless blank.
                return (
                    "[ERROR] The browser has no page loaded (about:blank), so a "
                    "screenshot would be blank. This tool only captures the browser "
                    "page — navigate to a URL first (action='navigate'). To capture "
                    "the user's actual desktop/screen, use the desktop screenshot "
                    "(the input_control tool's 'screenshot' action) instead."
                )
            png_bytes = await page.screenshot(type="png")
            b64 = base64.b64encode(png_bytes).decode()
            return f"[screenshot:image/png;base64,{b64}]"
        except Exception as exc:
            return f"[ERROR] screenshot failed: {exc}"

    async def _zoom(self, a: dict[str, Any]) -> str:
        """A magnified, coordinate-labelled view of a viewport region, so the model
        can read an exact CSS pixel it cannot resolve in the full screenshot."""
        page = await self._ensure_page()
        vp = page.viewport_size or {"width": 1280, "height": 720}
        w, h = vp["width"], vp["height"]
        factor = float(a.get("zoom") or 8)
        if a.get("x") is not None and a.get("y") is not None:
            cx, cy = int(a["x"]), int(a["y"])
        elif a.get("xpct") is not None and a.get("ypct") is not None:
            cx, cy = round(float(a["xpct"]) / 100.0 * w), round(float(a["ypct"]) / 100.0 * h)
        else:
            cx, cy = w // 2, h // 2
        x0, y0, rw, rh = host.zoom_region_box(w, h, cx, cy, factor)
        crop = await page.screenshot(type="png", clip={"x": x0, "y": y0, "width": rw, "height": rh})
        if host.pil_available():
            crop = host.annotate_zoom(crop, x0=x0, y0=y0, magnify=_ZOOM_OUT_LONG_EDGE / max(rw, rh))
            note = "Read the labelled grid for the exact pixel, then click with x/y."
        else:
            note = "Install Pillow for a coordinate-labelled grid."
        b64 = base64.b64encode(crop).decode()
        return (
            f"zoom of x[{x0}-{x0 + rw}] y[{y0}-{y0 + rh}] (viewport {w}x{h}). {note}\n"
            f"[screenshot:image/png;base64,{b64}]"
        )

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
