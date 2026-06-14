[Home](../README.md) › [Tools](../tools.md) › Browser

# Browser tool

`src/leuk/tools/browser.py:BrowserTool` drives a Chromium browser via Playwright.
It's a **context-aware, SPA/AJAX-ready** layer: it works with dynamic web apps
(client-side routing, AJAX, late-rendered DOM, iframes, open shadow DOM), not just
static pages. Optional — enable with the **Browser tool** toggle in
`/settings → General` (or `browser_enabled` / `BrowserConfig.enabled`), and:

```bash
uv sync --extra browser
uv run playwright install chromium
```

See [System Dependencies → Browser](../reference/system-dependencies.md#browser).

## Visibility

The browser window is **visible by default** so you can watch the agent drive it.
Set `browser.headless = true` (in `/settings` / `config.json`) to run it invisibly
— do this for headless servers or CI where there's no display.

## Context awareness

- `read_page` — a compact **accessibility-tree snapshot** of interactive/labelled
  nodes (role, name, value) plus url/title; resilient to re-renders and hashed
  class names.
- `find` — enumerate candidates by role/text/label/placeholder.

## SPA / AJAX readiness

Every action auto-waits for actionability and a short settling step
(`wait_for_load_state("networkidle")`). Helpers: `wait_for` (selector | text |
load-state) and `wait_for_network_idle`.

### Timeouts (the hot path)

Kept short so a mistargeted action fails fast and the model can retry instead of
blocking:

- `browser.timeout_ms` (default `6000`) — per-action timeout; override per call
  with the `timeout` arg for a genuinely slow element.
- `browser.settle_ms` (default `2500`) — the post-action `networkidle` wait is
  **capped** here, because pages with ads/websockets/polling never reach
  networkidle and would otherwise stall *every* action for the full wait.

On a click/hover timeout the error is trimmed (no verbose Playwright call log) and
points at faster alternatives (a more specific target, or an `xpct`/`ypct` click).

## Targeting

Pass either a CSS `selector` **or** a robust descriptor: `role`+`name`, `text`,
`label`, or `placeholder`.

For `click`/`hover`, a bare `text` match **prefers an interactive element**
(`a`, `button`, `[role=button]`, …) so a word like `Beginner` lands on the actual
button rather than a heading that merely contains it (which would resolve, then
time out). When no good selector exists (e.g. a vision-driven model reading a
screenshot), target by **position** instead: `xpct`/`ypct` (percent of the
viewport, `0`–`100`; top-left `0,0`, centre `50,50`) — resolution-independent, so
it survives the screenshot being scaled down for the model — or `x`/`y` (CSS
pixels). A selector, when given, always takes precedence. See
[Steering](../steering.md), which nudges local models to click by percentage.

## Actions

`navigate`, `read_page`, `find`, `click`, `fill`, `type`, `press`, `hover`,
`select`, `check`, `uncheck`, `scroll`, `wait_for`, `wait_for_network_idle`,
`go_back`, `go_forward`, `reload`, `get_url`, `get_title`, `screenshot`, `extract`,
`evaluate`, `upload`.

`screenshot` returns a `[screenshot:…]` tag the model sees natively
([Multimodal](../multimodal.md)).

## See also

- [Tools](../tools.md) · [Input Control](input_control.md) · [Multimodal](../multimodal.md)
