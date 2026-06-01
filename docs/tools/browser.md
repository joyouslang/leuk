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

Every action auto-waits for actionability and a settling step
(`wait_for_load_state("networkidle")`). Helpers: `wait_for` (selector | text |
load-state) and `wait_for_network_idle`.

## Targeting

Pass either a CSS `selector` **or** a robust descriptor: `role`+`name`, `text`,
`label`, or `placeholder`.

## Actions

`navigate`, `read_page`, `find`, `click`, `fill`, `type`, `press`, `hover`,
`select`, `check`, `uncheck`, `scroll`, `wait_for`, `wait_for_network_idle`,
`go_back`, `go_forward`, `reload`, `get_url`, `get_title`, `screenshot`, `extract`,
`evaluate`, `upload`.

`screenshot` returns a `[screenshot:…]` tag the model sees natively
([Multimodal](../multimodal.md)).

## See also

- [Tools](../tools.md) · [Input Control](input_control.md) · [Multimodal](../multimodal.md)
