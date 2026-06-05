"""Web search tool — find pages for a natural-language query (no API key).

Models (especially smaller ones) reach for a "search" capability and otherwise
misuse ``web_fetch`` with a query instead of a URL. This returns a ranked list of
results (title, URL, snippet) via DuckDuckGo's keyless HTML endpoint; the agent
then uses ``web_fetch`` on a result URL to read it.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from leuk.types import ToolSpec

_TIMEOUT = 20
_ENDPOINT = "https://html.duckduckgo.com/html/"
_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"


def _real_url(href: str) -> str:
    """DuckDuckGo wraps result links as /l/?uddg=<encoded real url> — unwrap it."""
    if not href:
        return ""
    query = parse_qs(urlparse(href).query)
    if "uddg" in query:
        return unquote(query["uddg"][0])
    if href.startswith("//"):
        return "https:" + href
    return href


def _parse_results(html: str, limit: int) -> list[tuple[str, str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[tuple[str, str, str]] = []
    blocks = soup.select("div.result") or soup.select("div.web-result")
    for res in blocks:
        a = res.select_one("a.result__a")
        if a is None:
            continue
        url = _real_url(str(a.get("href") or ""))
        title = a.get_text(strip=True)
        sn = res.select_one(".result__snippet")
        snippet = sn.get_text(" ", strip=True) if sn else ""
        if url and title:
            out.append((title, url, snippet))
        if len(out) >= limit:
            break
    return out


class WebSearchTool:
    """Search the web and return ranked results for a query."""

    name = "web_search"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_search",
            description=(
                "Search the web and return a ranked list of results (title, URL, "
                "snippet) for a natural-language QUERY. Use this to FIND pages; then "
                "use web_fetch on a result URL to read one. The argument is a search "
                "query, NOT a URL."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (natural-language query).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 8).",
                    },
                },
                "required": ["query"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return "[ERROR] 'query' is required"
        try:
            limit = max(1, min(20, int(arguments.get("max_results") or 8)))
        except (TypeError, ValueError):
            limit = 8
        try:
            async with httpx.AsyncClient(
                timeout=_TIMEOUT, follow_redirects=True, headers={"User-Agent": _UA}
            ) as client:
                resp = await client.get(_ENDPOINT, params={"q": query})
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            return f"[ERROR] search failed: {exc}"

        results = _parse_results(resp.text, limit)
        if not results:
            return f"No results for {query!r}."
        lines = [f"Search results for {query!r}:"]
        for i, (title, url, snippet) in enumerate(results, 1):
            entry = f"{i}. {title}\n   {url}"
            if snippet:
                entry += f"\n   {snippet}"
            lines.append(entry)
        return "\n".join(lines)
