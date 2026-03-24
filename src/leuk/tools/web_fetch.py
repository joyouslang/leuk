"""Web browsing tool -- fetch and extract content from URLs."""

from __future__ import annotations

from typing import Any

import httpx
from bs4 import BeautifulSoup

from leuk.types import ToolSpec

_MAX_RESPONSE_SIZE = 10_000_000  # 10 MB
_TIMEOUT = 30  # seconds
_MAX_OUTPUT_CHARS = 256_000


class WebFetchTool:
    """Fetch content from a URL and return it as readable text."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                follow_redirects=True,
                timeout=_TIMEOUT,
                headers={
                    "User-Agent": "leuk-agent/0.1 (https://github.com/leuk-agent)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
                },
                max_redirects=5,
            )
        return self._client

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_fetch",
            description=(
                "Fetch content from a URL and return it as readable text. "
                "Supports HTML pages (automatically extracts text), plain text, "
                "and JSON. Use this to look up documentation, APIs, or any web content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["text", "html", "raw"],
                        "description": (
                            "Output format: 'text' (default) extracts readable text from HTML, "
                            "'html' returns raw HTML, 'raw' returns the response body as-is"
                        ),
                    },
                    "selector": {
                        "type": "string",
                        "description": (
                            "Optional CSS selector to extract specific content from the page "
                            "(e.g., 'article', 'main', '.content'). Only used with 'text' format."
                        ),
                    },
                },
                "required": ["url"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        url = arguments["url"]
        fmt = arguments.get("format", "text")
        selector = arguments.get("selector")

        # Validate URL
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()
        except httpx.TimeoutException:
            return f"[ERROR] Request timed out after {_TIMEOUT}s"
        except httpx.TooManyRedirects:
            return "[ERROR] Too many redirects"
        except httpx.HTTPStatusError as exc:
            return (
                f"[ERROR] HTTP {exc.response.status_code}: {exc.response.reason_phrase}"
            )
        except httpx.RequestError as exc:
            return f"[ERROR] Request failed: {exc}"

        # Check size
        content_length = len(response.content)
        if content_length > _MAX_RESPONSE_SIZE:
            return f"[ERROR] Response too large ({content_length} bytes, max {_MAX_RESPONSE_SIZE})"

        content_type = response.headers.get("content-type", "")
        body = response.text

        if fmt == "raw":
            result = body
        elif fmt == "html":
            result = body
        else:
            # Extract text
            if "html" in content_type or body.strip().startswith("<"):
                result = _html_to_text(body, selector=selector)
            elif "json" in content_type:
                # Pretty-print JSON
                import json

                try:
                    data = json.loads(body)
                    result = json.dumps(data, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    result = body
            else:
                result = body

        # Truncate if needed
        if len(result) > _MAX_OUTPUT_CHARS:
            result = (
                result[:_MAX_OUTPUT_CHARS]
                + f"\n... [truncated, {len(result)} chars total]"
            )

        header = f"[{response.status_code}] {url} ({content_length} bytes)"
        return f"{header}\n\n{result}"


def _html_to_text(html: str, *, selector: str | None = None) -> str:
    """Extract readable text from HTML, optionally scoped to a CSS selector."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # If selector specified, narrow scope
    if selector:
        elements = soup.select(selector)
        if elements:
            text_parts = [el.get_text(separator="\n", strip=True) for el in elements]
            return "\n\n".join(text_parts)
        else:
            return f"[No elements matched selector: {selector}]"

    # Extract text
    text = soup.get_text(separator="\n", strip=True)

    # Clean up excessive blank lines
    lines = [line.strip() for line in text.splitlines()]
    cleaned = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= 1:
                cleaned.append("")
        else:
            blank_count = 0
            cleaned.append(line)

    return "\n".join(cleaned)
