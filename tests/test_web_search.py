"""Tests for the web_search tool and web_fetch's non-URL guard."""

from __future__ import annotations

import pytest

from leuk.tools.web_search import WebSearchTool, _parse_results, _real_url

_SAMPLE = """
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fminesweeperonline.com%2F&rut=x">
    Minesweeper Online</a>
  <a class="result__snippet">Play the classic game free.</a>
</div>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fminesweeper.online%2F">
    Minesweeper</a>
</div>
"""


class _Resp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        ...


class _Client:
    def __init__(self, *a, **k):
        ...

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, params=None):
        return _Resp(_SAMPLE)


class TestParsing:
    def test_unwraps_ddg_redirect(self):
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fb%3D1&rut=z"
        assert _real_url(href) == "https://example.com/a?b=1"

    def test_protocol_relative(self):
        assert _real_url("//example.com/x") == "https://example.com/x"

    def test_parse_results(self):
        rows = _parse_results(_SAMPLE, 10)
        assert rows[0][0] == "Minesweeper Online"
        assert rows[0][1] == "https://minesweeperonline.com/"
        assert "classic game" in rows[0][2]
        assert len(rows) == 2

    def test_parse_respects_limit(self):
        assert len(_parse_results(_SAMPLE, 1)) == 1


class TestExecute:
    @pytest.mark.asyncio
    async def test_returns_formatted_results(self, monkeypatch):
        import leuk.tools.web_search as ws

        monkeypatch.setattr(ws.httpx, "AsyncClient", _Client)
        out = await WebSearchTool().execute({"query": "minesweeper"})
        assert "Search results for 'minesweeper'" in out
        assert "https://minesweeperonline.com/" in out

    @pytest.mark.asyncio
    async def test_requires_query(self):
        out = await WebSearchTool().execute({"query": "   "})
        assert "[ERROR]" in out

    @pytest.mark.asyncio
    async def test_no_results(self, monkeypatch):
        import leuk.tools.web_search as ws

        class _Empty(_Client):
            async def get(self, url, params=None):
                return _Resp("<html></html>")

        monkeypatch.setattr(ws.httpx, "AsyncClient", _Empty)
        out = await WebSearchTool().execute({"query": "zzzzz"})
        assert "No results" in out


class TestWebFetchGuard:
    @pytest.mark.asyncio
    async def test_rejects_search_query(self):
        from leuk.tools.web_fetch import WebFetchTool

        out = await WebFetchTool().execute({"url": "simple online minesweeper game"})
        assert "[ERROR]" in out and "web_search" in out

    @pytest.mark.asyncio
    async def test_rejects_bare_word(self):
        from leuk.tools.web_fetch import WebFetchTool

        out = await WebFetchTool().execute({"url": "minesweeper"})
        assert "[ERROR]" in out and "not a URL" in out
