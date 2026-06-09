"""Tests for the Markdown → Telegram-HTML converter (pure logic, no aiogram)."""

from __future__ import annotations

from leuk.channels.markdown import markdown_to_telegram_html, split_for_telegram


class TestEscaping:
    def test_special_chars_escaped(self):
        # The whole point of HTML mode: <, >, & never break the message.
        out = markdown_to_telegram_html("if a < b && b > c")
        assert "&lt;" in out and "&gt;" in out and "&amp;" in out
        assert "<b>" not in out  # nothing was mistaken for a tag

    def test_plain_text_unchanged(self):
        assert markdown_to_telegram_html("just text") == "just text"


class TestInlineFormatting:
    def test_bold(self):
        assert markdown_to_telegram_html("**bold**") == "<b>bold</b>"

    def test_italic(self):
        assert markdown_to_telegram_html("*it*") == "<i>it</i>"

    def test_italic_underscore(self):
        assert markdown_to_telegram_html("_it_") == "<i>it</i>"

    def test_strikethrough(self):
        assert markdown_to_telegram_html("~~gone~~") == "<s>gone</s>"

    def test_bold_not_eaten_by_italic(self):
        assert markdown_to_telegram_html("**x**") == "<b>x</b>"

    def test_link(self):
        out = markdown_to_telegram_html("[docs](https://example.com)")
        assert out == '<a href="https://example.com">docs</a>'

    def test_heading_to_bold(self):
        assert markdown_to_telegram_html("## Title") == "<b>Title</b>"


class TestCode:
    def test_inline_code_escaped(self):
        out = markdown_to_telegram_html("`a < b`")
        assert out == "<code>a &lt; b</code>"

    def test_inline_code_not_formatted(self):
        # Markdown markers inside code stay literal.
        out = markdown_to_telegram_html("`**not bold**`")
        assert out == "<code>**not bold**</code>"

    def test_fenced_code_block(self):
        out = markdown_to_telegram_html("```\nx = 1 < 2\n```")
        assert out == "<pre>x = 1 &lt; 2\n</pre>"

    def test_fenced_code_with_language(self):
        out = markdown_to_telegram_html("```python\nprint(1)\n```")
        assert '<pre><code class="language-python">print(1)\n</code></pre>' == out


class TestSplit:
    def test_short_text_one_chunk(self):
        assert split_for_telegram("hello", limit=10) == ["hello"]

    def test_splits_on_line_boundaries(self):
        text = "a" * 8 + "\n" + "b" * 8 + "\n"
        chunks = split_for_telegram(text, limit=10)
        assert all(len(c) <= 10 for c in chunks)
        assert "".join(chunks) == text

    def test_hard_splits_oversized_line(self):
        text = "x" * 25
        chunks = split_for_telegram(text, limit=10)
        assert all(len(c) <= 10 for c in chunks)
        assert "".join(chunks) == text
