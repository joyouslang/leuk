"""Tests for session auto-title validation + fallback (refusal-as-title bug)."""

from __future__ import annotations

from leuk.cli.repl import _fallback_title, _good_title


class TestGoodTitle:
    def test_accepts_short_label(self):
        assert _good_title("Browser automation task")
        assert _good_title("Weather in Berlin")

    def test_rejects_empty(self):
        assert not _good_title("")
        assert not _good_title("   ")

    def test_rejects_english_refusal(self):
        assert not _good_title("I can't open a browser, I'm a text model")
        assert not _good_title("Sorry, I am unable to help with that")

    def test_rejects_russian_refusal(self):
        assert not _good_title("Я не могу запускать браузер или открывать сайты")
        assert not _good_title("К сожалению, не могу помочь")

    def test_rejects_too_long(self):
        assert not _good_title("This is a very long sentence that keeps going and going")

    def test_rejects_prose_sentence(self):
        assert not _good_title("The user wants to open a browser and check the news today.")


class TestFallbackTitle:
    def test_uses_first_words(self):
        t = _fallback_title("open a browser and go to example.com to find the weather")
        assert t.lower().startswith("open a browser")
        assert len(t) <= 60

    def test_capitalises(self):
        assert _fallback_title("hello world")[0].isupper()

    def test_empty_gives_default(self):
        assert _fallback_title("") == "New session"
        assert _fallback_title("    ") == "New session"

    def test_non_latin(self):
        t = _fallback_title("открой браузер и зайди на сайт")
        assert "браузер" in t
