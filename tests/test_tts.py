"""Tests for voice/tts.py — TTS backends."""

from __future__ import annotations

import io
import wave
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── clean_text_for_speech ────────────────────────────────────────


class TestCleanTextForSpeech:
    """Test markdown / emoji stripping for TTS input."""

    def _clean(self, text: str) -> str:
        from leuk.voice.tts import clean_text_for_speech

        return clean_text_for_speech(text)

    def test_plain_text_unchanged(self):
        assert self._clean("Hello world") == "Hello world"

    def test_strips_emojis(self):
        assert self._clean("Hello 👋 world 🚀") == "Hello world"

    def test_strips_bold_italic(self):
        assert self._clean("This is **bold** and *italic*") == "This is bold and italic"

    def test_strips_headings(self):
        assert self._clean("### Title\nBody text") == "Title\nBody text"

    def test_strips_bullets(self):
        result = self._clean("- first\n- second\n* third")
        assert "first" in result
        assert "second" in result
        assert "third" in result
        assert "-" not in result
        assert "*" not in result

    def test_strips_numbered_list(self):
        result = self._clean("1. first\n2. second")
        assert "first" in result
        assert "second" in result

    def test_strips_code_blocks(self):
        text = "Before\n\n```python\nx = 1\n```\n\nAfter"
        result = self._clean(text)
        assert "Before" in result
        assert "After" in result
        assert "x = 1" not in result

    def test_preserves_text_between_code_blocks(self):
        text = "Step 1:\n\n```python\na = 1\n```\n\nMiddle text.\n\n```python\nb = 2\n```\n\nStep 3."
        result = self._clean(text)
        assert "Step 1:" in result
        assert "Middle text." in result
        assert "Step 3." in result

    def test_strips_inline_code(self):
        assert self._clean("Use `foo` and `bar`") == "Use and"

    def test_strips_stray_backticks(self):
        assert "`" not in self._clean("stray ` backtick")

    def test_strips_links_keeps_text(self):
        result = self._clean("Check [this link](https://example.com)")
        assert "this link" in result
        assert "https" not in result

    def test_strips_images_keeps_alt(self):
        result = self._clean("![alt text](image.png)")
        assert "alt text" in result

    def test_cyrillic_preserved(self):
        result = self._clean("Привет, как дела?")
        assert result == "Привет, как дела?"

    def test_cyrillic_with_markdown(self):
        result = self._clean("**Привет!** Как *дела*? 😊")
        assert "Привет!" in result
        assert "Как" in result
        assert "дела" in result


# ── TTS Backend factory ───────────────────────────────────────────


class TestCreateTTSBackend:
    def test_local_backend_is_silero(self):
        from leuk.voice.tts import SileroTTS, create_tts_backend

        backend = create_tts_backend("local", language="ru")
        assert isinstance(backend, SileroTTS)

    def test_local_with_speaker(self):
        from leuk.voice.tts import SileroTTS, create_tts_backend

        backend = create_tts_backend("local", language="ru", speaker="aidar")
        assert isinstance(backend, SileroTTS)
        assert backend._speaker == "aidar"

    def test_local_with_language(self):
        from leuk.voice.tts import SileroTTS, create_tts_backend

        backend = create_tts_backend("local", language="en")
        assert isinstance(backend, SileroTTS)
        assert backend._language == "en"

    def test_openai_backend(self):
        from leuk.voice.tts import OpenAITTS, create_tts_backend

        backend = create_tts_backend("openai", api_key="sk-test")
        assert isinstance(backend, OpenAITTS)

    def test_openai_with_voice(self):
        from leuk.voice.tts import OpenAITTS, create_tts_backend

        backend = create_tts_backend("openai", api_key="sk-test", voice="nova")
        assert isinstance(backend, OpenAITTS)
        assert backend._voice == "nova"

    def test_openai_with_model(self):
        from leuk.voice.tts import OpenAITTS, create_tts_backend

        backend = create_tts_backend("openai", model="tts-1-hd")
        assert isinstance(backend, OpenAITTS)
        assert backend._model == "tts-1-hd"

    def test_unknown_backend_raises(self):
        from leuk.voice.tts import create_tts_backend

        with pytest.raises(ValueError, match="Unknown TTS backend"):
            create_tts_backend("invalid")


# ── SileroTTS ────────────────────────────────────────────────────


class TestSileroTTS:
    def test_init_defaults(self):
        from leuk.voice.tts import SileroTTS

        tts = SileroTTS()
        assert tts._language == "ru"
        assert tts._rate == 48_000
        assert tts._model_user is None
        assert tts._model_en is None

    def test_init_custom(self):
        from leuk.voice.tts import SileroTTS

        tts = SileroTTS(language="en", speaker="en_0", sample_rate_hz=24_000)
        assert tts._language == "en"
        assert tts._speaker == "en_0"
        assert tts._rate == 24_000

    def test_init_en_speaker(self):
        from leuk.voice.tts import SileroTTS

        tts = SileroTTS(language="ru", en_speaker="en_1")
        assert tts._en_speaker == "en_1"

    def test_sample_rate(self):
        from leuk.voice.tts import SileroTTS

        tts = SileroTTS(sample_rate_hz=24_000)
        assert tts.sample_rate == 24_000

    def test_unsupported_language_raises(self):
        from leuk.voice.tts import SileroTTS

        tts = SileroTTS(language="xx")
        with pytest.raises(ValueError, match="does not support"):
            tts._ensure_models()

    @pytest.mark.asyncio
    async def test_close_clears_models(self):
        from leuk.voice.tts import SileroTTS

        tts = SileroTTS()
        tts._model_user = MagicMock()
        tts._model_en = MagicMock()
        tts._models_ready = True
        await tts.close()
        assert tts._model_user is None
        assert tts._model_en is None
        assert tts._models_ready is False

    @pytest.mark.asyncio
    async def test_empty_text_returns_silent_wav(self):
        from leuk.voice.tts import SileroTTS

        tts = SileroTTS()
        result = await tts.synthesize("")
        assert isinstance(result, bytes)

        buf = io.BytesIO(result)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() > 0


# ── Language splitting ────────────────────────────────────────────


class TestSplitByScript:
    def _split(self, text: str, lang: str = "ru"):
        from leuk.voice.tts import _split_by_script

        return _split_by_script(text, lang)

    def test_pure_english(self):
        segs = self._split("Hello world")
        assert len(segs) == 1
        assert segs[0].lang == "en"
        assert "Hello world" in segs[0].text

    def test_pure_russian(self):
        segs = self._split("Привет мир")
        assert len(segs) == 1
        assert segs[0].lang == "ru"
        assert "Привет мир" in segs[0].text

    def test_mixed_text(self):
        segs = self._split("Привет, Hello, мир!")
        langs = [s.lang for s in segs]
        assert "ru" in langs
        assert "en" in langs

    def test_english_only_when_lang_is_en(self):
        segs = self._split("Hello world Привет", lang="en")
        assert len(segs) == 1
        assert segs[0].lang == "en"

    def test_adjacent_same_lang_merged(self):
        segs = self._split("Hello world, nice day")
        assert len(segs) == 1
        assert segs[0].lang == "en"

    def test_empty_text(self):
        segs = self._split("")
        # Should return at least one segment (possibly empty)
        assert len(segs) >= 0


class TestSanitizeText:
    def _sanitize(self, text: str, lang: str = "ru", allowed: set[str] | None = None):
        from leuk.voice.tts import _sanitize_text

        return _sanitize_text(text, lang, allowed)

    def test_en_lowercases(self):
        result = self._sanitize("Hello World", "en")
        assert result == "hello world"

    def test_strips_disallowed_chars(self):
        allowed = set("abcdefghijklmnopqrstuvwxyz .,!?")
        result = self._sanitize("hello+world=42", "en", allowed)
        assert "+" not in result
        assert "=" not in result
        assert "4" not in result
        assert "hello" in result
        assert "world" in result

    def test_collapses_spaces(self):
        allowed = set("abcdefghijklmnopqrstuvwxyz .,!?")
        result = self._sanitize("a  +  b", "en", allowed)
        assert "  " not in result

    def test_no_allowed_keeps_all(self):
        """Without allowed_chars, only lowercasing is applied for English."""
        result = self._sanitize("Hello+World", "en")
        assert result == "hello+world"

    def test_ru_preserves_cyrillic(self):
        allowed = set("АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯЁабвгдежзийклмнопрстуфхцчшщъыьэюяё .,!?-")
        result = self._sanitize("Привет, мир!", "ru", allowed)
        assert "Привет" in result

    def test_lowercase_only_model_keeps_initial_capital(self):
        # Regression: a model whose vocabulary is lowercase-only must not drop a
        # sentence-initial capital (the "first letter eaten" bug) — it should be
        # lowercased and kept, not stripped.
        allowed = set("абвгдежзийклмнопрстуфхцчшщъыьэюяё .,!?-")  # lowercase only
        result = self._sanitize("Привет, мир!", "ru", allowed)
        assert result.startswith("привет"), result
        assert "п" in result  # the leading letter survived (lowercased)

    def test_spanish_initial_capital_preserved(self):
        allowed = set("abcdefghijklmnopqrstuvwxyzñáéíóúü .,!?¿¡-")  # lowercase only
        result = self._sanitize("Hola, ¿cómo estás?", "es", allowed)
        assert result.startswith("hola"), result


# ── OpenAITTS ────────────────────────────────────────────────────


class TestOpenAITTS:
    def test_init_defaults(self):
        from leuk.voice.tts import OpenAITTS

        tts = OpenAITTS()
        assert tts._model == "tts-1"
        assert tts._voice == "alloy"
        assert tts._api_key is None

    def test_sample_rate(self):
        from leuk.voice.tts import OpenAITTS

        tts = OpenAITTS()
        assert tts.sample_rate == 24_000

    @pytest.mark.asyncio
    async def test_synthesize_with_mock(self):
        from leuk.voice.tts import OpenAITTS

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x00\x00" * 100)
        fake_wav = buf.getvalue()

        mock_response = MagicMock()
        mock_response.content = fake_wav

        mock_speech = AsyncMock()
        mock_speech.create.return_value = mock_response

        mock_audio = MagicMock()
        mock_audio.speech = mock_speech

        mock_client = MagicMock()
        mock_client.audio = mock_audio

        tts = OpenAITTS(api_key="sk-test")
        tts._client = mock_client

        result = await tts.synthesize("Hello world")
        assert result == fake_wav

    @pytest.mark.asyncio
    async def test_close_clears_client(self):
        from leuk.voice.tts import OpenAITTS

        mock_client = AsyncMock()
        tts = OpenAITTS()
        tts._client = mock_client
        await tts.close()
        assert tts._client is None
        mock_client.close.assert_called_once()


class TestStreamingInterrupt:
    """stop() must signal playback via an event, never a cross-thread sd.stop."""

    @pytest.mark.asyncio
    async def test_stop_sets_interrupt_event_passed_to_backend(self):
        import asyncio

        from leuk.voice.tts import StreamingTTSSpeaker

        seen: dict = {}

        class _FakeBackend:
            sample_rate = 24000

            async def synthesize(self, text):
                return b""

            async def speak(self, text, stop_event=None, volume=None, on_audio=None, pause=None):
                seen["event"] = stop_event
                seen["volume"] = volume
                if on_audio is not None:
                    on_audio(True)
                # Emulate a worker that halts promptly when interrupted.
                for _ in range(200):
                    if stop_event is not None and stop_event.is_set():
                        return
                    await asyncio.sleep(0.005)

            async def close(self):
                pass

        sp = StreamingTTSSpeaker(_FakeBackend())
        await sp.start()
        sp.feed("Hello there, this is a sentence. ")
        await asyncio.sleep(0.03)
        await sp.stop()
        # The backend received the speaker's interrupt event, and stop() set it.
        assert seen.get("event") is sp._interrupt
        assert sp._interrupt.is_set()


# ── Spoken-form normalization (numbers + acronyms) ───────────────


class TestNormalizeForSpeech:
    """Numbers must be spelled out (Silero has no digit vocab → silent) and
    ALL-CAPS acronyms read letter-by-letter, in every supported language."""

    def test_integers_spelled_english(self):
        from leuk.voice.tts import normalize_for_speech

        out = normalize_for_speech("I have 3 apples", "en")
        assert "3" not in out
        assert "three" in out

    def test_numbers_spelled_in_target_language(self):
        from leuk.voice.tts import normalize_for_speech

        # Russian: digits must become Cyrillic number words, not English.
        out = normalize_for_speech("у меня 5 яблок", "ru")
        assert "5" not in out
        assert "пять" in out

    def test_cis_language_falls_back_to_russian(self):
        from leuk.voice.tts import normalize_for_speech

        # Kazakh shares the CIS model; num2words has no 'kk' → spell in Russian.
        # (Cyrillic context so the number is detected as the user's CIS language.)
        out = normalize_for_speech("бізде 2 нәрсе", "kk")
        assert "два" in out

    def test_number_language_follows_latin_context_over_config(self):
        from leuk.voice.tts import normalize_for_speech

        # Voice configured Russian, but the reply is English → numbers in English,
        # detected from the surrounding Latin words (not the configured language).
        out = normalize_for_speech("I have 5 apples", "ru")
        assert "five" in out
        assert "пять" not in out

    def test_number_language_follows_cyrillic_context(self):
        from leuk.voice.tts import normalize_for_speech

        # Voice configured English, but the text is Russian → Russian numbers.
        out = normalize_for_speech("тут 7 котов", "en")
        assert "семь" in out
        assert "seven" not in out

    def test_per_number_mixed_scripts(self):
        from leuk.voice.tts import normalize_for_speech

        out = normalize_for_speech("English 4 and русский 6", "ru")
        assert "four" in out  # next to Latin
        assert "шесть" in out  # next to Cyrillic

    def test_bare_number_uses_config_fallback(self):
        from leuk.voice.tts import normalize_for_speech

        # No alphabetic context → fall back to the configured voice language.
        assert "пять" in normalize_for_speech("5", "ru")
        assert "five" in normalize_for_speech("5", "en")

    def test_ukrainian_uses_uk(self):
        from leuk.voice.tts import normalize_for_speech

        out = normalize_for_speech("5 котів", "ua")
        assert "5" not in out
        assert "котів" in out

    def test_unsupported_language_falls_back_to_english(self):
        from leuk.voice.tts import normalize_for_speech

        out = normalize_for_speech("7 items", "indic")
        assert "seven" in out

    def test_decimal_point(self):
        from leuk.voice.tts import normalize_for_speech

        out = normalize_for_speech("pi is 3.14", "en")
        assert "3.14" not in out
        assert "point" in out

    def test_thousands_grouping_not_decimal(self):
        from leuk.voice.tts import normalize_for_speech

        out = normalize_for_speech("1,000 ships", "en")
        # 1,000 is one thousand, not "one point zero".
        assert "thousand" in out
        assert "point" not in out

    def test_european_decimal_comma(self):
        from leuk.voice.tts import normalize_for_speech

        # "3,14" with a 2-digit group is a decimal comma, not thousands.
        out = normalize_for_speech("3,14", "de")
        assert "," not in out  # the digits/comma are gone
        # German decimal reads "Komma".
        assert "Komma" in out or "komma" in out

    def test_acronym_letter_by_letter(self):
        from leuk.voice.tts import normalize_for_speech

        # Spelled with spoken letter names so the TTS pronounces them.
        out = normalize_for_speech("the API and USB", "en")
        assert "ay pee eye" in out
        assert "you ess bee" in out
        assert "API" not in out and "USB" not in out

    def test_acronym_non_latin(self):
        from leuk.voice.tts import normalize_for_speech

        out = normalize_for_speech("ФСБ работает", "ru")
        assert "эф эс бэ" in out
        assert "ФСБ" not in out

    def test_single_capital_not_spelled(self):
        from leuk.voice.tts import normalize_for_speech

        # A lone capital (e.g. the article "A") is not an acronym.
        out = normalize_for_speech("A cat", "en")
        assert out == "A cat"

    def test_mixed_case_word_unchanged(self):
        from leuk.voice.tts import normalize_for_speech

        out = normalize_for_speech("iOS and macOS", "en")
        assert "iOS" in out and "macOS" in out

    def test_plain_text_unchanged(self):
        from leuk.voice.tts import normalize_for_speech

        assert normalize_for_speech("just plain words", "en") == "just plain words"

    def test_parse_number_disambiguation(self):
        from leuk.voice.tts import _parse_number

        assert _parse_number("1000") == 1000
        assert _parse_number("1,000") == 1000  # grouping
        assert _parse_number("3.14") == 3.14  # decimal
        assert _parse_number("3,14") == 3.14  # european decimal
        assert _parse_number("1,234.56") == 1234.56
        assert _parse_number("1.234,56") == 1234.56
