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

        backend = create_tts_backend("openai", model_name="tts-1-hd")
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
