"""Tests for voice/tts.py — TTS backends."""

from __future__ import annotations

import io
import wave
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── TTS Backend factory ───────────────────────────────────────────


class TestCreateTTSBackend:
    def test_local_backend(self):
        from leuk.voice.tts import LocalCoquiTTS, create_tts_backend

        backend = create_tts_backend("local")
        assert isinstance(backend, LocalCoquiTTS)
        # Default model is now XTTSv2
        assert backend._model_name == LocalCoquiTTS.DEFAULT_MODEL
        assert "xtts_v2" in backend._model_name

    def test_local_with_model(self):
        from leuk.voice.tts import LocalCoquiTTS, create_tts_backend

        backend = create_tts_backend("local", model_name="tts_models/en/ljspeech/glow-tts")
        assert isinstance(backend, LocalCoquiTTS)
        assert backend._model_name == "tts_models/en/ljspeech/glow-tts"

    def test_local_with_speaker_and_language(self):
        from leuk.voice.tts import LocalCoquiTTS, create_tts_backend

        backend = create_tts_backend("local", speaker="Ana Florence", language="ru")
        assert isinstance(backend, LocalCoquiTTS)
        assert backend._speaker == "Ana Florence"
        assert backend._language == "ru"

    def test_local_with_speaker_wav(self):
        from leuk.voice.tts import LocalCoquiTTS, create_tts_backend

        backend = create_tts_backend("local", speaker_wav="/path/to/voice.wav")
        assert isinstance(backend, LocalCoquiTTS)
        assert backend._speaker_wav == "/path/to/voice.wav"

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


# ── LocalCoquiTTS ─────────────────────────────────────────────────


class TestLocalCoquiTTS:
    def test_init_defaults(self):
        from leuk.voice.tts import LocalCoquiTTS

        tts = LocalCoquiTTS()
        assert tts._model_name == LocalCoquiTTS.DEFAULT_MODEL
        # GPU auto-detects via torch.cuda.is_available()
        assert isinstance(tts._gpu, bool)
        assert tts._tts is None
        assert tts._speaker == "Claribel Dervla"
        assert tts._language == "en"
        assert tts._speaker_wav is None

    def test_init_explicit_gpu(self):
        from leuk.voice.tts import LocalCoquiTTS

        tts_on = LocalCoquiTTS(gpu=True)
        assert tts_on._gpu is True
        tts_off = LocalCoquiTTS(gpu=False)
        assert tts_off._gpu is False

    def test_init_custom(self):
        from leuk.voice.tts import LocalCoquiTTS

        tts = LocalCoquiTTS(
            model_name="custom/model", gpu=True, speaker="Ana Florence", language="ru"
        )
        assert tts._model_name == "custom/model"
        assert tts._gpu is True
        assert tts._speaker == "Ana Florence"
        assert tts._language == "ru"

    def test_init_with_speaker_wav(self):
        from leuk.voice.tts import LocalCoquiTTS

        tts = LocalCoquiTTS(speaker_wav="/path/to/voice.wav")
        assert tts._speaker_wav == "/path/to/voice.wav"

    def test_sample_rate_default(self):
        from leuk.voice.tts import LocalCoquiTTS

        tts = LocalCoquiTTS()
        assert tts.sample_rate == 22_050

    @pytest.mark.asyncio
    async def test_synthesize_with_mock(self):
        """Verify synthesize produces WAV bytes via mocked model."""
        np = pytest.importorskip("numpy")
        from leuk.voice.tts import LocalCoquiTTS

        # Mock TTS model that returns float samples
        mock_tts = MagicMock()
        mock_tts.tts.return_value = [0.0, 0.5, -0.5, 0.0]

        tts = LocalCoquiTTS()
        tts._tts = mock_tts

        result = await tts.synthesize("Hello world, this is a test.")
        assert isinstance(result, bytes)

        # Verify it's valid WAV
        buf = io.BytesIO(result)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() == 4

        mock_tts.tts.assert_called_once_with(text="Hello world, this is a test.")

    @pytest.mark.asyncio
    async def test_close_clears_model(self):
        from leuk.voice.tts import LocalCoquiTTS

        tts = LocalCoquiTTS()
        tts._tts = MagicMock()
        await tts.close()
        assert tts._tts is None

    def test_ensure_model_raises_without_coqui(self):
        """_ensure_model raises ImportError when coqui-tts is absent."""
        from unittest.mock import patch

        from leuk.voice.tts import LocalCoquiTTS

        tts = LocalCoquiTTS()
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _blocked_import(name, *args, **kwargs):
            if name == "TTS.api":
                raise ImportError("No module named 'TTS'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_blocked_import):
            tts._tts = None  # force re-init
            with pytest.raises(ImportError, match="coqui-tts"):
                tts._ensure_model()


class TestPrepareText:
    """Tests for LocalCoquiTTS._prepare_text — short/non-Latin text guard."""

    def _make_tts(self):
        from leuk.voice.tts import LocalCoquiTTS

        return LocalCoquiTTS()

    def test_normal_text_unchanged(self):
        tts = self._make_tts()
        assert tts._prepare_text("Hello, world!") == "Hello, world!"

    def test_short_text_padded(self):
        tts = self._make_tts()
        result = tts._prepare_text("Ok")
        assert result.startswith("Ok")
        # Count vocab-surviving chars to verify padding
        from leuk.voice.tts import _DEFAULT_VOCAB

        surviving = sum(1 for ch in result if ch in _DEFAULT_VOCAB)
        assert surviving >= 8

    def test_single_char_padded(self):
        tts = self._make_tts()
        result = tts._prepare_text("Y")
        assert result.startswith("Y")

    def test_four_chars_padded(self):
        tts = self._make_tts()
        result = tts._prepare_text("Yes!")
        assert result.startswith("Yes!")

    def test_empty_returns_empty(self):
        tts = self._make_tts()
        assert tts._prepare_text("") == ""

    def test_whitespace_returns_empty(self):
        tts = self._make_tts()
        assert tts._prepare_text("   ") == ""

    def test_eight_chars_not_padded(self):
        tts = self._make_tts()
        result = tts._prepare_text("Abcdefgh")
        assert result == "Abcdefgh"

    def test_long_text_not_padded(self):
        tts = self._make_tts()
        text = "This is a long sentence that should not be modified at all."
        assert tts._prepare_text(text) == text

    def test_cyrillic_returns_empty(self):
        """Pure Cyrillic text has 0 vocab-surviving chars → empty."""
        tts = self._make_tts()
        assert tts._prepare_text("Привет мир") == ""

    def test_cyrillic_with_punctuation_only(self):
        """Cyrillic with punctuation — only punctuation survives.

        If the surviving chars are fewer than _MIN_TEXT_LENGTH, padding
        is added so the model doesn't crash.
        """
        tts = self._make_tts()
        result = tts._prepare_text("Да!")
        # '!' survives but it's only 1 char → either padded or empty
        if result:
            from leuk.voice.tts import _DEFAULT_VOCAB

            surviving = sum(1 for ch in result if ch in _DEFAULT_VOCAB)
            assert surviving >= 8

    def test_emoji_only_returns_empty(self):
        tts = self._make_tts()
        assert tts._prepare_text("\U0001f600\U0001f680\u2728") == ""

    def test_mixed_latin_cyrillic_keeps_latin(self):
        """Mixed text: Latin chars survive, Cyrillic discarded."""
        tts = self._make_tts()
        result = tts._prepare_text("Hello Привет world")
        assert result != ""
        assert "Hello" in result
        assert "world" in result

    def test_numbers_not_in_vocab(self):
        """Digits are NOT in the default Tacotron2-DDC vocabulary."""
        tts = self._make_tts()
        # "12345678" — all digits, none in the vocab
        result = tts._prepare_text("12345678")
        assert result == ""

    def test_multilingual_model_accepts_cyrillic(self):
        """When vocab is None (multilingual model), all text is accepted."""
        tts = self._make_tts()
        tts._vocab = None  # simulate multilingual model
        result = tts._prepare_text("Привет мир")
        assert result == "Привет мир"

    def test_multilingual_model_accepts_emoji(self):
        tts = self._make_tts()
        tts._vocab = None
        result = tts._prepare_text("\U0001f600 Hello \U0001f680")
        assert result == "\U0001f600 Hello \U0001f680"

    def test_multilingual_model_empty_still_empty(self):
        tts = self._make_tts()
        tts._vocab = None
        assert tts._prepare_text("") == ""
        assert tts._prepare_text("   ") == ""


class TestSynthesizeEmptyText:
    """Test that synthesize returns silence for empty/unspeakable text."""

    @pytest.mark.asyncio
    async def test_empty_text_returns_silent_wav(self):
        from leuk.voice.tts import LocalCoquiTTS

        tts = LocalCoquiTTS()
        # Don't need the actual model for empty text
        result = await tts.synthesize("")
        assert isinstance(result, bytes)

        buf = io.BytesIO(result)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() > 0  # has some silence samples

    @pytest.mark.asyncio
    async def test_whitespace_returns_silent_wav(self):
        from leuk.voice.tts import LocalCoquiTTS

        tts = LocalCoquiTTS()
        result = await tts.synthesize("   \n  ")
        assert isinstance(result, bytes)

        buf = io.BytesIO(result)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() > 0

    @pytest.mark.asyncio
    async def test_cyrillic_returns_silent_wav(self):
        """Pure non-Latin text returns silence (no vocab-surviving chars)."""
        from leuk.voice.tts import LocalCoquiTTS

        tts = LocalCoquiTTS()
        result = await tts.synthesize("Привет мир!")
        assert isinstance(result, bytes)

        buf = io.BytesIO(result)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() > 0  # silence, not crash

    @pytest.mark.asyncio
    async def test_short_text_uses_padded_text(self):
        """Short text is padded before being sent to the model."""
        np = pytest.importorskip("numpy")
        from leuk.voice.tts import LocalCoquiTTS, _DEFAULT_VOCAB

        mock_tts = MagicMock()
        mock_tts.tts.return_value = [0.0, 0.5, -0.5, 0.0]

        tts = LocalCoquiTTS()
        tts._tts = mock_tts

        await tts.synthesize("Ok")

        # The text passed to tts.tts() should be the padded version
        call_args = mock_tts.tts.call_args
        actual_text = call_args[1]["text"] if "text" in call_args[1] else call_args[0][0]
        surviving = sum(1 for ch in actual_text if ch in _DEFAULT_VOCAB)
        assert surviving >= 8
        assert actual_text.startswith("Ok")


# ── OpenAITTS ─────────────────────────────────────────────────────


class TestOpenAITTS:
    def test_init_defaults(self):
        from leuk.voice.tts import OpenAITTS

        tts = OpenAITTS()
        assert tts._model == "tts-1"
        assert tts._voice == "alloy"
        assert tts._api_key is None

    def test_init_custom(self):
        from leuk.voice.tts import OpenAITTS

        tts = OpenAITTS(api_key="sk-test", model="tts-1-hd", voice="shimmer")
        assert tts._api_key == "sk-test"
        assert tts._model == "tts-1-hd"
        assert tts._voice == "shimmer"

    def test_sample_rate(self):
        from leuk.voice.tts import OpenAITTS

        tts = OpenAITTS()
        assert tts.sample_rate == 24_000

    @pytest.mark.asyncio
    async def test_synthesize_with_mock(self):
        """Verify synthesize calls OpenAI API correctly."""
        from leuk.voice.tts import OpenAITTS

        # Create fake WAV bytes for response
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

        mock_speech.create.assert_called_once_with(
            model="tts-1",
            voice="alloy",
            input="Hello world",
            response_format="wav",
        )

    @pytest.mark.asyncio
    async def test_close_clears_client(self):
        from leuk.voice.tts import OpenAITTS

        mock_client = AsyncMock()
        tts = OpenAITTS()
        tts._client = mock_client
        await tts.close()
        assert tts._client is None
        mock_client.close.assert_called_once()
