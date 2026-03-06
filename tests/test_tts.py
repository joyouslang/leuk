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

    def test_local_with_model(self):
        from leuk.voice.tts import LocalCoquiTTS, create_tts_backend

        backend = create_tts_backend("local", model_name="tts_models/en/ljspeech/glow-tts")
        assert isinstance(backend, LocalCoquiTTS)
        assert backend._model_name == "tts_models/en/ljspeech/glow-tts"

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
        assert tts._gpu is False
        assert tts._tts is None

    def test_init_custom(self):
        from leuk.voice.tts import LocalCoquiTTS

        tts = LocalCoquiTTS(model_name="custom/model", gpu=True)
        assert tts._model_name == "custom/model"
        assert tts._gpu is True

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

        result = await tts.synthesize("Hello")
        assert isinstance(result, bytes)

        # Verify it's valid WAV
        buf = io.BytesIO(result)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() == 4

        mock_tts.tts.assert_called_once_with(text="Hello")

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
