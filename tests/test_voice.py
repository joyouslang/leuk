"""Tests for voice/ package — STT backends, recorder, and audio clips.

Since voice dependencies (sounddevice, numpy, faster-whisper) may not be
installed in the test environment, we mock them extensively or skip
hardware-dependent tests.
"""

from __future__ import annotations

import io
import wave
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── AudioClip ──────────────────────────────────────────────────────


class TestAudioClip:
    """Test AudioClip data structure and WAV encoding."""

    def test_duration(self):
        np = pytest.importorskip("numpy")
        from leuk.voice.recorder import AudioClip

        samples = np.zeros(16000, dtype=np.int16)
        clip = AudioClip(samples=samples, sample_rate=16000)
        assert clip.duration == pytest.approx(1.0)

    def test_duration_half_second(self):
        np = pytest.importorskip("numpy")
        from leuk.voice.recorder import AudioClip

        samples = np.zeros(8000, dtype=np.int16)
        clip = AudioClip(samples=samples, sample_rate=16000)
        assert clip.duration == pytest.approx(0.5)

    def test_to_wav_bytes(self):
        """to_wav_bytes produces valid WAV data."""
        np = pytest.importorskip("numpy")
        from leuk.voice.recorder import AudioClip

        samples = np.array([0, 1000, -1000, 0], dtype=np.int16)
        clip = AudioClip(samples=samples, sample_rate=16000, channels=1)
        wav_data = clip.to_wav_bytes()

        buf = io.BytesIO(wav_data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 4

    def test_to_wav_bytes_empty(self):
        """Empty clip produces valid WAV with zero frames."""
        np = pytest.importorskip("numpy")
        from leuk.voice.recorder import AudioClip

        samples = np.array([], dtype=np.int16)
        clip = AudioClip(samples=samples, sample_rate=16000)
        wav_data = clip.to_wav_bytes()

        buf = io.BytesIO(wav_data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 0


# ── VOICE_AVAILABLE flag ──────────────────────────────────────────


class TestVoiceAvailability:
    def test_require_voice_raises_when_unavailable(self):
        import leuk.voice as voice_mod

        original = voice_mod.VOICE_AVAILABLE
        original_reason = voice_mod._MISSING_REASON
        try:
            voice_mod.VOICE_AVAILABLE = False
            voice_mod._MISSING_REASON = "test reason"
            with pytest.raises(ImportError, match="test reason"):
                voice_mod.require_voice()
        finally:
            voice_mod.VOICE_AVAILABLE = original
            voice_mod._MISSING_REASON = original_reason

    def test_require_voice_passes_when_available(self):
        import leuk.voice as voice_mod

        original = voice_mod.VOICE_AVAILABLE
        try:
            voice_mod.VOICE_AVAILABLE = True
            voice_mod.require_voice()  # should not raise
        finally:
            voice_mod.VOICE_AVAILABLE = original


# ── STT Backend factory ───────────────────────────────────────────


class TestCreateSTTBackend:
    def test_local_backend(self):
        from leuk.voice.stt import LocalWhisperSTT, create_stt_backend

        backend = create_stt_backend("local", model_size="tiny")
        assert isinstance(backend, LocalWhisperSTT)

    def test_openai_backend(self):
        from leuk.voice.stt import OpenAIWhisperSTT, create_stt_backend

        backend = create_stt_backend("openai", api_key="test-key")
        assert isinstance(backend, OpenAIWhisperSTT)

    def test_unknown_backend_raises(self):
        from leuk.voice.stt import create_stt_backend

        with pytest.raises(ValueError, match="Unknown STT backend"):
            create_stt_backend("invalid")

    def test_local_with_language(self):
        from leuk.voice.stt import LocalWhisperSTT, create_stt_backend

        backend = create_stt_backend("local", language="en")
        assert isinstance(backend, LocalWhisperSTT)
        assert backend._language == "en"

    def test_openai_with_language(self):
        from leuk.voice.stt import OpenAIWhisperSTT, create_stt_backend

        backend = create_stt_backend("openai", language="fr", api_key="key")
        assert isinstance(backend, OpenAIWhisperSTT)
        assert backend._language == "fr"


# ── LocalWhisperSTT ───────────────────────────────────────────────


class TestLocalWhisperSTT:
    def test_init_defaults(self):
        from leuk.voice.stt import LocalWhisperSTT

        stt = LocalWhisperSTT()
        assert stt._model_size == "base"
        assert stt._device == "cpu"
        assert stt._compute_type == "int8"
        assert stt._language is None

    def test_init_custom(self):
        from leuk.voice.stt import LocalWhisperSTT

        stt = LocalWhisperSTT(
            model_size="large-v3", device="cuda", compute_type="float16", language="en"
        )
        assert stt._model_size == "large-v3"
        assert stt._device == "cuda"
        assert stt._compute_type == "float16"
        assert stt._language == "en"

    @pytest.mark.asyncio
    async def test_transcribe_with_mock(self):
        np = pytest.importorskip("numpy")
        from leuk.voice.recorder import AudioClip
        from leuk.voice.stt import LocalWhisperSTT

        mock_segment = MagicMock()
        mock_segment.text = " Hello world "

        mock_info = MagicMock()
        mock_info.duration = 1.0
        mock_info.language = "en"
        mock_info.language_probability = 0.99

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        stt = LocalWhisperSTT()
        stt._model = mock_model

        samples = np.zeros(16000, dtype=np.int16)
        clip = AudioClip(samples=samples, sample_rate=16000)

        result = await stt.transcribe(clip)
        assert result == "Hello world"
        mock_model.transcribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_multiple_segments(self):
        np = pytest.importorskip("numpy")
        from leuk.voice.recorder import AudioClip
        from leuk.voice.stt import LocalWhisperSTT

        seg1 = MagicMock()
        seg1.text = " Hello "
        seg2 = MagicMock()
        seg2.text = " world "

        mock_info = MagicMock()
        mock_info.duration = 2.0
        mock_info.language = "en"
        mock_info.language_probability = 0.99

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], mock_info)

        stt = LocalWhisperSTT()
        stt._model = mock_model

        samples = np.zeros(32000, dtype=np.int16)
        clip = AudioClip(samples=samples, sample_rate=16000)

        result = await stt.transcribe(clip)
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_close_clears_model(self):
        from leuk.voice.stt import LocalWhisperSTT

        stt = LocalWhisperSTT()
        stt._model = MagicMock()
        await stt.close()
        assert stt._model is None

    def test_ensure_model_raises_without_faster_whisper(self):
        """_ensure_model raises ImportError when faster-whisper is absent."""
        import importlib
        import sys
        from unittest.mock import patch

        from leuk.voice.stt import LocalWhisperSTT

        stt = LocalWhisperSTT()
        # Temporarily hide faster_whisper from the import system
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _blocked_import(name, *args, **kwargs):
            if name == "faster_whisper":
                raise ImportError("No module named 'faster_whisper'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_blocked_import):
            with pytest.raises(ImportError, match="faster-whisper"):
                stt._model = None  # force re-init
                stt._ensure_model()


# ── OpenAIWhisperSTT ──────────────────────────────────────────────


class TestOpenAIWhisperSTT:
    def test_init_defaults(self):
        from leuk.voice.stt import OpenAIWhisperSTT

        stt = OpenAIWhisperSTT()
        assert stt._model == "whisper-1"
        assert stt._language is None
        assert stt._api_key is None

    def test_init_custom(self):
        from leuk.voice.stt import OpenAIWhisperSTT

        stt = OpenAIWhisperSTT(api_key="sk-test", model="whisper-1", language="en")
        assert stt._api_key == "sk-test"
        assert stt._language == "en"

    @pytest.mark.asyncio
    async def test_transcribe_with_mock(self):
        np = pytest.importorskip("numpy")
        from leuk.voice.recorder import AudioClip
        from leuk.voice.stt import OpenAIWhisperSTT

        mock_response = MagicMock()
        mock_response.text = "Transcribed text"

        mock_transcriptions = AsyncMock()
        mock_transcriptions.create.return_value = mock_response

        mock_audio = MagicMock()
        mock_audio.transcriptions = mock_transcriptions

        mock_client = MagicMock()
        mock_client.audio = mock_audio

        stt = OpenAIWhisperSTT(api_key="sk-test")
        stt._client = mock_client

        samples = np.zeros(16000, dtype=np.int16)
        clip = AudioClip(samples=samples, sample_rate=16000)

        result = await stt.transcribe(clip)
        assert result == "Transcribed text"
        mock_transcriptions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_with_language(self):
        np = pytest.importorskip("numpy")
        from leuk.voice.recorder import AudioClip
        from leuk.voice.stt import OpenAIWhisperSTT

        mock_response = MagicMock()
        mock_response.text = "Bonjour"

        mock_transcriptions = AsyncMock()
        mock_transcriptions.create.return_value = mock_response

        mock_audio = MagicMock()
        mock_audio.transcriptions = mock_transcriptions

        mock_client = MagicMock()
        mock_client.audio = mock_audio

        stt = OpenAIWhisperSTT(api_key="sk-test", language="fr")
        stt._client = mock_client

        samples = np.zeros(16000, dtype=np.int16)
        clip = AudioClip(samples=samples, sample_rate=16000)

        result = await stt.transcribe(clip)
        assert result == "Bonjour"

        # Check that language was passed
        call_kwargs = mock_transcriptions.create.call_args
        assert call_kwargs[1].get("language") == "fr" or (
            call_kwargs.kwargs and call_kwargs.kwargs.get("language") == "fr"
        )

    @pytest.mark.asyncio
    async def test_close_clears_client(self):
        from leuk.voice.stt import OpenAIWhisperSTT

        mock_client = AsyncMock()
        stt = OpenAIWhisperSTT()
        stt._client = mock_client
        await stt.close()
        assert stt._client is None
        mock_client.close.assert_called_once()


# ── MicRecorder ───────────────────────────────────────────────────


class TestMicRecorder:
    def _skip_if_no_voice(self):
        import leuk.voice as voice_mod

        if not voice_mod.VOICE_AVAILABLE:
            pytest.skip("voice dependencies not installed")

    def test_init_requires_voice(self):
        """MicRecorder raises if voice deps missing."""
        import leuk.voice as voice_mod

        original = voice_mod.VOICE_AVAILABLE
        original_reason = voice_mod._MISSING_REASON
        try:
            voice_mod.VOICE_AVAILABLE = False
            voice_mod._MISSING_REASON = "missing deps"
            from leuk.voice.recorder import MicRecorder

            with pytest.raises(ImportError, match="missing deps"):
                MicRecorder()
        finally:
            voice_mod.VOICE_AVAILABLE = original
            voice_mod._MISSING_REASON = original_reason

    def test_not_recording_initially(self):
        self._skip_if_no_voice()
        from leuk.voice.recorder import MicRecorder

        rec = MicRecorder()
        assert not rec.is_recording

    def test_stop_without_start_raises(self):
        self._skip_if_no_voice()
        from leuk.voice.recorder import MicRecorder

        rec = MicRecorder()
        with pytest.raises(RuntimeError, match="Not currently recording"):
            rec.stop()

    def test_cancel_when_not_recording(self):
        self._skip_if_no_voice()
        from leuk.voice.recorder import MicRecorder

        rec = MicRecorder()
        rec.cancel()  # should not raise
        assert not rec.is_recording

    def test_default_config(self):
        self._skip_if_no_voice()
        from leuk.voice.recorder import DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE, MicRecorder

        rec = MicRecorder()
        assert rec.sample_rate == DEFAULT_SAMPLE_RATE
        assert rec.channels == DEFAULT_CHANNELS

    def test_custom_config(self):
        self._skip_if_no_voice()
        from leuk.voice.recorder import MicRecorder

        rec = MicRecorder(sample_rate=44100, channels=2)
        assert rec.sample_rate == 44100
        assert rec.channels == 2


# ── Recorder constants ────────────────────────────────────────────


class TestRecorderConstants:
    def test_default_sample_rate(self):
        from leuk.voice.recorder import DEFAULT_SAMPLE_RATE

        assert DEFAULT_SAMPLE_RATE == 16_000

    def test_default_channels(self):
        from leuk.voice.recorder import DEFAULT_CHANNELS

        assert DEFAULT_CHANNELS == 1

    def test_default_dtype(self):
        from leuk.voice.recorder import DEFAULT_DTYPE

        assert DEFAULT_DTYPE == "int16"
