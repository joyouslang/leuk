"""Tests for voice/ package — STT backends, recorder, and audio clips.

Since voice dependencies (sounddevice, numpy, transformers/torch) may not be
installed in the test environment, we mock them extensively or skip
hardware-dependent tests.
"""

from __future__ import annotations

import asyncio
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
        assert stt._model_size == "turbo"
        # Device auto-detects: "cuda" if available, else "cpu"
        assert stt._device in ("cpu", "cuda")
        assert stt._language is None

    def test_init_explicit_cpu(self):
        from leuk.voice.stt import LocalWhisperSTT

        stt = LocalWhisperSTT(device="cpu")
        assert stt._device == "cpu"

    def test_init_custom(self):
        from leuk.voice.stt import LocalWhisperSTT

        stt = LocalWhisperSTT(model_size="large-v3", device="cuda", language="en")
        assert stt._model_size == "large-v3"
        assert stt._device == "cuda"
        assert stt._language == "en"

    def test_model_id_mapping(self):
        """Short model names map to HuggingFace IDs."""
        from leuk.voice.stt import _MODEL_ID_MAP

        assert _MODEL_ID_MAP["base"] == "openai/whisper-base"
        assert _MODEL_ID_MAP["large-v3"] == "openai/whisper-large-v3"
        assert _MODEL_ID_MAP["turbo"] == "openai/whisper-large-v3-turbo"

    def test_full_model_id_passthrough(self):
        """A full HuggingFace model ID is used as-is."""
        from leuk.voice.stt import LocalWhisperSTT, _MODEL_ID_MAP

        stt = LocalWhisperSTT(model_size="openai/whisper-large-v3")
        # Not in the map, so it should pass through unchanged
        assert _MODEL_ID_MAP.get(stt._model_size, stt._model_size) == "openai/whisper-large-v3"

    @pytest.mark.asyncio
    async def test_transcribe_with_mock(self):
        np = pytest.importorskip("numpy")
        from leuk.voice.recorder import AudioClip
        from leuk.voice.stt import LocalWhisperSTT

        mock_pipe = MagicMock()
        mock_pipe.return_value = {"text": " Hello world "}

        stt = LocalWhisperSTT()
        stt._pipe = mock_pipe

        samples = np.zeros(16000, dtype=np.int16)
        clip = AudioClip(samples=samples, sample_rate=16000)

        result = await stt.transcribe(clip)
        assert result == "Hello world"
        mock_pipe.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_with_language(self):
        np = pytest.importorskip("numpy")
        from leuk.voice.recorder import AudioClip
        from leuk.voice.stt import LocalWhisperSTT

        mock_pipe = MagicMock()
        mock_pipe.return_value = {"text": "Привет мир"}

        stt = LocalWhisperSTT(language="ru")
        stt._pipe = mock_pipe

        samples = np.zeros(16000, dtype=np.int16)
        clip = AudioClip(samples=samples, sample_rate=16000)

        result = await stt.transcribe(clip)
        assert result == "Привет мир"

        # Check language was passed in generate_kwargs
        call_args = mock_pipe.call_args
        gen_kwargs = call_args[1].get("generate_kwargs") or call_args.kwargs.get("generate_kwargs")
        assert gen_kwargs is not None
        assert gen_kwargs["language"] == "ru"

    @pytest.mark.asyncio
    async def test_close_clears_pipe(self):
        from leuk.voice.stt import LocalWhisperSTT

        stt = LocalWhisperSTT()
        stt._pipe = MagicMock()
        await stt.close()
        assert stt._pipe is None

    def test_ensure_model_raises_without_transformers(self):
        """_ensure_model raises ImportError when transformers is absent."""
        from unittest.mock import patch

        from leuk.voice.stt import LocalWhisperSTT

        stt = LocalWhisperSTT()
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _blocked_import(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("No module named 'transformers'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_blocked_import):
            with pytest.raises(ImportError, match="transformers"):
                stt._pipe = None  # force re-init
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

    def test_output_format_is_whisper(self):
        """Output clips are always 16 kHz mono regardless of device."""
        self._skip_if_no_voice()
        from leuk.voice.recorder import MicRecorder, WHISPER_SAMPLE_RATE

        rec = MicRecorder()
        assert rec.sample_rate == WHISPER_SAMPLE_RATE
        assert rec.channels == 1

    def test_custom_device(self):
        self._skip_if_no_voice()
        from leuk.voice.recorder import MicRecorder

        rec = MicRecorder(device=0)
        assert rec.device == 0

    def test_start_wraps_portaudio_error(self):
        """start() wraps PortAudioError in RuntimeError with guidance."""
        self._skip_if_no_voice()
        from unittest.mock import patch

        import sounddevice as sd

        from leuk.voice.recorder import MicRecorder

        rec = MicRecorder()
        with (
            patch(
                "leuk.voice.recorder._query_input_device",
                side_effect=sd.PortAudioError("mock device error"),
            ),
            patch("sounddevice.query_devices", return_value=[]),
        ):
            with pytest.raises(RuntimeError, match="Cannot open any audio input device"):
                rec.start()


# ── Peek & recent_rms ─────────────────────────────────────────────


class TestPeekAndRms:
    """Tests for peek() and recent_rms() using a mocked recorder."""

    def _make_recording_recorder(self):
        """Create a MicRecorder that *appears* to be recording (mocked)."""
        np = pytest.importorskip("numpy")
        from leuk.voice.recorder import MicRecorder

        rec = MicRecorder()
        rec._recording = True
        rec._device_rate = 16000
        rec._device_channels = 1
        # Simulate 1 second of a 440 Hz sine wave.
        t = np.linspace(0, 1.0, 16000, dtype=np.float64)
        sine = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
        rec._frames = [sine.reshape(-1, 1)]
        return rec, np

    def test_peek_returns_clip_without_stopping(self):
        rec, np = self._make_recording_recorder()
        clip = rec.peek()
        assert rec.is_recording  # still recording
        assert clip.sample_rate == 16000
        assert clip.channels == 1
        assert len(clip.samples) > 0

    def test_peek_raises_when_not_recording(self):
        pytest.importorskip("numpy")
        from leuk.voice.recorder import MicRecorder

        rec = MicRecorder()
        with pytest.raises(RuntimeError, match="Not currently recording"):
            rec.peek()

    def test_recent_rms_nonzero_for_signal(self):
        rec, _ = self._make_recording_recorder()
        rms = rec.recent_rms(0.5)
        assert rms > 0

    def test_recent_rms_zero_for_silence(self):
        np = pytest.importorskip("numpy")
        from leuk.voice.recorder import MicRecorder

        rec = MicRecorder()
        rec._recording = True
        rec._device_rate = 16000
        rec._device_channels = 1
        rec._frames = [np.zeros((8000, 1), dtype=np.int16)]
        assert rec.recent_rms(0.5) == 0.0

    def test_recent_rms_empty_frames(self):
        pytest.importorskip("numpy")
        from leuk.voice.recorder import MicRecorder

        rec = MicRecorder()
        assert rec.recent_rms() == 0.0


# ── Device skip list ─────────────────────────────────────────────


class TestDeviceSkipList:
    def test_skip_names(self):
        from leuk.voice.recorder import _SKIP_DEVICE_NAMES

        for name in ("default", "jack", "pipewire", "pulse"):
            assert name in _SKIP_DEVICE_NAMES

    def test_skip_substrings(self):
        from leuk.voice.recorder import _SKIP_DEVICE_SUBSTRINGS

        assert "dummy" in _SKIP_DEVICE_SUBSTRINGS
        assert "speech-dispatcher" in _SKIP_DEVICE_SUBSTRINGS


# ── Suppress stderr ──────────────────────────────────────────────


class TestSuppressStderr:
    def test_suppress_stderr_context(self):
        """_suppress_stderr should silence fd 2 inside the context."""
        import os

        from leuk.voice.recorder import _suppress_stderr

        with _suppress_stderr():
            # Writing to fd 2 should not raise
            os.write(2, b"this should go to /dev/null\n")
        # After context, stderr should work again
        # (we can't easily verify, but no crash = success)


# ── Resampling ────────────────────────────────────────────────────


class TestResample:
    def test_same_rate_noop(self):
        np = pytest.importorskip("numpy")
        from leuk.voice.recorder import _resample

        samples = np.array([1, 2, 3, 4], dtype=np.int16)
        result = _resample(samples, 16000, 16000)
        assert np.array_equal(result, samples)

    def test_downsample(self):
        np = pytest.importorskip("numpy")
        from leuk.voice.recorder import _resample

        # 48 kHz → 16 kHz = 1/3 the samples
        samples = np.arange(4800, dtype=np.int16)
        result = _resample(samples, 48000, 16000)
        assert len(result) == 1600

    def test_upsample(self):
        np = pytest.importorskip("numpy")
        from leuk.voice.recorder import _resample

        # 8 kHz → 16 kHz = 2x the samples
        samples = np.arange(800, dtype=np.int16)
        result = _resample(samples, 8000, 16000)
        assert len(result) == 1600


# ── Recorder constants ────────────────────────────────────────────


class TestRecorderConstants:
    def test_whisper_sample_rate(self):
        from leuk.voice.recorder import WHISPER_SAMPLE_RATE

        assert WHISPER_SAMPLE_RATE == 16_000

    def test_default_sample_rate(self):
        from leuk.voice.recorder import DEFAULT_SAMPLE_RATE

        assert DEFAULT_SAMPLE_RATE == 16_000

    def test_default_channels(self):
        from leuk.voice.recorder import DEFAULT_CHANNELS

        assert DEFAULT_CHANNELS == 1

    def test_default_dtype(self):
        from leuk.voice.recorder import DEFAULT_DTYPE

        assert DEFAULT_DTYPE == "int16"


# ── ContinuousVAD ────────────────────────────────────────────────


class TestContinuousVAD:
    """Tests for the ContinuousVAD background monitor."""

    def test_vad_constants_exported(self):
        from leuk.voice.recorder import (
            VAD_MIN_SPEECH_DURATION,
            VAD_POLL_INTERVAL,
            VAD_SILENCE_TIMEOUT,
        )

        assert VAD_SILENCE_TIMEOUT > 0
        assert VAD_MIN_SPEECH_DURATION > 0
        assert VAD_POLL_INTERVAL > 0

    def test_init_not_active(self):
        pytest.importorskip("numpy")
        from leuk.voice.recorder import ContinuousVAD, MicRecorder

        rec = MicRecorder()
        vad = ContinuousVAD(rec, on_speech=lambda clip: None)
        assert not vad.active

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """ContinuousVAD can be started and stopped without errors.

        Uses a mock recorder that raises on start() to avoid hardware access.
        """
        np = pytest.importorskip("numpy")
        from unittest.mock import MagicMock, PropertyMock
        from leuk.voice.recorder import ContinuousVAD, MicRecorder

        rec = MicRecorder()
        # Mock start to avoid hardware
        rec.start = MagicMock(side_effect=RuntimeError("no device"))
        rec.cancel = MagicMock()

        callback = MagicMock()
        vad = ContinuousVAD(rec, on_speech=callback)

        vad.start()
        assert vad.active
        await asyncio.sleep(0.3)  # let it fail and retry once
        await vad.stop()
        assert not vad.active

    @pytest.mark.asyncio
    async def test_double_start_safe(self):
        np = pytest.importorskip("numpy")
        from unittest.mock import MagicMock
        from leuk.voice.recorder import ContinuousVAD, MicRecorder

        rec = MicRecorder()
        rec.start = MagicMock(side_effect=RuntimeError("no device"))
        rec.cancel = MagicMock()

        vad = ContinuousVAD(rec, on_speech=lambda c: None)
        vad.start()
        vad.start()  # should not crash or double-start
        await vad.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self):
        pytest.importorskip("numpy")
        from leuk.voice.recorder import ContinuousVAD, MicRecorder

        rec = MicRecorder()
        vad = ContinuousVAD(rec, on_speech=lambda c: None)
        await vad.stop()  # should not crash

    @pytest.mark.asyncio
    async def test_speech_detection_triggers_callback(self):
        """Simulates speech → silence → callback with a mock recorder."""
        np = pytest.importorskip("numpy")
        from unittest.mock import MagicMock
        from leuk.voice.recorder import AudioClip, ContinuousVAD, MicRecorder

        # Build a mock recorder that simulates speech then silence
        rec = MicRecorder()

        # Sequence of VAD predictions: silence → speech → silence
        vad_predictions = [
            False, False,         # silence (listening)
            True, True, True,     # speech detected
            False, False, False,  # silence resumes
            False, False, False,
            False, False, False,
        ]
        vad_idx = [0]

        samples = np.zeros(16000, dtype=np.int16)
        clip = AudioClip(samples=samples, sample_rate=16000)

        rec.start = MagicMock()
        rec.stop = MagicMock(return_value=clip)
        rec.cancel = MagicMock()
        rec.peek = MagicMock(return_value=clip)
        rec._recording = True  # is_recording reads this field

        speech_clips: list[AudioClip] = []

        async def on_speech(c: AudioClip) -> None:
            speech_clips.append(c)

        vad = ContinuousVAD(
            rec,
            on_speech=on_speech,
            sensitivity=0.5,
            silence_timeout=0.3,
            min_duration=0.1,
        )

        # Mock _is_speech to use our prediction sequence instead of Silero
        def mock_is_speech(audio_samples):
            idx = min(vad_idx[0], len(vad_predictions) - 1)
            vad_idx[0] += 1
            return vad_predictions[idx]

        vad._is_speech = mock_is_speech
        # Pre-set a mock VAD model so _ensure_vad() doesn't try to load
        vad._vad_model = MagicMock()

        vad.start()
        await asyncio.sleep(2.0)  # give it time to detect speech + silence
        await vad.stop()

        assert len(speech_clips) >= 1
