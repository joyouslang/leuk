"""Text-to-speech backends.

Two implementations:
    - ``LocalCoquiTTS`` — uses ``coqui-tts`` for fully offline speech synthesis.
    - ``OpenAITTS`` — uses the OpenAI TTS API for cloud-based synthesis.

Both accept text and play audio through sounddevice, with optional
background (non-blocking) playback.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Default playback sample rate for TTS output
_PLAYBACK_RATE = 22_050  # coqui-tts default


class TTSBackend(ABC):
    """Abstract text-to-speech backend."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to WAV audio bytes.

        Returns raw WAV file bytes.
        """

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Synthesize and play text through speakers.

        Playback is non-blocking (runs in background thread).
        """

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output audio sample rate."""


class LocalCoquiTTS(TTSBackend):
    """Offline TTS using coqui-tts (Tacotron2-DDC by default).

    Parameters
    ----------
    model_name:
        Coqui TTS model identifier.  Default uses Tacotron2-DDC (~100MB).
    gpu:
        Whether to use GPU acceleration.  Default False (CPU).
    """

    # Default lightweight model
    DEFAULT_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

    def __init__(
        self,
        model_name: str | None = None,
        gpu: bool = False,
    ) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._gpu = gpu
        self._tts: object | None = None
        self._sample_rate: int = _PLAYBACK_RATE

    def _ensure_model(self) -> object:
        """Lazy-load the TTS model."""
        if self._tts is None:
            try:
                from TTS.api import TTS
            except ImportError as exc:
                raise ImportError(
                    "coqui-tts is not installed. Install with: uv pip install leuk[voice]"
                ) from exc

            logger.info("Loading coqui-tts model: %s", self._model_name)
            self._tts = TTS(model_name=self._model_name, gpu=self._gpu)
            # Try to get actual sample rate from model config
            try:
                config = self._tts.synthesizer.output_sample_rate  # type: ignore[union-attr]
                if config:
                    self._sample_rate = int(config)
            except AttributeError, TypeError:
                pass
        return self._tts

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def synthesize(self, text: str) -> bytes:
        import numpy as np

        tts = self._ensure_model()

        def _run() -> bytes:
            # coqui-tts returns a list of float samples
            wav_list = tts.tts(text=text)  # type: ignore[union-attr]
            samples = np.array(wav_list, dtype=np.float32)

            # Convert to int16 WAV
            int_samples = (samples * 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                wf.writeframes(int_samples.tobytes())
            return buf.getvalue()

        return await asyncio.to_thread(_run)

    async def speak(self, text: str) -> None:
        import numpy as np
        import sounddevice as sd

        wav_bytes = await self.synthesize(text)

        # Decode WAV to play
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            rate = wf.getframerate()

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        # Play in background thread (non-blocking)
        await asyncio.to_thread(sd.play, samples, rate)

    async def close(self) -> None:
        self._tts = None


class OpenAITTS(TTSBackend):
    """Cloud-based TTS using OpenAI's TTS API.

    Parameters
    ----------
    api_key:
        OpenAI API key.  If None, reads from env.
    model:
        TTS model name.  Default ``"tts-1"`` (fast).
        Use ``"tts-1-hd"`` for higher quality.
    voice:
        Voice ID.  Default ``"alloy"``.
        Options: alloy, echo, fable, onyx, nova, shimmer.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "tts-1",
        voice: str = "alloy",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._client: object | None = None
        self._sample_rate = 24_000  # OpenAI TTS output rate

    def _ensure_client(self) -> object:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ImportError("openai package is required for OpenAI TTS.") from exc

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def synthesize(self, text: str) -> bytes:
        client = self._ensure_client()

        response = await client.audio.speech.create(  # type: ignore[union-attr]
            model=self._model,
            voice=self._voice,
            input=text,
            response_format="wav",
        )
        return response.content  # type: ignore[union-attr]

    async def speak(self, text: str) -> None:
        import numpy as np
        import sounddevice as sd

        wav_bytes = await self.synthesize(text)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            rate = wf.getframerate()

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        await asyncio.to_thread(sd.play, samples, rate)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()  # type: ignore[union-attr]
            self._client = None


def create_tts_backend(
    backend: str = "local",
    *,
    model_name: str | None = None,
    voice: str = "alloy",
    api_key: str | None = None,
) -> TTSBackend:
    """Factory for creating TTS backends.

    Parameters
    ----------
    backend:
        ``"local"`` for coqui-tts, ``"openai"`` for OpenAI API.
    model_name:
        Model identifier (local: coqui model name, openai: tts-1/tts-1-hd).
    voice:
        Voice ID (openai only).
    api_key:
        OpenAI API key (openai backend only).
    """
    match backend:
        case "local":
            return LocalCoquiTTS(model_name=model_name)
        case "openai":
            return OpenAITTS(
                api_key=api_key,
                model=model_name or "tts-1",
                voice=voice,
            )
        case _:
            raise ValueError(f"Unknown TTS backend: {backend!r}")
