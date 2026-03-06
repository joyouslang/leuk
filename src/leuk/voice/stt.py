"""Speech-to-text backends.

Two implementations:
    - ``LocalWhisperSTT`` — uses ``faster-whisper`` for fully offline transcription.
    - ``OpenAIWhisperSTT`` — uses the OpenAI Whisper API for cloud-based transcription.

Both accept an ``AudioClip`` (from recorder.py) and return the transcribed text.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leuk.voice.recorder import AudioClip

logger = logging.getLogger(__name__)


class STTBackend(ABC):
    """Abstract speech-to-text backend."""

    @abstractmethod
    async def transcribe(self, clip: AudioClip) -> str:
        """Transcribe an audio clip to text.

        Returns the transcribed text, or an empty string if nothing was
        detected.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any resources."""


class LocalWhisperSTT(STTBackend):
    """Offline transcription using faster-whisper.

    Parameters
    ----------
    model_size:
        Whisper model size.  ``"base"`` is a good default (~150MB,
        acceptable accuracy).  Other options: ``"tiny"``, ``"small"``,
        ``"medium"``, ``"large-v3"``.
    device:
        Compute device: ``"cpu"`` or ``"cuda"``.
    compute_type:
        Quantization: ``"int8"`` (fastest on CPU), ``"float16"`` (GPU),
        ``"float32"`` (most precise).
    language:
        Force a language code (e.g. ``"en"``).  ``None`` = auto-detect.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model: object | None = None

    def _ensure_model(self) -> object:
        """Lazy-load the model on first use."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ImportError(
                    "faster-whisper is not installed. Install with: uv pip install leuk[voice]"
                ) from exc

            logger.info(
                "Loading faster-whisper model %s (device=%s, compute=%s)",
                self._model_size,
                self._device,
                self._compute_type,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    async def transcribe(self, clip: AudioClip) -> str:
        import asyncio

        import numpy as np

        model = self._ensure_model()

        # faster-whisper expects float32 audio in [-1, 1] range
        audio = clip.samples.astype(np.float32) / 32768.0

        def _run() -> str:
            segments, info = model.transcribe(  # type: ignore[union-attr]
                audio,
                language=self._language,
                beam_size=5,
                vad_filter=True,
            )
            text = " ".join(seg.text.strip() for seg in segments)
            logger.debug(
                "Transcribed %.1fs audio → %d chars (lang=%s, prob=%.2f)",
                info.duration,
                len(text),
                info.language,
                info.language_probability,
            )
            return text

        return await asyncio.to_thread(_run)

    async def close(self) -> None:
        self._model = None


class OpenAIWhisperSTT(STTBackend):
    """Cloud-based transcription using the OpenAI Whisper API.

    Parameters
    ----------
    api_key:
        OpenAI API key.  If ``None``, reads from ``OPENAI_API_KEY`` env var.
    model:
        The Whisper model to use.  Default ``"whisper-1"``.
    language:
        Optional language code.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "whisper-1",
        language: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._language = language
        self._client: object | None = None

    def _ensure_client(self) -> object:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for OpenAI Whisper STT. "
                    "It should already be installed as a core dependency."
                ) from exc

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def transcribe(self, clip: AudioClip) -> str:
        import io

        client = self._ensure_client()
        wav_bytes = clip.to_wav_bytes()

        # OpenAI expects a file-like object with a name
        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "recording.wav"

        kwargs: dict = {
            "model": self._model,
            "file": audio_file,
        }
        if self._language:
            kwargs["language"] = self._language

        response = await client.audio.transcriptions.create(**kwargs)  # type: ignore[union-attr]
        text = response.text.strip()
        logger.debug("OpenAI Whisper transcribed → %d chars", len(text))
        return text

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()  # type: ignore[union-attr]
            self._client = None


def create_stt_backend(
    backend: str = "local",
    *,
    model_size: str = "base",
    language: str | None = None,
    api_key: str | None = None,
) -> STTBackend:
    """Factory for creating STT backends.

    Parameters
    ----------
    backend:
        ``"local"`` for faster-whisper, ``"openai"`` for OpenAI API.
    model_size:
        Whisper model size (local only).
    language:
        Language code to force.
    api_key:
        OpenAI API key (openai backend only).
    """
    match backend:
        case "local":
            return LocalWhisperSTT(
                model_size=model_size,
                language=language,
            )
        case "openai":
            return OpenAIWhisperSTT(
                api_key=api_key,
                language=language,
            )
        case _:
            raise ValueError(f"Unknown STT backend: {backend!r}")
