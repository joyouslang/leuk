"""Speech-to-text backends.

Two implementations:
    - ``LocalWhisperSTT`` — uses HuggingFace ``transformers`` for fully offline
      transcription.  Pure PyTorch, so it works on both CUDA (NVIDIA) and
      ROCm (AMD) GPUs.
    - ``OpenAIWhisperSTT`` — uses the OpenAI Whisper API for cloud-based
      transcription.

Both accept an ``AudioClip`` (from recorder.py) and return the transcribed text.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leuk.voice.recorder import AudioClip

logger = logging.getLogger(__name__)

# Map short model names to HuggingFace model IDs.
_MODEL_ID_MAP: dict[str, str] = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
    "turbo": "openai/whisper-large-v3-turbo",
}


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
    """Offline transcription using HuggingFace transformers.

    Uses ``AutoModelForSpeechSeq2Seq`` with the OpenAI Whisper weights.
    Since inference is pure PyTorch it works on any device PyTorch supports
    (CPU, CUDA/NVIDIA, ROCm/AMD).

    Parameters
    ----------
    model_size:
        Whisper model size.  ``"base"`` is a good default (~150 MB,
        acceptable accuracy).  Other options: ``"tiny"``, ``"small"``,
        ``"medium"``, ``"large-v3"``, ``"turbo"``.  You may also pass a
        full HuggingFace model ID directly (e.g.
        ``"openai/whisper-large-v3"``).
    device:
        Compute device: ``"cpu"`` or ``"cuda"``.  ``None`` (default)
        auto-detects via ``torch.cuda.is_available()``.
    language:
        Force a language code (e.g. ``"en"``).  ``None`` = auto-detect.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str | None = None,
        language: str | None = None,
        batch_size: int = 8,
    ) -> None:
        self._model_size = model_size
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self._device = device
        self._language = language
        self._batch_size = batch_size if device != "cpu" else 1
        self._pipe: object | None = None

    def _ensure_model(self) -> object:
        """Lazy-load the transformers pipeline on first use."""
        if self._pipe is None:
            try:
                import os

                import torch
                from transformers import (
                    AutoModelForSpeechSeq2Seq,
                    AutoProcessor,
                    pipeline,
                )
            except ImportError as exc:
                raise ImportError(
                    "transformers is not installed. Install with: uv pip install 'leuk[voice]'"
                ) from exc

            # ── Silence all non-essential output from the HF stack ────
            # Covers: forced_decoder_ids deprecation, multilingual default,
            # sequential-on-GPU hint, unauthenticated HF Hub requests,
            # "layers were not sharded" from accelerate, loading progress
            # bars, etc.
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            for _logger_name in (
                "transformers",
                "huggingface_hub",
                "accelerate",
            ):
                logging.getLogger(_logger_name).setLevel(logging.ERROR)

            import transformers as _tf

            _tf.utils.logging.set_verbosity_error()
            _tf.utils.logging.disable_progress_bar()

            try:
                import huggingface_hub as _hfh

                _hfh.utils.logging.set_verbosity_error()
            except Exception:
                pass

            model_id = _MODEL_ID_MAP.get(self._model_size, self._model_size)
            dtype = torch.float16 if self._device != "cpu" else torch.float32

            logger.info(
                "Loading Whisper model %s (device=%s, dtype=%s)",
                model_id,
                self._device,
                dtype,
            )

            # Use SDPA (Scaled Dot-Product Attention) on GPU for fused,
            # memory-efficient attention kernels.  Falls back gracefully on
            # older PyTorch / CPU.
            attn_kwargs: dict[str, str] = {}
            if self._device != "cpu":
                attn_kwargs["attn_implementation"] = "sdpa"

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                **attn_kwargs,
            )
            model.to(self._device)  # type: ignore[union-attr]
            processor = AutoProcessor.from_pretrained(model_id)

            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                dtype=dtype,
                device=self._device,
                # Chunked long-form: split audio into 30s windows and batch
                # multiple chunks together on GPU for parallel decoding.
                chunk_length_s=30,
                batch_size=self._batch_size,
            )
        return self._pipe

    async def transcribe(self, clip: AudioClip) -> str:
        import asyncio

        import numpy as np

        pipe = self._ensure_model()

        # Whisper expects float32 audio in [-1, 1] range at 16 kHz
        audio = clip.samples.astype(np.float32) / 32768.0

        generate_kwargs: dict[str, object] = {"task": "transcribe"}
        if self._language:
            generate_kwargs["language"] = self._language

        def _run() -> str:
            import torch

            kwargs: dict[str, object] = {}
            if generate_kwargs:
                kwargs["generate_kwargs"] = generate_kwargs
            with torch.inference_mode():
                result = pipe(  # type: ignore[operator]
                    audio,
                    return_timestamps=True,
                    **kwargs,
                )
            text: str = result["text"].strip()  # type: ignore[index]
            logger.debug(
                "Transcribed %.1fs audio → %d chars",
                len(clip.samples) / clip.sample_rate,
                len(text),
            )
            return text

        return await asyncio.to_thread(_run)

    async def close(self) -> None:
        self._pipe = None


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
        ``"local"`` for HuggingFace transformers Whisper,
        ``"openai"`` for OpenAI API.
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
