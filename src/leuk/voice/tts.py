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

# Tacotron2-DDC uses conv1d with kernel_size=5 in the text encoder.
# Inputs shorter than this (after tokenisation) crash with:
#   "Kernel size can't be greater than actual input size"
# The check must count *vocabulary-surviving* characters, not raw length,
# because the tokenizer silently discards anything outside its charset
# (e.g. Cyrillic, CJK, emojis).
_MIN_TEXT_LENGTH = 8  # generous margin above kernel_size=5

# Default character vocabulary for Tacotron2-DDC (ljspeech).
# Characters outside this set are silently dropped by the tokenizer.
# Loaded from the model config at runtime when available; this is the
# fallback used before the model is loaded.
_DEFAULT_VOCAB = frozenset("_-!'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


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
    """Offline TTS using coqui-tts.

    Supports both single-speaker models (e.g. Tacotron2-DDC) and
    multi-speaker/multilingual models (e.g. XTTSv2).

    Parameters
    ----------
    model_name:
        Coqui TTS model identifier.  Default uses XTTSv2 (~1.9 GB),
        which supports 17 languages and voice cloning.
    gpu:
        Whether to use GPU acceleration.  ``None`` (default) auto-detects
        via ``torch.cuda.is_available()``.  Users who install CPU-only
        PyTorch get CPU transparently; users with CUDA PyTorch get GPU.
    speaker:
        Speaker name for multi-speaker models.  Default ``"Claribel Dervla"``.
        Ignored for single-speaker models.
    language:
        Language code for multilingual models (e.g. ``"en"``, ``"ru"``).
        Default ``"en"``.  Ignored for monolingual models.
    speaker_wav:
        Path to a WAV file for voice cloning (XTTSv2 only).  When set,
        overrides the ``speaker`` parameter.
    """

    # Default model — XTTSv2 (multilingual, multi-speaker)
    DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

    # Fallback lightweight English-only model
    ENGLISH_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

    def __init__(
        self,
        model_name: str | None = None,
        gpu: bool | None = None,
        speaker: str | None = None,
        language: str | None = None,
        speaker_wav: str | None = None,
    ) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        if gpu is None:
            try:
                import torch

                gpu = torch.cuda.is_available()
            except ImportError:
                gpu = False
        self._gpu = gpu
        self._speaker = speaker or "Claribel Dervla"
        self._language = language or "en"
        self._speaker_wav = speaker_wav
        self._tts: object | None = None
        self._sample_rate: int = _PLAYBACK_RATE
        self._vocab: frozenset[str] | None = _DEFAULT_VOCAB
        self._is_multilingual: bool = False
        self._is_multi_speaker: bool = False

    def _ensure_model(self) -> object:
        """Lazy-load the TTS model."""
        if self._tts is None:
            try:
                import warnings

                import torch

                # ── Compatibility shims for coqui-tts + transformers 5.x ──
                # 1) ``isin_mps_friendly`` was removed in transformers 5.x
                #    but coqui-tts still imports it at the module level.
                import transformers.pytorch_utils as _tpu  # type: ignore[import-untyped]

                if not hasattr(_tpu, "isin_mps_friendly"):
                    _tpu.isin_mps_friendly = torch.isin  # type: ignore[attr-defined]

                # 2) coqui-tts gates on ``is_torchcodec_available()`` when
                #    torch ≥ 2.9, but only uses torchcodec for audio *file*
                #    I/O — we never hit that path (we feed raw samples).
                #    Bypass the check so users don't need to install the
                #    package.
                import transformers.utils.import_utils as _iu

                _iu.is_torchcodec_available.cache_clear()
                _iu.is_torchcodec_available = lambda: True  # type: ignore[assignment]

                # Suppress the ``gpu`` deprecation warning from TTS()
                warnings.filterwarnings(
                    "ignore", message=r".*`gpu` will be deprecated.*"
                )

                from TTS.api import TTS
            except ImportError as exc:
                raise ImportError(
                    "coqui-tts is not installed. Install with: uv pip install 'leuk[voice]'"
                ) from exc

            # Suppress the noisy per-character "Character 'X' not found in
            # the vocabulary" warnings from the coqui tokenizer.  We handle
            # out-of-vocab text ourselves in _prepare_text.
            logging.getLogger("TTS.tts.utils.text.tokenizer").setLevel(logging.ERROR)

            logger.info("Loading coqui-tts model: %s", self._model_name)
            self._tts = TTS(model_name=self._model_name, gpu=self._gpu)

            # Detect model capabilities
            self._is_multilingual = getattr(self._tts, "is_multi_lingual", False)
            self._is_multi_speaker = getattr(self._tts, "is_multi_speaker", False)

            # Try to get actual sample rate from model config
            try:
                config = self._tts.synthesizer.output_sample_rate  # type: ignore[union-attr]
                if config:
                    self._sample_rate = int(config)
            except (AttributeError, TypeError):
                pass

            # Extract the model's character vocabulary so _prepare_text can
            # accurately predict which characters survive tokenisation.
            # Multilingual models (XTTS) use BPE tokenizers — no char vocab.
            try:
                chars = self._tts.synthesizer.tts_config.characters  # type: ignore[union-attr]
                if chars is not None:
                    charset = getattr(chars, "characters", None) or ""
                    if charset:
                        self._vocab = frozenset(charset)
                        logger.debug("TTS vocab: %d chars", len(self._vocab))
                else:
                    # Multilingual model (e.g. XTTS) — no character filter needed
                    self._vocab = None
                    logger.debug("TTS model uses BPE tokenizer — no char vocab filter")
            except (AttributeError, TypeError):
                pass  # keep _DEFAULT_VOCAB
        return self._tts

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _prepare_text(self, text: str) -> str:
        """Prepare text for TTS synthesis.

        For character-based models (Tacotron2-DDC), the text encoder uses
        conv1d with ``kernel_size=5``.  If the phoneme/character sequence is
        shorter than the kernel, PyTorch raises
        ``"Kernel size can't be greater than actual input size"``.

        The tokenizer silently discards characters outside the model's
        vocabulary (e.g. Cyrillic, CJK, emojis when using an English-only
        model).  We count only *vocabulary-surviving* characters to decide
        whether the text is too short.

        For multilingual models (XTTSv2) that use BPE tokenizers, there is
        no character vocabulary to filter against — all text is accepted.
        Only basic emptiness checks are applied.

        Returns ``""`` when the text has too few usable characters (the
        caller should return silence instead of sending it to the model).
        """
        cleaned = text.strip()
        if not cleaned:
            return ""

        # Multilingual models (XTTS) accept all scripts via BPE — no
        # character-level filtering needed.
        if self._vocab is None:
            return cleaned

        # Character-based model: count surviving characters.
        vocab = self._vocab
        surviving = [ch for ch in cleaned if ch in vocab]

        # Check if there are any *letter* characters the model can
        # pronounce.  Spaces and punctuation alone are not useful speech.
        has_letters = any(ch.isalpha() for ch in surviving)
        if not has_letters:
            if any(ch.isalpha() for ch in cleaned):
                # The original text *has* letters, but none survive the
                # vocab filter → language mismatch (e.g. Cyrillic with
                # an English model).
                logger.warning(
                    "TTS: text contains letters but none are in the model vocabulary "
                    "(%d chars, %d surviving) — returning silence.  "
                    "The model may not support this language.",
                    len(cleaned),
                    len(surviving),
                )
            return ""

        effective_len = len(surviving)
        if effective_len < _MIN_TEXT_LENGTH:
            # Pad with periods (synthesise as brief pauses)
            pad_needed = _MIN_TEXT_LENGTH - effective_len
            cleaned = cleaned + " ." * ((pad_needed + 1) // 2)

        return cleaned

    async def synthesize(self, text: str) -> bytes:
        import numpy as np

        safe_text = self._prepare_text(text)
        if not safe_text:
            # Return a short silent WAV rather than crashing
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                wf.writeframes(b"\x00\x00" * self._sample_rate)  # ~1s silence
            return buf.getvalue()

        tts = self._ensure_model()

        # Build kwargs for multi-speaker / multilingual models
        tts_kwargs: dict[str, object] = {}
        if self._is_multi_speaker:
            if self._speaker_wav:
                tts_kwargs["speaker_wav"] = self._speaker_wav
            else:
                tts_kwargs["speaker"] = self._speaker
        if self._is_multilingual:
            tts_kwargs["language"] = self._language

        def _run() -> bytes:
            # coqui-tts returns a list of float samples
            wav_list = tts.tts(text=safe_text, **tts_kwargs)  # type: ignore[union-attr]
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
        # Skip playback entirely if there's nothing pronounceable
        safe_text = self._prepare_text(text)
        if not safe_text:
            return

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
    speaker: str | None = None,
    language: str | None = None,
    speaker_wav: str | None = None,
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
    speaker:
        Speaker name for multi-speaker local models (e.g. XTTSv2).
    language:
        Language code for multilingual local models (e.g. ``"ru"``).
    speaker_wav:
        Path to WAV for voice cloning (XTTSv2 only).
    """
    match backend:
        case "local":
            return LocalCoquiTTS(
                model_name=model_name,
                speaker=speaker,
                language=language,
                speaker_wav=speaker_wav,
            )
        case "openai":
            return OpenAITTS(
                api_key=api_key,
                model=model_name or "tts-1",
                voice=voice,
            )
        case _:
            raise ValueError(f"Unknown TTS backend: {backend!r}")
