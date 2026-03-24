"""Text-to-speech backends.

Two implementations:
    - ``SileroTTS`` — uses Silero Models for fast, multilingual offline TTS.
      Runs at 3–17× realtime (CPU/GPU).  Supports Russian, English, German,
      French, Spanish, and many more.  Default backend.
    - ``OpenAITTS`` — uses the OpenAI TTS API for cloud-based synthesis.

All backends accept text and play audio through sounddevice.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# ── Regex patterns for stripping markdown / non-speech content ───
# Applied by ``clean_text_for_speech`` before sending to any TTS backend.
#
# Code-fence regex: match ``` optionally followed by a language tag and
# everything up to the closing ```.  Using a non-greedy ``.*?`` with
# ``re.DOTALL`` so it stops at the *first* closing fence (not the last).
_MD_CODE_BLOCK = re.compile(r"```[a-zA-Z]*\n.*?```", re.DOTALL)
_MD_INLINE_CODE = re.compile(r"`[^`]+`")
_MD_BOLD_ITALIC = re.compile(r"\*{1,3}(.+?)\*{1,3}")
_MD_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_MD_BULLET = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
_MD_NUMBERED = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
# Emoji ranges: emoticons, dingbats, symbols, supplemental symbols, flags, etc.
_EMOJI = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f900-\U0001f9ff"  # supplemental symbols
    "\U0001fa00-\U0001fa6f"  # chess symbols
    "\U0001fa70-\U0001faff"  # symbols extended-A
    "\U00002702-\U000027b0"  # dingbats
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0000200d"  # ZWJ
    "\U00002600-\U000026ff"  # misc symbols
    "\U0000231a-\U0000231b"
    "\U00002934-\U00002935"
    "\U000025aa-\U000025ab"
    "\U000025fb-\U000025fe"
    "\U00002b05-\U00002b07"
    "\U00002b1b-\U00002b1c"
    "\U00002b50\U00002b55"
    "\U00003030\U0000303d"
    "\U00003297\U00003299"
    "]+",
    flags=re.UNICODE,
)
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{2,}")


def clean_text_for_speech(text: str) -> str:
    """Strip markdown formatting, code blocks, emojis, and other
    non-speech content from LLM output to produce clean text for TTS.

    The result is plain natural-language text suitable for any TTS backend.
    """
    t = text
    # Remove fenced code blocks first (may contain anything).
    t = _MD_CODE_BLOCK.sub(" ", t)
    # Inline code → drop content (variable names aren't speakable).
    t = _MD_INLINE_CODE.sub("", t)
    # Images → alt text only
    t = _MD_IMAGE.sub(r"\1", t)
    # Links → link text only
    t = _MD_LINK.sub(r"\1", t)
    # Bold / italic → plain text
    t = _MD_BOLD_ITALIC.sub(r"\1", t)
    # Headings, bullets, numbered lists → plain text
    t = _MD_HEADING.sub("", t)
    t = _MD_BULLET.sub("", t)
    t = _MD_NUMBERED.sub("", t)
    # Emojis
    t = _EMOJI.sub("", t)
    # Horizontal rules
    t = re.sub(r"^---+$", "", t, flags=re.MULTILINE)
    # Strip remaining stray backticks (from unclosed fences).
    t = t.replace("`", "")
    # Collapse whitespace
    t = _MULTI_SPACE.sub(" ", t)
    t = _MULTI_NEWLINE.sub("\n", t)
    return t.strip()


# ── Silero TTS language ↔ model mapping ─────────────────────────
# Silero loads models via torch.hub with (language, speaker) where
# ``speaker`` is actually the model ID.  The ``language`` parameter
# determines the text processing pipeline (character set, stress, etc).
_SILERO_LANG_MODELS: dict[str, str] = {
    "ru": "v5_cis_base",
    "en": "v3_en",
    "de": "v3_de",
    "es": "v3_es",
    "fr": "v3_fr",
    # CIS languages (v5_cis_base covers all CIS)
    "ba": "v5_cis_base",  # Bashkir
    "kk": "v5_cis_base",  # Kazakh
    "tt": "v5_cis_base",  # Tatar
    "ua": "v5_cis_base",  # Ukrainian
    "uz": "v5_cis_base",  # Uzbek
    "cy": "v5_cis_base",  # Kyrgyz
    "xal": "v5_cis_base",  # Kalmyk
    # Indic
    "indic": "v4_indic",
}

# Default speakers per language model.
_SILERO_DEFAULT_SPEAKERS: dict[str, str] = {
    "ru": "ru_karina",
    "en": "en_0",
    "de": "karlsson",
    "es": "es_0",
    "fr": "fr_0",
}

# Speakers from older models (v4_ru) that don't exist in v5_cis_base.
# Map them to the closest available speaker.
_SILERO_SPEAKER_COMPAT: dict[str, str] = {
    "xenia": "ru_karina",
    "aidar": "ru_marat",
    "baya": "ru_aigul",
    "kseniya": "ru_karina",
    "eugene": "ru_eduard",
}


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


# ── Language detection for dual-model routing ──────────────────
# Characters in the Basic Latin block (ASCII letters, digits, common
# punctuation) are routed to the English model.  Everything else
# (Cyrillic, CJK, Arabic, Devanagari, …) goes to the user's language model.
_LATIN_RUN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9 ,.\-:;!?'\"()]*[A-Za-z0-9.,!?'\")]|[A-Za-z]"
)
_NON_LATIN_RUN = re.compile(
    r"[^\x00-\x7F][^\x00-\x7F ,.\-:;!?'\"()]*[^\x00-\x7F.,!?'\")]|[^\x00-\x7F]"
)


@dataclass
class _TextSegment:
    """A piece of text tagged with a language for model routing."""

    text: str
    lang: str  # "en" or the user's configured language code


def _split_by_script(text: str, user_lang: str) -> list[_TextSegment]:
    """Split *text* into alternating Latin (→ English) and non-Latin (→ user lang) runs.

    Punctuation and whitespace between runs attach to the preceding segment.
    If user_lang is ``"en"``, everything goes to the English model.
    """
    if user_lang == "en":
        return [_TextSegment(text=text, lang="en")]

    segments: list[_TextSegment] = []
    pos = 0
    length = len(text)

    while pos < length:
        lat = _LATIN_RUN.search(text, pos)
        nlat = _NON_LATIN_RUN.search(text, pos)

        # Pick whichever match starts first
        if lat and (nlat is None or lat.start() <= nlat.start()):
            # Include any skipped whitespace/punctuation before this match
            prefix = text[pos : lat.start()]
            seg_text = prefix + lat.group()
            segments.append(_TextSegment(text=seg_text, lang="en"))
            pos = lat.end()
        elif nlat:
            prefix = text[pos : nlat.start()]
            seg_text = prefix + nlat.group()
            segments.append(_TextSegment(text=seg_text, lang=user_lang))
            pos = nlat.end()
        else:
            # Only whitespace/punctuation left — attach to last segment
            tail = text[pos:]
            if segments:
                segments[-1].text += tail
            else:
                segments.append(_TextSegment(text=tail, lang=user_lang))
            break

    # Merge adjacent segments with the same language
    merged: list[_TextSegment] = []
    for seg in segments:
        if merged and merged[-1].lang == seg.lang:
            merged[-1].text += seg.text
        else:
            merged.append(seg)

    return merged


def _sanitize_text(text: str, lang: str, allowed_chars: set[str] | None = None) -> str:
    """Sanitize *text* for Silero TTS.

    The only transformation is lowercasing for English (the ``v3_en``
    model only accepts lowercase).  Everything else is handled by
    stripping characters not in the model's ``symbols`` set — this is
    the safest approach because each model version defines its own
    character vocabulary, and hardcoded replacement tables inevitably
    go out of sync or produce awkward output (e.g. digit-by-digit
    reading instead of proper number pronunciation).
    """
    if lang == "en":
        text = text.lower()

    # Drop any character the model can't handle
    if allowed_chars:
        text = "".join(ch for ch in text if ch in allowed_chars)

    text = re.sub(r"  +", " ", text)
    return text.strip()


# ── Silero TTS (default) ────────────────────────────────────────


class SileroTTS(TTSBackend):
    """Fast offline TTS using Silero Models (torch.hub).

    Loads **two** models when the user's language is not English:
    one for English text and one for the user's selected language.
    Text is split by script (Latin → English model, non-Latin → user
    language model) so mixed-language output is spoken naturally.

    When the user's language *is* English, only one model is loaded.

    Parameters
    ----------
    language:
        Language code (``"ru"``, ``"en"``, ``"de"``, …).  Default ``"ru"``.
    speaker:
        Speaker name for the user's language model.  ``None`` = default.
    en_speaker:
        Speaker name for the English model.  ``None`` = ``"en_0"``.
    sample_rate_hz:
        Output sample rate: 8000, 24000, or 48000.  Default 48000.
    """

    def __init__(
        self,
        language: str = "ru",
        speaker: str | None = None,
        en_speaker: str | None = None,
        sample_rate_hz: int = 48_000,
    ) -> None:
        self._language = language
        self._speaker = speaker
        self._en_speaker = en_speaker or _SILERO_DEFAULT_SPEAKERS.get("en", "en_0")
        self._rate = sample_rate_hz
        # Two model slots: user language and English.
        self._model_user: object | None = None
        self._model_en: object | None = None
        self._models_ready = False
        # Allowed character sets (populated from model.symbols on load).
        self._user_chars: set[str] | None = None
        self._en_chars: set[str] | None = None

    def _ensure_models(self) -> None:
        if self._models_ready:
            return

        import torch

        _root_level = logging.getLogger().getEffectiveLevel()
        if _root_level > logging.INFO:
            import os

            os.environ.setdefault("MIOPEN_LOG_LEVEL", "0")
        if _root_level > logging.DEBUG:
            logging.getLogger("httpx").setLevel(logging.WARNING)

        # --- Load user's language model ---
        user_model_id = _SILERO_LANG_MODELS.get(self._language)
        if user_model_id is None:
            raise ValueError(
                f"Silero TTS does not support language {self._language!r}.  "
                f"Supported: {', '.join(sorted(_SILERO_LANG_MODELS))}"
            )

        logger.info(
            "Loading Silero TTS model %s (lang=%s)", user_model_id, self._language
        )
        self._model_user, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=self._language,
            speaker=user_model_id,
        )

        # Resolve speaker — apply compat mapping for old v4_ru names,
        # then validate against what the loaded model actually supports.
        if self._speaker is None:
            self._speaker = _SILERO_DEFAULT_SPEAKERS.get(self._language, "ru_karina")
        if self._speaker in _SILERO_SPEAKER_COMPAT:
            old = self._speaker
            self._speaker = _SILERO_SPEAKER_COMPAT[old]
            logger.info("Speaker %s → %s (compat)", old, self._speaker)
        if hasattr(self._model_user, "speakers") and self._speaker not in self._model_user.speakers:
            fallback = _SILERO_DEFAULT_SPEAKERS.get(self._language, self._model_user.speakers[0])
            logger.warning(
                "Speaker %s not in model, falling back to %s", self._speaker, fallback,
            )
            self._speaker = fallback
        # Extract allowed character set from model
        if hasattr(self._model_user, "symbols"):
            self._user_chars = set(self._model_user.symbols)
        logger.info("Silero %s ready (speaker=%s)", self._language, self._speaker)

        # --- Load English model (if user language is not English) ---
        if self._language != "en":
            en_model_id = _SILERO_LANG_MODELS["en"]
            logger.info("Loading Silero TTS model %s (lang=en)", en_model_id)
            self._model_en, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="en",
                speaker=en_model_id,
            )
            # Validate en speaker too
            if hasattr(self._model_en, "speakers") and self._en_speaker not in self._model_en.speakers:
                self._en_speaker = _SILERO_DEFAULT_SPEAKERS.get("en", "en_0")
            if hasattr(self._model_en, "symbols"):
                self._en_chars = set(self._model_en.symbols)
            logger.info("Silero en ready (speaker=%s)", self._en_speaker)
        else:
            self._model_en = self._model_user
            self._en_chars = self._user_chars

        self._models_ready = True

    def _get_model_and_speaker(self, lang: str) -> tuple[object, str, set[str] | None]:
        """Return (model, speaker, allowed_chars) for the given language code."""
        if lang == "en":
            return self._model_en, self._en_speaker, self._en_chars  # type: ignore[return-value]
        return self._model_user, self._speaker or "ru_karina", self._user_chars  # type: ignore[return-value]

    @property
    def sample_rate(self) -> int:
        return self._rate

    async def synthesize(self, text: str) -> bytes:
        import numpy as np

        if not text or not text.strip():
            return self._silent_wav()

        self._ensure_models()
        segments = _split_by_script(text.strip(), self._language)

        all_samples: list[np.ndarray] = []

        for seg in segments:
            model, speaker, allowed = self._get_model_and_speaker(seg.lang)
            safe = _sanitize_text(seg.text, seg.lang, allowed)
            if not safe:
                continue
            # Silero crashes on text with no letters (just digits/punct).
            # Skip segments that have no alphabetic content.
            if not any(ch.isalpha() for ch in safe):
                continue

            def _run(m: object = model, t: str = safe, s: str = speaker) -> np.ndarray:
                import torch

                with torch.inference_mode():
                    audio = m.apply_tts(text=t, speaker=s, sample_rate=self._rate)  # type: ignore[union-attr]
                if hasattr(audio, "numpy"):
                    return audio.cpu().numpy()
                return np.array(audio, dtype=np.float32)

            samples = await asyncio.to_thread(_run)
            all_samples.append(samples)

        if not all_samples:
            return self._silent_wav()

        combined = np.concatenate(all_samples)
        int_samples = (combined * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._rate)
            wf.writeframes(int_samples.tobytes())
        return buf.getvalue()

    async def speak(self, text: str) -> None:
        if not text or not text.strip():
            return

        import numpy as np
        import sounddevice as sd

        wav_bytes = await self.synthesize(text)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            rate = wf.getframerate()

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        def _play_and_wait() -> None:
            sd.play(samples, rate)
            sd.wait()

        await asyncio.to_thread(_play_and_wait)

    async def close(self) -> None:
        self._model_user = None
        self._model_en = None
        self._models_ready = False

    def _silent_wav(self) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._rate)
            wf.writeframes(b"\x00\x00" * (self._rate // 2))
        return buf.getvalue()


# ── OpenAI TTS (cloud) ──────────────────────────────────────────


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
        self._sample_rate = 24_000

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

        def _play_and_wait() -> None:
            sd.play(samples, rate)
            sd.wait()  # block until playback finishes

        await asyncio.to_thread(_play_and_wait)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()  # type: ignore[union-attr]
            self._client = None


# ── Factory ─────────────────────────────────────────────────────


def create_tts_backend(
    backend: str = "local",
    *,
    voice: str = "alloy",
    model: str = "tts-1",
    api_key: str | None = None,
    speaker: str | None = None,
    en_speaker: str | None = None,
    language: str | None = None,
) -> TTSBackend:
    """Factory for creating TTS backends.

    Parameters
    ----------
    backend:
        ``"local"`` (default) for Silero TTS (fast, multilingual),
        ``"openai"`` for OpenAI API.
    voice:
        Voice ID (openai only).
    model:
        OpenAI model name (``"tts-1"`` or ``"tts-1-hd"``).
    api_key:
        OpenAI API key (openai backend only).
    speaker:
        Silero speaker name for the user's language (e.g. ``"ru_karina"``).
    en_speaker:
        Silero speaker name for the English model (e.g. ``"en_0"``).
    language:
        Language code (``"ru"``, ``"en"``, …).  Determines the Silero model.
    """
    match backend:
        case "local":
            return SileroTTS(
                language=language or "ru",
                speaker=speaker,
                en_speaker=en_speaker,
            )
        case "openai":
            return OpenAITTS(
                api_key=api_key,
                model=model,
                voice=voice,
            )
        case _:
            raise ValueError(f"Unknown TTS backend: {backend!r}")
