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
import threading
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


# ── Spoken-form normalization (numbers + acronyms) ──────────────
# Silero's character vocabulary has **no digits**, so a raw number is dropped
# entirely and comes out silent. We spell numbers as words (via num2words) and
# read ALL-CAPS acronyms letter-by-letter, *before* the text is split by script
# and synthesized. The language of each number is **detected from the text
# surrounding it** (not the configured voice), the same way the dual-model TTS
# routes by script: a number among Cyrillic words is spelled in the user's
# language, a number among Latin words in English — so e.g. an English reply
# spoken by a Russian-configured voice still reads its numbers in English.

# Configured Silero language → num2words language code. Languages num2words
# doesn't implement fall back to a close relative (CIS → Russian) or English.
_NUM2WORDS_LANG: dict[str, str] = {
    "ru": "ru",
    "en": "en",
    "de": "de",
    "es": "es",
    "fr": "fr",
    "ua": "uk",  # Ukrainian
}
# CIS languages share Silero's Russian-based ``v5_cis_base`` model; spell their
# numbers in Russian (num2words has no per-CIS-language support).
_CIS_LANGS = frozenset({"ba", "kk", "tt", "uz", "cy", "xal"})
# Latin-script Silero languages: a number among Latin (ASCII) words is spelled in
# the configured language when that language is itself Latin-script, otherwise in
# English (the script splitter routes Latin runs to the English model).
_LATIN_LANGS = frozenset({"en", "de", "es", "fr"})

# A contiguous number: digits with optional ``.``/``,`` group/decimal separators.
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
# A run of letters (any script). ``[^\W\d_]`` is "letter" under Unicode ``re``.
_WORD_RE = re.compile(r"[^\W\d_]+")


def _num2words_lang(lang: str) -> str:
    """Map a configured voice language to a num2words language code."""
    if lang in _NUM2WORDS_LANG:
        return _NUM2WORDS_LANG[lang]
    if lang in _CIS_LANGS:
        return "ru"
    return "en"


def _nearest_letter(text: str, start: int, end: int) -> str | None:
    """The alphabetic character closest to the span ``[start, end)`` (the number),
    scanning outward in both directions; ``None`` if there are no letters."""
    left = next(
        ((start - 1 - i, text[i]) for i in range(start - 1, -1, -1) if text[i].isalpha()),
        None,
    )
    right = next(
        ((i - end, text[i]) for i in range(end, len(text)) if text[i].isalpha()),
        None,
    )
    if left and right:
        return left[1] if left[0] <= right[0] else right[1]
    return left[1] if left else (right[1] if right else None)


def _detected_number_lang(text: str, start: int, end: int, user_lang: str) -> str:
    """Detect the language to spell a number in, from the script of the text that
    surrounds it. Falls back to the configured *user_lang* when there's no
    alphabetic context (e.g. a bare "5")."""
    ch = _nearest_letter(text, start, end)
    if ch is None:
        return user_lang
    if ch.isascii():  # Latin context
        return user_lang if user_lang in _LATIN_LANGS else "en"
    # Non-Latin (e.g. Cyrillic) context → the user's language, or Russian if the
    # configured voice is Latin-script (num2words only covers ru/uk non-Latin).
    return user_lang if user_lang not in _LATIN_LANGS else "ru"


def _parse_number(token: str) -> int | float | None:
    """Parse a number token to an int/float, disambiguating ``.``/``,`` as
    decimal vs. thousands separators. Heuristic: the *last* separator is the
    decimal point unless the trailing group is exactly 3 digits (grouping)."""
    s = token
    has_dot = "." in s
    has_comma = "," in s
    if has_dot and has_comma:
        if s.rfind(",") > s.rfind("."):  # "1.234,56" → comma is decimal
            s = s.replace(".", "").replace(",", ".")
        else:  # "1,234.56" → dot is decimal
            s = s.replace(",", "")
    elif has_comma:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) != 3:
            s = s.replace(",", ".")  # "3,14" → decimal
        else:
            s = s.replace(",", "")  # "1,000" / "1,000,000" → grouping
    elif has_dot:
        parts = s.split(".")
        if not (len(parts) == 2 and len(parts[1]) != 3):
            s = s.replace(".", "")  # "1.000" → grouping (else keep as decimal)
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return None


def _spell_numbers(text: str, lang: str) -> str:
    """Replace number tokens with their spelled-out words, each in the language
    *detected from the surrounding text* (with *lang* as the fallback)."""
    try:
        from num2words import num2words
    except ImportError:
        return text  # graceful: leave numbers as-is if the dep is missing

    def _repl(m: re.Match[str]) -> str:
        value = _parse_number(m.group(0))
        if value is None:
            return m.group(0)
        detected = _detected_number_lang(text, m.start(), m.end(), lang)
        n2w = _num2words_lang(detected)
        for code in (n2w, "en"):
            try:
                return num2words(value, lang=code).replace("-", " ")
            except Exception:  # noqa: BLE001 — unsupported lang/value → next fallback
                continue
        return m.group(0)

    return _NUMBER_RE.sub(_repl, text)


# Spoken names of individual letters, so an acronym is read out as letters the
# TTS can actually pronounce (a bare spaced "c a a" comes out as run-together
# phonemes, not "see ay ay"). Latin letters route to the English model and
# Cyrillic to the user-language model, so those two alphabets are all we need.
_LETTER_NAMES: dict[str, str] = {
    # Latin (English letter names)
    "A": "ay", "B": "bee", "C": "see", "D": "dee", "E": "ee", "F": "eff",
    "G": "gee", "H": "aitch", "I": "eye", "J": "jay", "K": "kay", "L": "el",
    "M": "em", "N": "en", "O": "oh", "P": "pee", "Q": "cue", "R": "ar",
    "S": "ess", "T": "tee", "U": "you", "V": "vee", "W": "double-you",
    "X": "ex", "Y": "why", "Z": "zee",
    # Cyrillic (Russian letter names)
    "А": "а", "Б": "бэ", "В": "вэ", "Г": "гэ", "Д": "дэ", "Е": "е", "Ж": "жэ",
    "З": "зэ", "И": "и", "Й": "и краткое", "К": "ка", "Л": "эль", "М": "эм",
    "Н": "эн", "О": "о", "П": "пэ", "Р": "эр", "С": "эс", "Т": "тэ", "У": "у",
    "Ф": "эф", "Х": "ха", "Ц": "цэ", "Ч": "че", "Ш": "ша", "Щ": "ща",
    "Ъ": "твёрдый знак", "Ы": "ы", "Ь": "мягкий знак", "Э": "э", "Ю": "ю",
    "Я": "я",
}


def _spell_acronyms(text: str) -> str:
    """Read ALL-CAPS letter runs (acronyms) out as their spoken letter names,
    e.g. ``API`` → ``ay pee eye``, ``ФСБ`` → ``эф эс бэ``. Each letter is mapped
    to a pronounceable name so the TTS doesn't slur the bare letters together.
    Works in any script (``str.isupper`` is Unicode-aware)."""

    def _repl(m: re.Match[str]) -> str:
        word = m.group(0)
        if len(word) >= 2 and word.isupper():
            return " ".join(_LETTER_NAMES.get(ch, ch) for ch in word)
        return word

    return _WORD_RE.sub(_repl, text)


def normalize_for_speech(text: str, lang: str) -> str:
    """Normalize *text* into a fully pronounceable form: spell ALL-CAPS acronyms
    letter-by-letter and numbers as words. Numbers are spelled in the language
    detected from the text around them (the script of the neighbouring words),
    using *lang* — the configured voice language — only as a fallback."""
    text = _spell_acronyms(text)
    text = _spell_numbers(text, lang)
    return text


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
    async def speak(
        self,
        text: str,
        stop_event: "threading.Event | None" = None,
    ) -> None:
        """Synthesize and play text through speakers.

        Playback runs in a background thread. If *stop_event* is set mid-playback
        (e.g. an interrupt), playback stops promptly between audio blocks.
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

    # Map characters the model can't handle. Many Silero models expose only
    # lowercase letters in ``symbols``; the previous code *dropped* anything not
    # in the set, which silently ate every sentence-initial capital (the "first
    # letter is inaudible" bug). Instead, fall back to the lowercase form before
    # dropping, so capitalised words are still pronounced.
    if allowed_chars:
        out: list[str] = []
        for ch in text:
            if ch in allowed_chars:
                out.append(ch)
            elif ch.lower() in allowed_chars:
                out.append(ch.lower())
            # else: genuinely unsupported — drop it
        text = "".join(out)

    text = re.sub(r"  +", " ", text)
    return text.strip()


# Silence padded around each spoken sentence: a lead-in so the device's start-up
# latency doesn't clip the first phoneme, and a trailing pause so consecutive
# sentences / lines don't run together.
_LEADIN_SILENCE_S = 0.06
_TRAILING_PAUSE_S = 0.18

def _play_blocking(
    samples: "np.ndarray",
    rate: int,
    *,
    stop_event: "threading.Event | None" = None,
    blocksize: int = 1024,
) -> None:
    """Play *samples* via an explicit, thread-owned ``OutputStream``.

    Unlike ``sd.play()``/``sd.stop()`` (which share a global stream and crash
    PortAudio when stopped from another thread mid-playback), this opens, writes,
    and closes the stream entirely on the calling (worker) thread. Interruption
    is signalled by *stop_event* — a plain :class:`threading.Event` set by any
    thread — checked between block writes, so no PortAudio call ever crosses
    threads (which would double-free PortAudio and crash the process).
    """
    import numpy as np
    import sounddevice as sd

    data = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
    # Pad with a short lead-in (so the audio device's start-up latency doesn't
    # swallow the first phoneme — the "first letter cut off" bug) and a trailing
    # silence (a natural pause before the next sentence / line).
    lead = int(_LEADIN_SILENCE_S * rate)
    tail = int(_TRAILING_PAUSE_S * rate)
    if lead or tail:
        data = np.concatenate(
            [np.zeros(lead, dtype=np.float32), data, np.zeros(tail, dtype=np.float32)]
        )
    data = data.reshape(-1, 1)
    stream = sd.OutputStream(samplerate=rate, channels=data.shape[1], dtype="float32")
    stream.start()
    try:
        for i in range(0, len(data), blocksize):
            if stop_event is not None and stop_event.is_set():
                break
            stream.write(data[i : i + blocksize])
    finally:
        # stop() drains the small in-flight buffer then halts; close() frees the
        # stream — both on this thread, so PortAudio is never touched concurrently.
        try:
            stream.stop()
        except Exception:  # noqa: BLE001
            pass
        stream.close()


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
        from leuk.voice.recorder import suppress_model_load_noise

        with suppress_model_load_noise():
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
            from leuk.voice.recorder import suppress_model_load_noise

            with suppress_model_load_noise():
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
        # Spell numbers as words and acronyms letter-by-letter *before* splitting
        # by script — Silero has no digits in its vocab, so raw numbers are silent.
        normalized = normalize_for_speech(text.strip(), self._language)
        segments = _split_by_script(normalized, self._language)

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

    async def speak(
        self,
        text: str,
        stop_event: "threading.Event | None" = None,
    ) -> None:
        if not text or not text.strip():
            return

        import numpy as np

        wav_bytes = await self.synthesize(text)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            rate = wf.getframerate()

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        await asyncio.to_thread(
            _play_blocking,
            samples,
            rate,
            stop_event=stop_event,
        )

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

    async def speak(
        self,
        text: str,
        stop_event: "threading.Event | None" = None,
    ) -> None:
        import numpy as np

        wav_bytes = await self.synthesize(text)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            rate = wf.getframerate()

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        await asyncio.to_thread(
            _play_blocking,
            samples,
            rate,
            stop_event=stop_event,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()  # type: ignore[union-attr]
            self._client = None


# ── Streaming TTS speaker ─────────────────────────────────────

# A sentence boundary: ending punctuation followed by whitespace/end-of-string,
# OR any line break. Treating each line (bullets, list items, paragraphs) as its
# own utterance means it's spoken separately, with a natural pause between lines.
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)|\n+")

# Sentinel pushed to the sentence queue to signal shutdown.
_STOP_SENTINEL = object()


class StreamingTTSSpeaker:
    """Queues sentence-by-sentence TTS while text streams in real-time.

    Usage::

        speaker = StreamingTTSSpeaker(tts_backend)
        await speaker.start()

        # During streaming — called from StreamRenderer._on_text_delta:
        speaker.feed(token)
        speaker.feed(token)
        ...

        # After streaming completes:
        await speaker.flush()   # speak remaining buffer, wait for playback
        await speaker.stop()

    The playback loop runs as a background ``asyncio.Task``.  Sentences are
    spoken sequentially (each clip finishes before the next starts).
    """

    def __init__(self, backend: TTSBackend) -> None:
        self._backend = backend
        self._buffer = ""
        self._sentence_queue: asyncio.Queue[str | object] = asyncio.Queue()
        self._playback_task: asyncio.Task[None] | None = None
        self._stopped = False
        # Set on stop(); the playback worker checks it between audio blocks and
        # halts — no cross-thread PortAudio call (which would crash). This is how
        # a Ctrl-C interrupt cuts off the currently-playing clip promptly.
        self._interrupt = threading.Event()

    async def start(self) -> None:
        """Start the background playback loop."""
        self._stopped = False
        self._interrupt.clear()
        self._playback_task = asyncio.create_task(
            self._playback_loop(), name="streaming-tts"
        )

    def feed(self, token: str) -> None:
        """Feed a text delta token.  Complete sentences are queued for TTS."""
        self._buffer += token
        while True:
            match = _SENTENCE_END_RE.search(self._buffer)
            if match is None:
                break
            sentence = self._buffer[: match.end()]
            self._buffer = self._buffer[match.end() :]
            cleaned = clean_text_for_speech(sentence)
            if cleaned.strip():
                self._sentence_queue.put_nowait(cleaned)

    async def flush(self) -> None:
        """Speak any remaining buffered text, then wait for the queue to drain."""
        if self._buffer.strip():
            cleaned = clean_text_for_speech(self._buffer)
            self._buffer = ""
            if cleaned.strip():
                self._sentence_queue.put_nowait(cleaned)
        # Signal end and wait for the loop to finish all pending sentences.
        self._sentence_queue.put_nowait(_STOP_SENTINEL)
        if self._playback_task is not None:
            await self._playback_task

    async def stop(self) -> None:
        """Cancel playback immediately and discard pending sentences.

        Signals the playback worker (via a :class:`threading.Event`) to stop
        writing audio between blocks, so the currently-playing clip is cut off
        promptly **without** any cross-thread PortAudio call — calling
        ``sd.stop()`` from here while a worker thread is mid-playback double-frees
        PortAudio's stream and crashes the process.
        """
        self._stopped = True
        self._interrupt.set()
        if self._playback_task is not None and not self._playback_task.done():
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
        self._playback_task = None
        self._buffer = ""

    async def _playback_loop(self) -> None:
        """Background loop: dequeue sentences and speak them sequentially."""
        while not self._stopped:
            item = await self._sentence_queue.get()
            if item is _STOP_SENTINEL:
                self._sentence_queue.task_done()
                break
            try:
                await self._backend.speak(item, self._interrupt)  # type: ignore[arg-type]
            except Exception:
                logger.debug("Streaming TTS playback error", exc_info=True)
            finally:
                self._sentence_queue.task_done()


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
