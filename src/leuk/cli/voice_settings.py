"""Voice settings dialog for the leuk REPL.

Provides an interactive menu to configure STT/TTS backends, models,
language, speaker, etc.  Settings are persisted to config.json.
"""

from __future__ import annotations

from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import radiolist_dialog
from prompt_toolkit.styles import Style

_DIALOG_STYLE = Style.from_dict(
    {
        "dialog": "bg:#1a1a2e",
        "dialog frame.label": "bg:#16213e #e0e0e0 bold",
        "dialog.body": "bg:#1a1a2e #e0e0e0",
        "dialog shadow": "bg:#0f0f0f",
        "button": "bg:#16213e #e0e0e0",
        "button.focused": "bg:#0f3460 #ffffff bold",
        "radio-list": "bg:#1a1a2e #e0e0e0",
        "radio": "#00aa00",
        "radio-checked": "#00ff00 bold",
    }
)

# ── STT models ────────────────────────────────────────────────────

# (value, display_label) — ordered from fastest/smallest to best quality.
STT_MODELS: list[tuple[str, str]] = [
    ("tiny", "  tiny          — fastest, lowest accuracy (~40 MB)"),
    ("base", "  base          — fast, fair accuracy (~150 MB)"),
    ("small", "  small         — balanced speed/quality (~500 MB)"),
    ("medium", "  medium        — good quality, slower (~1.5 GB)"),
    ("turbo", "  turbo         — best speed/quality trade-off (~800 MB)"),
    ("large-v3", "  large-v3      — highest accuracy, slowest (~3 GB)"),
]

# ── TTS models ────────────────────────────────────────────────────

TTS_MODELS: list[tuple[str, str]] = [
    (
        "tts_models/multilingual/multi-dataset/xtts_v2",
        "  XTTSv2      — multilingual, multi-speaker, slow (~1.9 GB)",
    ),
    (
        "tts_models/en/vctk/vits",
        "  VITS (VCTK)   — English multi-speaker, very fast (~120 MB)",
    ),
    (
        "tts_models/en/ljspeech/vits",
        "  VITS (LJS)    — English single-speaker, very fast (~120 MB)",
    ),
    (
        "tts_models/en/ljspeech/tacotron2-DDC",
        "  Tacotron2-DDC — English, medium speed (~100 MB)",
    ),
    (
        "tts_models/en/ljspeech/fast_pitch",
        "  FastPitch     — English, fast (~100 MB)",
    ),
    (
        "tts_models/en/jenny/jenny",
        "  Jenny         — English, natural female voice (~800 MB)",
    ),
]

# Languages supported by XTTSv2 (and used for STT language hint).
LANGUAGES: list[tuple[str, str]] = [
    ("", "  (auto-detect)"),
    ("en", "  English"),
    ("es", "  Español"),
    ("fr", "  Français"),
    ("de", "  Deutsch"),
    ("it", "  Italiano"),
    ("pt", "  Português"),
    ("pl", "  Polski"),
    ("tr", "  Türkçe"),
    ("ru", "  Русский"),
    ("nl", "  Nederlands"),
    ("cs", "  Čeština"),
    ("ar", "  العربية"),
    ("zh-cn", "  中文"),
    ("hu", "  Magyar"),
    ("ko", "  한국어"),
    ("ja", "  日本語"),
    ("hi", "  हिन्दी"),
]

# Speakers for XTTSv2 — curated subset with clearer voices.
XTTS_SPEAKERS: list[tuple[str, str]] = [
    ("Claribel Dervla", "  Claribel Dervla      (female)"),
    ("Daisy Studious", "  Daisy Studious       (female)"),
    ("Gracie Wise", "  Gracie Wise          (female)"),
    ("Sofia Hellen", "  Sofia Hellen         (female)"),
    ("Nova Hogarth", "  Nova Hogarth         (female)"),
    ("Alma María", "  Alma María           (female)"),
    ("Lilya Stainthorpe", "  Lilya Stainthorpe    (female)"),
    ("Camilla Holmström", "  Camilla Holmström    (female)"),
    ("Andrew Chipper", "  Andrew Chipper       (male)"),
    ("Craig Gutsy", "  Craig Gutsy          (male)"),
    ("Damien Black", "  Damien Black         (male)"),
    ("Gilberto Mathias", "  Gilberto Mathias     (male)"),
    ("Viktor Eka", "  Viktor Eka           (male)"),
    ("Baldur Sanjin", "  Baldur Sanjin        (male)"),
    ("Eugenio Mataracı", "  Eugenio Mataracı     (male)"),
    ("Kumar Dahl", "  Kumar Dahl           (male)"),
    ("Filip Traverse", "  Filip Traverse       (male)"),
]


def _radio(
    title: str,
    text: str,
    values: list[tuple[str, str]],
    default: str | None,
) -> str | None:
    """Show a radiolist dialog and return the selected value (or None)."""
    return radiolist_dialog(
        title=HTML(f"<b>{title}</b>"),
        text=HTML(text),
        values=values,
        default=default,
        style=_DIALOG_STYLE,
    ).run()


def run_voice_settings(current: dict[str, str | None]) -> dict[str, str | None] | None:
    """Show the voice settings menu.  Blocking (run via ``asyncio.to_thread``).

    *current* is the existing config dict (from ``load_persistent_config()``).
    Returns updated settings dict, or ``None`` if the user cancels at any step.
    """
    updates: dict[str, str | None] = {}

    # ── 1. STT model ─────────────────────────────────────────────
    cur_stt = current.get("stt_model_size") or "turbo"
    stt = _radio(
        "Speech-to-Text Model (Whisper)",
        "Larger models are more accurate but use more VRAM and are slower.\n"
        "<b>turbo</b> is recommended for GPU users.",
        STT_MODELS,
        cur_stt,
    )
    if stt is None:
        return None
    updates["stt_model_size"] = stt

    # ── 2. Language ──────────────────────────────────────────────
    cur_lang = current.get("stt_language") or ""
    lang = _radio(
        "Voice Language",
        "Used for both speech recognition (STT) and text-to-speech (TTS).\n"
        "Setting a language improves STT accuracy for non-English speech.",
        LANGUAGES,
        cur_lang,
    )
    if lang is None:
        return None
    updates["stt_language"] = lang or None
    updates["tts_language"] = lang or "en"

    # ── 3. TTS model ─────────────────────────────────────────────
    cur_tts = current.get("tts_model_name") or "tts_models/multilingual/multi-dataset/xtts_v2"
    tts = _radio(
        "Text-to-Speech Model",
        "XTTSv2 supports many languages but is <b>slow</b>.\n"
        "VITS models are <b>very fast</b> but English-only.",
        TTS_MODELS,
        cur_tts,
    )
    if tts is None:
        return None
    updates["tts_model_name"] = tts

    # ── 4. Speaker (only for multi-speaker models) ───────────────
    is_xtts = "xtts" in tts.lower()
    if is_xtts:
        cur_speaker = current.get("tts_speaker") or "Claribel Dervla"
        speaker = _radio(
            "TTS Speaker Voice",
            "Choose a speaker voice for XTTSv2.",
            XTTS_SPEAKERS,
            cur_speaker,
        )
        if speaker is None:
            return None
        updates["tts_speaker"] = speaker
    else:
        updates["tts_speaker"] = None

    return updates
