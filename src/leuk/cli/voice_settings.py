"""Voice settings dialog for the leuk REPL.

Provides an interactive menu to configure STT/TTS backends, models,
language, speaker, VAD sensitivity, etc.  Settings are persisted to
config.json.
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

# ── VAD sensitivity ──────────────────────────────────────────────

VAD_SENSITIVITY_OPTIONS: list[tuple[str, str]] = [
    ("0.2", "  Low           — strict, ignores quiet speech"),
    ("0.5", "  Medium        — balanced (default)"),
    ("0.7", "  High          — sensitive, picks up quiet speech"),
    ("0.9", "  Very high     — very sensitive, may trigger on non-speech"),
]

VAD_SILENCE_TIMEOUT_OPTIONS: list[tuple[str, str]] = [
    ("0.5", "  0.5s  — very fast, may cut off mid-sentence"),
    ("0.8", "  0.8s  — fast, good for short utterances"),
    ("1.0", "  1.0s  — balanced (default)"),
    ("1.5", "  1.5s  — patient, better for longer pauses"),
    ("2.0", "  2.0s  — very patient, allows long pauses"),
]

VAD_MIN_SPEECH_OPTIONS: list[tuple[str, str]] = [
    ("0.3", "  0.3s  — pick up short words"),
    ("0.5", "  0.5s  — balanced (default)"),
    ("1.0", "  1.0s  — ignore brief sounds, full sentences only"),
]

# Languages supported by Silero TTS + used for STT language hint.
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

    # ── 3. VAD sensitivity ───────────────────────────────────────
    cur_sens = str(current.get("vad_sensitivity", "0.5"))
    sens = _radio(
        "VAD Sensitivity (Silero VAD)",
        "How sensitive the voice activity detector is to speech.\n"
        "<b>Medium</b> works well for most environments.\n"
        "Increase if your speech is not detected; decrease if background\n"
        "noise triggers transcription.",
        VAD_SENSITIVITY_OPTIONS,
        cur_sens,
    )
    if sens is None:
        return None
    updates["vad_sensitivity"] = sens

    # ── 4. Silence timeout ───────────────────────────────────────
    cur_timeout = str(current.get("vad_silence_timeout", "1.0"))
    timeout = _radio(
        "Silence Timeout",
        "How long to wait after speech stops before ending the segment.\n"
        "Shorter = faster response, but may cut off mid-thought.\n"
        "Longer = captures full sentences with natural pauses.",
        VAD_SILENCE_TIMEOUT_OPTIONS,
        cur_timeout,
    )
    if timeout is None:
        return None
    updates["vad_silence_timeout"] = timeout

    # ── 5. Minimum speech duration ───────────────────────────────
    cur_min = str(current.get("vad_min_speech", "0.5"))
    min_speech = _radio(
        "Minimum Speech Duration",
        "Segments shorter than this are discarded as noise.\n"
        "Lower = picks up short words; higher = ignores brief sounds.",
        VAD_MIN_SPEECH_OPTIONS,
        cur_min,
    )
    if min_speech is None:
        return None
    updates["vad_min_speech"] = min_speech

    return updates
