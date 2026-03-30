"""Unified settings dialog for the leuk REPL.

Provides a tabbed interface to configure all user-facing settings
persisted in ``~/.config/leuk/config.json``.  The dialog is blocking
and designed to run via ``asyncio.to_thread(run_settings, ...)``.

Tabs
~~~~
* **General** — read-only display of current provider and model.
* **Speech-to-Text** — STT model size, recognition language.
* **Text-to-Speech** — TTS language, voice/speaker for both user
  language and English.
* **Voice Activity** — VAD sensitivity, silence timeout, minimum
  speech duration (shown with visual slider bars).
"""

from __future__ import annotations

from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.shortcuts import message_dialog
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Button, Dialog, Label, RadioList

# ── Shared dialog style ─────────────────────────────────────────

DIALOG_STYLE = Style.from_dict(
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

# ── Option lists ────────────────────────────────────────────────

BACKENDS: list[tuple[str, str]] = [
    ("local", "local         -- offline, uses GPU/CPU (Whisper / Silero)"),
    ("openai", "openai        -- cloud, uses OpenAI API key"),
]

OPENAI_TTS_VOICES: list[tuple[str, str]] = [
    ("alloy", "alloy         -- neutral, balanced"),
    ("echo", "echo          -- warm, confident"),
    ("fable", "fable         -- expressive, storytelling"),
    ("nova", "nova          -- friendly, energetic"),
    ("onyx", "onyx          -- deep, authoritative"),
    ("shimmer", "shimmer       -- clear, optimistic"),
]

STT_MODELS: list[tuple[str, str]] = [
    ("tiny", "tiny          -- fastest, lowest accuracy (~40 MB)"),
    ("base", "base          -- fast, fair accuracy (~150 MB)"),
    ("small", "small         -- balanced speed/quality (~500 MB)"),
    ("medium", "medium        -- good quality, slower (~1.5 GB)"),
    ("turbo", "turbo         -- best speed/quality trade-off (~800 MB)"),
    ("large-v3", "large-v3      -- highest accuracy, slowest (~3 GB)"),
]

LANGUAGES: list[tuple[str, str]] = [
    ("", "(auto-detect)"),
    ("en", "English"),
    ("es", "Espanol"),
    ("fr", "Francais"),
    ("de", "Deutsch"),
    ("it", "Italiano"),
    ("pt", "Portugues"),
    ("pl", "Polski"),
    ("tr", "Turkce"),
    ("ru", "Russkiy"),
    ("nl", "Nederlands"),
    ("cs", "Cestina"),
    ("ar", "al-Arabiyyah"),
    ("zh-cn", "Zhongwen"),
    ("hu", "Magyar"),
    ("ko", "Hangugeo"),
    ("ja", "Nihongo"),
    ("hi", "Hindi"),
]

VAD_SENSITIVITY_OPTIONS: list[tuple[str, str]] = [
    ("0.1", "0.1 -- very strict, speech must be loud and clear"),
    ("0.2", "0.2 -- strict, ignores quiet speech"),
    ("0.3", "0.3 -- conservative"),
    ("0.4", "0.4 -- slightly below default"),
    ("0.5", "0.5 -- balanced (default)"),
    ("0.6", "0.6 -- slightly above default"),
    ("0.7", "0.7 -- sensitive, picks up quiet speech"),
    ("0.8", "0.8 -- very sensitive"),
    ("0.9", "0.9 -- extremely sensitive, may trigger on noise"),
]

VAD_SILENCE_TIMEOUT_OPTIONS: list[tuple[str, str]] = [
    ("0.5", "0.5s -- very fast, may cut off mid-sentence"),
    ("0.8", "0.8s -- fast, good for short utterances"),
    ("1.0", "1.0s -- balanced (default)"),
    ("1.5", "1.5s -- patient, better for longer pauses"),
    ("2.0", "2.0s -- very patient, allows long pauses"),
]

VAD_MIN_SPEECH_OPTIONS: list[tuple[str, str]] = [
    ("0.3", "0.3s -- pick up short words"),
    ("0.5", "0.5s -- balanced (default)"),
    ("1.0", "1.0s -- ignore brief sounds, full sentences only"),
]

# ── Silero TTS speaker lists per language ───────────────────────
# Curated from snakers4/silero-models.  Speakers are stable across
# model versions; the fallback logic in SileroTTS._ensure_models()
# handles any mismatches.

SILERO_SPEAKERS: dict[str, list[tuple[str, str]]] = {
    "ru": [
        ("ru_karina", "Karina (female, default)"),
        ("ru_marat", "Marat (male)"),
        ("ru_aigul", "Aigul (female)"),
        ("ru_eduard", "Eduard (male)"),
    ],
    "en": [
        ("en_0", "Speaker 0 (default)"),
        ("en_1", "Speaker 1"),
        ("en_2", "Speaker 2"),
        ("en_3", "Speaker 3"),
        ("en_4", "Speaker 4"),
        ("en_5", "Speaker 5"),
    ],
    "de": [
        ("karlsson", "Karlsson (default)"),
        ("eva_k", "Eva"),
    ],
    "es": [
        ("es_0", "Speaker 0 (default)"),
        ("es_1", "Speaker 1"),
        ("es_2", "Speaker 2"),
    ],
    "fr": [
        ("fr_0", "Speaker 0 (default)"),
        ("fr_1", "Speaker 1"),
        ("fr_2", "Speaker 2"),
        ("fr_3", "Speaker 3"),
        ("fr_4", "Speaker 4"),
        ("fr_5", "Speaker 5"),
    ],
}


# ── Helpers ─────────────────────────────────────────────────────


def _slider_bar(value: float, min_v: float, max_v: float, width: int = 10) -> str:
    """Render a text slider like ``[======----]``."""
    if max_v <= min_v:
        return "[" + "=" * width + "]"
    ratio = (value - min_v) / (max_v - min_v)
    filled = max(0, min(width, round(ratio * width)))
    return "[" + "=" * filled + "-" * (width - filled) + "]"


def _radio(
    title: str,
    text: str,
    values: list[tuple[str, str]],
    default: str | None,
) -> str | None:
    """Show a radiolist dialog and return the selected value (or None).

    Builds the dialog manually (instead of ``radiolist_dialog``) so we can
    add an Esc keybinding — prompt_toolkit's built-in helper omits it.
    """
    radio_list: RadioList[str | None] = RadioList(values=values, default=default)

    def ok_handler() -> None:
        app.exit(result=radio_list.current_value)

    dialog = Dialog(
        title=HTML(f"<b>{title}</b>"),
        body=HSplit([Label(text=HTML(text), dont_extend_height=True), radio_list], padding=1),
        buttons=[
            Button(text="Ok", handler=ok_handler),
            Button(text="Cancel", handler=lambda: app.exit(result=None)),
        ],
        with_background=True,
    )

    bindings = KeyBindings()

    @bindings.add("tab")
    def _tab(event: object) -> None:
        from prompt_toolkit.key_binding.bindings.focus import focus_next
        focus_next(event)  # type: ignore[arg-type]

    @bindings.add("s-tab")
    def _stab(event: object) -> None:
        from prompt_toolkit.key_binding.bindings.focus import focus_previous
        focus_previous(event)  # type: ignore[arg-type]

    @bindings.add("escape")
    def _esc(event: object) -> None:
        app.exit(result=None)

    app: Application[str | None] = Application(
        layout=Layout(dialog),
        key_bindings=merge_key_bindings([load_key_bindings(), bindings]),
        mouse_support=True,
        style=DIALOG_STYLE,
        full_screen=True,
    )
    return app.run()


def _effective(key: str, updates: dict, current: dict, fallback: str = "") -> str:
    """Return the effective value for a config key (updates > current > fallback)."""
    v = updates.get(key)
    if v is not None:
        return str(v)
    v = current.get(key)
    if v is not None:
        return str(v)
    return fallback


def _lang_display(code: str) -> str:
    """Short display name for a language code."""
    for val, label in LANGUAGES:
        if val == code:
            return label
    return code or "(auto)"


def _speaker_display(speaker_id: str, lang: str) -> str:
    """Short display name for a speaker ID."""
    speakers = SILERO_SPEAKERS.get(lang, [])
    for val, label in speakers:
        if val == speaker_id:
            return label
    return speaker_id or "(default)"


# ── Tab: General (read-only) ───────────────────────────────────


def _show_general(current: dict) -> None:
    """Display current provider and model as read-only info."""
    provider = current.get("last_provider", "(not set)")
    model = current.get("last_model", "(not set)")
    message_dialog(
        title=HTML("<b>General</b>"),
        text=(
            f"Provider:  {provider}\n"
            f"Model:     {model}\n"
            "\n"
            "Use /auth to change provider.\n"
            "Use /models to change model."
        ),
        style=DIALOG_STYLE,
    ).run()


# ── Tab: Speech-to-Text ────────────────────────────────────────


def _run_stt_tab(current: dict, updates: dict) -> None:
    """STT settings sub-menu loop."""
    while True:
        cur_backend = _effective("stt_backend", updates, current, "local")
        cur_model = _effective("stt_model_size", updates, current, "turbo")
        cur_lang = _effective("stt_language", updates, current, "")

        items: list[tuple[str, str]] = [
            ("stt_backend", f"  Backend         {cur_backend}"),
        ]
        if cur_backend == "local":
            items.append(("stt_model_size", f"  Whisper Model   {cur_model}"))
        items += [
            ("stt_language", f"  Language         {_lang_display(cur_lang)}"),
            ("back", "  << Back"),
        ]

        choice = _radio("Speech-to-Text", "Select a setting to configure.", items, None)
        if choice is None or choice == "back":
            return

        if choice == "stt_backend":
            result = _radio(
                "STT Backend",
                "<b>local</b> uses Whisper on your GPU/CPU (no internet needed).\n"
                "<b>openai</b> uses the OpenAI Whisper API (requires API key).",
                [(v, "  " + lbl) for v, lbl in BACKENDS],
                cur_backend,
            )
            if result is not None:
                updates["stt_backend"] = result

        elif choice == "stt_model_size":
            result = _radio(
                "STT Model (Whisper)",
                "Larger models are more accurate but use more VRAM.\n"
                "<b>turbo</b> is recommended for GPU, <b>base</b> for CPU.",
                [(v, "  " + lbl) for v, lbl in STT_MODELS],
                cur_model,
            )
            if result is not None:
                updates["stt_model_size"] = result

        elif choice == "stt_language":
            result = _radio(
                "STT Language",
                "Setting a language improves recognition accuracy.\n"
                "Leave as <b>(auto-detect)</b> for multilingual use.",
                [(v, "  " + lbl) for v, lbl in LANGUAGES],
                cur_lang,
            )
            if result is not None:
                updates["stt_language"] = result or None


# ── Tab: Text-to-Speech ────────────────────────────────────────


def _get_speaker_options(lang: str) -> list[tuple[str, str]]:
    """Return speaker options for a language, with a (default) entry."""
    speakers = SILERO_SPEAKERS.get(lang, [])
    if not speakers:
        return [("", "  (default for this language)")]
    return [(v, "  " + lbl) for v, lbl in speakers]


def _run_tts_tab(current: dict, updates: dict) -> None:
    """TTS settings sub-menu loop."""
    while True:
        cur_backend = _effective("tts_backend", updates, current, "local")
        cur_lang = _effective("tts_language", updates, current, "en")
        cur_speaker = _effective("tts_speaker", updates, current, "")
        cur_en_speaker = _effective("tts_en_speaker", updates, current, "en_0")
        cur_voice = _effective("tts_voice", updates, current, "alloy")

        items: list[tuple[str, str]] = [
            ("tts_backend", f"  Backend            {cur_backend}"),
        ]

        if cur_backend == "openai":
            items.append(("tts_voice", f"  OpenAI Voice       {cur_voice}"))
        else:
            # Silero-specific options
            items.append(
                ("tts_language", f"  TTS Language       {_lang_display(cur_lang)}")
            )
            items.append(
                (
                    "tts_speaker",
                    f"  Voice ({cur_lang})      "
                    f"{_speaker_display(cur_speaker, cur_lang)}",
                )
            )
            # Only show English speaker option when user lang is not English
            if cur_lang != "en":
                items.append(
                    (
                        "tts_en_speaker",
                        f"  Voice (en)         "
                        f"{_speaker_display(cur_en_speaker, 'en')}",
                    )
                )

        items.append(("back", "  << Back"))

        choice = _radio("Text-to-Speech", "Select a setting to configure.", items, None)
        if choice is None or choice == "back":
            return

        if choice == "tts_backend":
            result = _radio(
                "TTS Backend",
                "<b>local</b> uses Silero TTS offline (fast, no internet).\n"
                "<b>openai</b> uses the OpenAI TTS API (higher quality, requires API key).",
                [(v, "  " + lbl) for v, lbl in BACKENDS],
                cur_backend,
            )
            if result is not None:
                updates["tts_backend"] = result

        elif choice == "tts_voice":
            result = _radio(
                "OpenAI TTS Voice",
                "Select a voice for OpenAI text-to-speech.",
                [(v, "  " + lbl) for v, lbl in OPENAI_TTS_VOICES],
                cur_voice,
            )
            if result is not None:
                updates["tts_voice"] = result

        elif choice == "tts_language":
            # Filter to languages that Silero TTS actually supports
            tts_languages = [
                (v, "  " + lbl)
                for v, lbl in LANGUAGES
                if v in ("en", "es", "fr", "de", "ru", "it", "pt", "")
                or v in SILERO_SPEAKERS
            ]
            result = _radio(
                "TTS Language",
                "Language for text-to-speech output.\n"
                "Only languages with Silero TTS models are shown.",
                tts_languages,
                cur_lang,
            )
            if result is not None:
                updates["tts_language"] = result or "en"
                # Reset speaker if language changed
                if result != cur_lang:
                    updates.pop("tts_speaker", None)

        elif choice == "tts_speaker":
            options = _get_speaker_options(cur_lang)
            result = _radio(
                f"TTS Voice ({cur_lang})",
                "Select a voice for your language.",
                options,
                cur_speaker,
            )
            if result is not None:
                updates["tts_speaker"] = result

        elif choice == "tts_en_speaker":
            options = _get_speaker_options("en")
            result = _radio(
                "TTS Voice (English)",
                "Select a voice for English text segments.",
                options,
                cur_en_speaker,
            )
            if result is not None:
                updates["tts_en_speaker"] = result


# ── Tab: Voice Activity Detection ──────────────────────────────


def _run_vad_tab(current: dict, updates: dict) -> None:
    """VAD settings sub-menu loop with slider visualization."""
    while True:
        cur_sens = float(_effective("vad_sensitivity", updates, current, "0.5"))
        cur_timeout = float(_effective("vad_silence_timeout", updates, current, "1.0"))
        cur_min = float(_effective("vad_min_speech", updates, current, "0.5"))

        sens_bar = _slider_bar(cur_sens, 0.1, 0.9)
        timeout_bar = _slider_bar(cur_timeout, 0.5, 2.0)
        min_bar = _slider_bar(cur_min, 0.3, 1.0)

        choice = _radio(
            "Voice Activity Detection",
            "Select a setting to configure.\n"
            "Bars show the current value relative to the range.",
            [
                (
                    "vad_sensitivity",
                    f"  Sensitivity        {cur_sens:.1f}  {sens_bar}",
                ),
                (
                    "vad_silence_timeout",
                    f"  Silence Timeout    {cur_timeout:.1f}s {timeout_bar}",
                ),
                (
                    "vad_min_speech",
                    f"  Min Speech         {cur_min:.1f}s {min_bar}",
                ),
                ("back", "  << Back"),
            ],
            None,
        )
        if choice is None or choice == "back":
            return

        if choice == "vad_sensitivity":
            result = _radio(
                "VAD Sensitivity",
                "How sensitive the voice activity detector is to speech.\n"
                "<b>0.5</b> works well for most environments.\n"
                "Increase if your speech is not detected;\n"
                "decrease if background noise triggers false positives.",
                [
                    (v, f"  {_slider_bar(float(v), 0.1, 0.9)}  {lbl}")
                    for v, lbl in VAD_SENSITIVITY_OPTIONS
                ],
                str(cur_sens),
            )
            if result is not None:
                updates["vad_sensitivity"] = result

        elif choice == "vad_silence_timeout":
            result = _radio(
                "Silence Timeout",
                "How long to wait after speech stops before ending the segment.\n"
                "Shorter = faster response. Longer = captures full sentences.",
                [
                    (v, f"  {_slider_bar(float(v), 0.5, 2.0)}  {lbl}")
                    for v, lbl in VAD_SILENCE_TIMEOUT_OPTIONS
                ],
                str(cur_timeout),
            )
            if result is not None:
                updates["vad_silence_timeout"] = result

        elif choice == "vad_min_speech":
            result = _radio(
                "Minimum Speech Duration",
                "Segments shorter than this are discarded as noise.\n"
                "Lower = picks up short words; higher = ignores brief sounds.",
                [
                    (v, f"  {_slider_bar(float(v), 0.3, 1.0)}  {lbl}")
                    for v, lbl in VAD_MIN_SPEECH_OPTIONS
                ],
                str(cur_min),
            )
            if result is not None:
                updates["vad_min_speech"] = result


# ── Main entry point ────────────────────────────────────────────


def run_settings(current: dict[str, Any]) -> dict[str, Any] | None:
    """Show the unified settings dialog.  Blocking.

    *current* is the full config dict from ``load_persistent_config()``.
    Returns a dict of updated keys, or ``None`` if no changes were made.
    """
    updates: dict[str, Any] = {}

    while True:
        # Build main menu items with current-value summaries
        cur_provider = current.get("last_provider", "(not set)")
        cur_model = current.get("last_model", "(not set)")
        cur_stt_backend = _effective("stt_backend", updates, current, "local")
        cur_stt = _effective("stt_model_size", updates, current, "turbo")
        cur_stt_lang = _lang_display(_effective("stt_language", updates, current, ""))
        cur_tts_backend = _effective("tts_backend", updates, current, "local")
        cur_tts_lang = _lang_display(_effective("tts_language", updates, current, "en"))
        cur_tts_voice = _effective("tts_voice", updates, current, "alloy")
        cur_sens = float(_effective("vad_sensitivity", updates, current, "0.5"))
        sens_bar = _slider_bar(cur_sens, 0.1, 0.9)

        stt_summary = f"{cur_stt_backend}"
        if cur_stt_backend == "local":
            stt_summary += f", {cur_stt}"
        stt_summary += f", {cur_stt_lang}"

        tts_summary = f"{cur_tts_backend}"
        if cur_tts_backend == "openai":
            tts_summary += f", {cur_tts_voice}"
        else:
            tts_summary += f", {cur_tts_lang}"

        tab = _radio(
            "Settings",
            "Select a category to configure.\n"
            "Press <b>Esc</b> to save and exit.",
            [
                (
                    "general",
                    f"  General            {cur_provider} / {cur_model}",
                ),
                (
                    "stt",
                    f"  Speech-to-Text     {stt_summary}",
                ),
                (
                    "tts",
                    f"  Text-to-Speech     {tts_summary}",
                ),
                (
                    "vad",
                    f"  Voice Activity     sens: {cur_sens:.1f} {sens_bar}",
                ),
            ],
            None,
        )

        if tab is None:
            # Esc at main menu → save and exit
            return updates if updates else None

        if tab == "general":
            _show_general(current)
        elif tab == "stt":
            _run_stt_tab(current, updates)
        elif tab == "tts":
            _run_tts_tab(current, updates)
        elif tab == "vad":
            _run_vad_tab(current, updates)
