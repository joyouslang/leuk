"""Unified settings dialog for the leuk REPL.

Provides a tabbed interface to configure all user-facing settings
persisted in ``~/.config/leuk/config.json``.  The dialog is blocking
and designed to run via ``asyncio.to_thread(run_settings, ...)``.

Tabs
~~~~
* **General** — theme and read-only provider/model display.
* **Voice** — everything speech-related in one place: speech-to-text
  (backend, Whisper model, input language), text-to-speech (backend,
  output language, voices/speakers), and voice-activity detection
  (sensitivity, silence timeout, minimum speech).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from html import escape as _esc
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Button, Dialog, Label, RadioList


def _title_html(title: str) -> HTML:
    """Bold dialog title, escaped — titles are plain text and may contain & < >."""
    return HTML(f"<b>{_esc(title)}</b>")


# ── Shared dialog style ─────────────────────────────────────────


def dialog_style() -> Style:
    """A prompt_toolkit dialog style derived from the **active leuk theme**.

    Backgrounds stay at the terminal default so dialogs blend with the REPL;
    accents (frame label, selection, buttons) use the chosen theme's palette.
    Read live so it reflects the current theme.
    """
    import leuk.cli.theme as _theme

    p = _theme.PALETTE
    return Style.from_dict(
        {
            "dialog": "bg:default",
            "dialog.body": f"{p['fg']} bg:default",
            "dialog frame.label": f"bold {p['yellow']}",
            "frame.border": p["grey"],
            "dialog shadow": f"bg:{p['grey']}",
            "button": p["fg"],
            "button.focused": f"bold bg:{p['blue']} #ffffff",
            "radio-list": f"{p['fg']} bg:default",
            "radio": p["grey"],
            "radio-selected": f"bold {p['yellow']}",
            "radio-checked": f"bold {p['green']}",
            "label": f"{p['fg']} bg:default",
            "text-area": f"{p['fg']} bg:default",
            "text-area.cursor": p["yellow"],
        }
    )

# ── Option lists ────────────────────────────────────────────────

# Option labels are intentionally plain (no "-- description" comments):
# the BACKENDS list is shared between STT and TTS, so a shared comment
# would be wrong for one of them. Keep values self-explanatory instead.
BACKENDS: list[tuple[str, str]] = [
    ("local", "local"),
    ("openai", "openai"),
]

OPENAI_TTS_VOICES: list[tuple[str, str]] = [
    ("alloy", "alloy"),
    ("echo", "echo"),
    ("fable", "fable"),
    ("nova", "nova"),
    ("onyx", "onyx"),
    ("shimmer", "shimmer"),
]

STT_MODELS: list[tuple[str, str]] = [
    ("tiny", "tiny"),
    ("base", "base"),
    ("small", "small"),
    ("medium", "medium"),
    ("turbo", "turbo"),
    ("large-v3", "large-v3"),
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
    values: Sequence[tuple[str | None, str]],
    default: str | None,
) -> str | None:
    """Show a radiolist dialog and return the selected value (or None).

    Builds the dialog manually (instead of ``radiolist_dialog``) so we can
    add an Esc keybinding — prompt_toolkit's built-in helper omits it.
    """
    radio_list: RadioList[str | None] = RadioList(values=values, default=default)

    def _confirm_highlighted() -> None:
        """Exit with the value currently highlighted by the arrow cursor.

        Avoids the two-step ``RadioList`` flow (Enter to mark, then Tab to
        the Ok button, then Enter again). One Enter on the list confirms.
        """
        idx = getattr(radio_list, "_selected_index", None)
        if idx is not None and 0 <= idx < len(radio_list.values):
            app.exit(result=radio_list.values[idx][0])
        else:
            app.exit(result=radio_list.current_value)

    dialog = Dialog(
        title=_title_html(title),
        body=HSplit([Label(text=HTML(text), dont_extend_height=True), radio_list], padding=1),
        buttons=[
            Button(text="Ok", handler=_confirm_highlighted),
            Button(text="Cancel", handler=lambda: app.exit(result=None)),
        ],
        with_background=True,
    )

    bindings = KeyBindings()

    # Enter on the radio list confirms the highlighted option immediately.
    # ``eager`` beats RadioList's own Enter handler; the focus filter keeps
    # Enter on the Ok/Cancel buttons working normally.
    from prompt_toolkit.filters import Condition

    @bindings.add(
        "enter",
        filter=Condition(lambda: app.layout.has_focus(radio_list)),
        eager=True,
    )
    def _enter(event: object) -> None:
        _confirm_highlighted()

    # 'q' cancels while the list is focused (options are chosen with arrows/Enter,
    # so a literal 'q' is never needed as input here).
    @bindings.add("q", filter=Condition(lambda: app.layout.has_focus(radio_list)), eager=True)
    def _q(event: object) -> None:
        app.exit(result=None)

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
        style=dialog_style(),
        full_screen=True,
    )
    return app.run()


def _input(title: str, text: str, default: str = "") -> str | None:
    """A themed single-line input dialog. Enter confirms, Esc cancels. None = cancel."""
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.widgets import TextArea

    area = TextArea(text=default, multiline=False)
    bindings = KeyBindings()

    def _ok() -> None:
        app.exit(result=area.text)

    dialog = Dialog(
        title=_title_html(title),
        body=HSplit([Label(text=HTML(text), dont_extend_height=True), area], padding=1),
        buttons=[
            Button(text="Ok", handler=_ok),
            Button(text="Cancel", handler=lambda: app.exit(result=None)),
        ],
        with_background=True,
    )

    @bindings.add("enter", filter=Condition(lambda: app.layout.has_focus(area)), eager=True)
    def _enter(event: object) -> None:
        _ok()

    @bindings.add("escape")
    def _esc(event: object) -> None:
        app.exit(result=None)

    app: Application[str | None] = Application(
        layout=Layout(dialog, focused_element=area),
        key_bindings=merge_key_bindings([load_key_bindings(), bindings]),
        mouse_support=True,
        style=dialog_style(),
        full_screen=True,
    )
    return app.run()


def _busy(title: str, text: str, fn: "Callable[[], Any]") -> Any:
    """Run blocking *fn* while showing an animated spinner. Returns fn()'s result.

    *fn* runs in a thread-pool executor so the dialog can animate; its result (or
    exception) is propagated. Used to keep the UI responsive during a network
    search/resolve.
    """
    import asyncio
    import itertools

    frames = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    state = {"frame": "⠋"}
    result: dict[str, Any] = {}

    def _get_text() -> HTML:
        state["frame"] = next(frames)
        return HTML(f"{text}  <b>{state['frame']}</b>")

    dialog = Dialog(
        title=_title_html(title),
        body=Label(text=_get_text, dont_extend_height=True),
        with_background=True,
    )
    app: Application[None] = Application(
        layout=Layout(dialog),
        mouse_support=True,
        style=dialog_style(),
        full_screen=True,
        refresh_interval=0.08,
    )

    async def _main() -> None:
        async def _work() -> None:
            loop = asyncio.get_event_loop()
            try:
                result["value"] = await loop.run_in_executor(None, fn)
            except Exception as exc:  # noqa: BLE001 — re-raised below
                result["error"] = exc
            app.exit()

        app.create_background_task(_work())
        await app.run_async()

    asyncio.run(_main())
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _message(title: str, text: str) -> None:
    """A themed message dialog with a single Ok (Enter / Esc / q closes)."""
    bindings = KeyBindings()
    dialog = Dialog(
        title=_title_html(title),
        body=Label(text=HTML(text), dont_extend_height=True),
        buttons=[Button(text="Ok", handler=lambda: app.exit(result=None))],
        with_background=True,
    )

    @bindings.add("enter")
    @bindings.add("escape")
    @bindings.add("q")
    def _close(event: object) -> None:
        app.exit(result=None)

    app: Application[None] = Application(
        layout=Layout(dialog),
        key_bindings=merge_key_bindings([load_key_bindings(), bindings]),
        mouse_support=True,
        style=dialog_style(),
        full_screen=True,
    )
    app.run()


def _effective(key: str, updates: dict, current: dict, fallback: str = "") -> str:
    """Return the effective value for a config key (updates > current > fallback)."""
    v = updates.get(key)
    if v is not None:
        return str(v)
    v = current.get(key)
    if v is not None:
        return str(v)
    return fallback


def _effective_bool(key: str, updates: dict, current: dict, fallback: bool = False) -> bool:
    """Effective boolean value (updates > current > fallback)."""
    if key in updates:
        return bool(updates[key])
    if key in current:
        return bool(current[key])
    return fallback


def _yesno(title: str, text: str, current_value: bool) -> bool | None:
    """Show an Enabled/Disabled radio; return the chosen bool or None (cancel)."""
    res = _radio(
        title,
        text,
        [("true", "  Enabled"), ("false", "  Disabled")],
        "true" if current_value else "false",
    )
    if res is None:
        return None
    return res == "true"


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


# ── Tab: General ───────────────────────────────────────────────


def _theme_display(key: str) -> str:
    """Display label for a theme key."""
    from leuk.cli.theme import THEMES

    entry = THEMES.get(key)
    return entry["label"] if entry else key


def _run_general_tab(current: dict, updates: dict) -> None:
    """General settings: colour theme (+ read-only provider/model info)."""
    from leuk.cli.theme import DEFAULT_THEME, theme_choices

    while True:
        cur_theme = _effective("theme", updates, current, DEFAULT_THEME)
        provider = current.get("last_provider", "(not set)")
        model = current.get("last_model", "(not set)")
        br_on = _effective_bool("browser_enabled", updates, current, False)
        ic_on = _effective_bool("input_control_enabled", updates, current, False)
        ic_auto = _effective_bool("input_control_auto_approve", updates, current, False)
        sk_on = _effective_bool("skills_enabled", updates, current, False)
        mon_on = _effective_bool("monitoring_enabled", updates, current, False)
        media_mode = _effective("media_render", updates, current, "metadata")

        choice = _radio(
            "General",
            f"Provider: <b>{provider}</b>  ·  Model: <b>{model}</b>\n"
            "(use /auth and /model to change those)",
            [
                ("theme", f"  Theme              {_theme_display(cur_theme)}"),
                ("browser", f"  Browser tool       {'on' if br_on else 'off'}"),
                ("monitoring", f"  Monitoring (read-only)  {'on' if mon_on else 'off'}"),
                ("input_control", f"  Desktop control    {'on' if ic_on else 'off'}"),
                (
                    "input_auto",
                    f"  Desktop auto-approve  {'ON (danger)' if ic_auto else 'off'}",
                ),
                ("skills", f"  Agent skills       {'on' if sk_on else 'off'}"),
                ("media", f"  Media in history   {media_mode}"),
                ("back", "  << Back"),
            ],
            None,
        )
        if choice is None or choice == "back":
            return

        if choice == "browser":
            res = _yesno(
                "Browser Tool",
                "Let the agent drive a Chromium browser (navigate, click, read "
                "pages, fill forms) for web tasks. Requires <b>playwright</b> "
                "(<i>uv pip install leuk[browser]</i> and <i>playwright install "
                "chromium</i>). Takes effect on restart.",
                br_on,
            )
            if res is not None:
                updates["browser_enabled"] = res
        elif choice == "theme":
            result = _radio(
                "Colour Theme",
                "Choose a colour theme for the REPL.",
                [(key, "  " + label) for key, label in theme_choices()],
                cur_theme,
            )
            if result is not None:
                updates["theme"] = result
        elif choice == "monitoring":
            res = _yesno(
                "Monitoring (read-only)",
                "Let the agent gather host data without controlling it: take "
                "screenshots, read the screen resolution, and report system info "
                "(OS, CPU, memory, disk, uptime). No keyboard/mouse control. "
                "Takes effect on restart.",
                mon_on,
            )
            if res is not None:
                updates["monitoring_enabled"] = res
        elif choice == "input_control":
            res = _yesno(
                "Desktop Control",
                "Allow the agent to control the real keyboard and mouse "
                "(requires <b>ydotool</b> + /dev/uinput permissions). High risk.",
                ic_on,
            )
            if res is not None:
                updates["input_control_enabled"] = res
        elif choice == "input_auto":
            res = _yesno(
                "Desktop Auto-Approve",
                "<b>DANGER:</b> auto-approve desktop-control actions without "
                "prompting. The agent self-verifies and escalates risky actions, "
                "but it can move the mouse, type, and trigger irreversible actions.",
                ic_auto,
            )
            if res is not None:
                updates["input_control_auto_approve"] = res
        elif choice == "skills":
            res = _yesno(
                "Agent Skills",
                "Expose installed SKILL.md skills to the model (instructions-only; "
                "each skill is inert until you trust it). Install/manage with "
                "<b>/skills</b>. Takes effect on restart.",
                sk_on,
            )
            if res is not None:
                updates["skills_enabled"] = res
        elif choice == "media":
            result = _radio(
                "Media in history",
                "How images/audio/video render in the history browser.",
                [
                    ("metadata", "  metadata — compact info line (safe, headless)"),
                    ("inline", "  inline — ANSI thumbnail; Enter opens/plays"),
                ],
                media_mode,
            )
            if result is not None:
                updates["media_render"] = result


def _get_speaker_options(lang: str) -> list[tuple[str, str]]:
    """Return speaker options for a language, with a (default) entry."""
    speakers = SILERO_SPEAKERS.get(lang, [])
    if not speakers:
        return [("", "  (default for this language)")]
    return [(v, "  " + lbl) for v, lbl in speakers]


# ── Tab: Voice (STT + TTS + VAD combined) ──────────────────────


def _run_voice_tab(current: dict, updates: dict) -> None:
    """Unified Voice settings: speech-to-text, text-to-speech, and voice
    activity detection, all in a single menu."""
    while True:
        stt_backend = _effective("stt_backend", updates, current, "local")
        stt_model = _effective("stt_model_size", updates, current, "turbo")
        stt_lang = _effective("stt_language", updates, current, "")
        tts_backend = _effective("tts_backend", updates, current, "local")
        tts_lang = _effective("tts_language", updates, current, "en")
        speaker = _effective("tts_speaker", updates, current, "")
        en_speaker = _effective("tts_en_speaker", updates, current, "en_0")
        voice = _effective("tts_voice", updates, current, "alloy")
        sens = float(_effective("vad_sensitivity", updates, current, "0.5"))
        timeout = float(_effective("vad_silence_timeout", updates, current, "1.0"))
        minspeech = float(_effective("vad_min_speech", updates, current, "0.5"))

        items: list[tuple[str, str]] = [
            ("_stt", "  ── Speech-to-Text ──"),
            ("stt_backend", f"  Input Backend       {stt_backend}"),
        ]
        if stt_backend == "local":
            items.append(("stt_model_size", f"  Whisper Model       {stt_model}"))
        items += [
            ("stt_language", f"  Input Language      {_lang_display(stt_lang)}"),
            ("_tts", "  ── Text-to-Speech ──"),
            ("tts_backend", f"  Output Backend      {tts_backend}"),
        ]
        if tts_backend == "openai":
            items.append(("tts_voice", f"  OpenAI Voice        {voice}"))
        else:
            items.append(("tts_language", f"  Output Language     {_lang_display(tts_lang)}"))
            items.append(
                ("tts_speaker", f"  Voice ({tts_lang})         {_speaker_display(speaker, tts_lang)}")
            )
            if tts_lang != "en":
                items.append(
                    ("tts_en_speaker", f"  Voice (en)          {_speaker_display(en_speaker, 'en')}")
                )
        items += [
            ("_vad", "  ── Voice Activity ──"),
            ("vad_sensitivity", f"  Sensitivity         {sens:.1f} {_slider_bar(sens, 0.1, 0.9)}"),
            ("vad_silence_timeout", f"  Silence Timeout     {timeout:.1f}s {_slider_bar(timeout, 0.5, 2.0)}"),
            ("vad_min_speech", f"  Min Speech          {minspeech:.1f}s {_slider_bar(minspeech, 0.3, 1.0)}"),
            ("back", "  << Back"),
        ]

        choice = _radio("Voice", "Speech-to-text, text-to-speech and voice activity.", items, None)
        if choice is None or choice == "back":
            return
        if choice.startswith("_"):  # section header — not selectable
            continue

        if choice == "stt_backend":
            res = _radio(
                "Input Backend",
                "<b>local</b> uses Whisper on your GPU/CPU (no internet needed).\n"
                "<b>openai</b> uses the OpenAI Whisper API (requires API key).",
                [(v, "  " + lbl) for v, lbl in BACKENDS],
                stt_backend,
            )
            if res is not None:
                updates["stt_backend"] = res
        elif choice == "stt_model_size":
            res = _radio(
                "Whisper Model",
                "Larger models are more accurate but use more VRAM.\n"
                "<b>turbo</b> is recommended for GPU, <b>base</b> for CPU.",
                [(v, "  " + lbl) for v, lbl in STT_MODELS],
                stt_model,
            )
            if res is not None:
                updates["stt_model_size"] = res
        elif choice == "stt_language":
            res = _radio(
                "Input Language",
                "Setting a language improves recognition accuracy and avoids\n"
                "Whisper hallucinating other languages on background noise.\n"
                "Leave as <b>(auto-detect)</b> only for true multilingual use.",
                [(v, "  " + lbl) for v, lbl in LANGUAGES],
                stt_lang,
            )
            if res is not None:
                updates["stt_language"] = res or None
        elif choice == "tts_backend":
            res = _radio(
                "Output Backend",
                "<b>local</b> uses Silero TTS offline (fast, no internet).\n"
                "<b>openai</b> uses the OpenAI TTS API (higher quality, API key).",
                [(v, "  " + lbl) for v, lbl in BACKENDS],
                tts_backend,
            )
            if res is not None:
                updates["tts_backend"] = res
        elif choice == "tts_voice":
            res = _radio(
                "OpenAI TTS Voice",
                "Select a voice for OpenAI text-to-speech.",
                [(v, "  " + lbl) for v, lbl in OPENAI_TTS_VOICES],
                voice,
            )
            if res is not None:
                updates["tts_voice"] = res
        elif choice == "tts_language":
            tts_languages = [
                (v, "  " + lbl)
                for v, lbl in LANGUAGES
                if v in ("en", "es", "fr", "de", "ru", "it", "pt", "") or v in SILERO_SPEAKERS
            ]
            res = _radio(
                "Output Language",
                "Language for text-to-speech output.\n"
                "Only languages with Silero TTS models are shown.",
                tts_languages,
                tts_lang,
            )
            if res is not None:
                updates["tts_language"] = res or "en"
                if res != tts_lang:
                    updates.pop("tts_speaker", None)
        elif choice == "tts_speaker":
            res = _radio(
                f"Voice ({tts_lang})",
                "Select a voice for your language.",
                _get_speaker_options(tts_lang),
                speaker,
            )
            if res is not None:
                updates["tts_speaker"] = res
        elif choice == "tts_en_speaker":
            res = _radio(
                "Voice (English)",
                "Select a voice for English text segments.",
                _get_speaker_options("en"),
                en_speaker,
            )
            if res is not None:
                updates["tts_en_speaker"] = res
        elif choice == "vad_sensitivity":
            res = _radio(
                "VAD Sensitivity",
                "How sensitive the detector is to speech. <b>0.5</b> suits most.\n"
                "Increase if your speech is missed; decrease if noise triggers it.",
                [(v, f"  {_slider_bar(float(v), 0.1, 0.9)}  {lbl}") for v, lbl in VAD_SENSITIVITY_OPTIONS],
                str(sens),
            )
            if res is not None:
                updates["vad_sensitivity"] = res
        elif choice == "vad_silence_timeout":
            res = _radio(
                "Silence Timeout",
                "How long to wait after speech stops before ending the segment.\n"
                "Shorter = faster response. Longer = captures full sentences.",
                [(v, f"  {_slider_bar(float(v), 0.5, 2.0)}  {lbl}") for v, lbl in VAD_SILENCE_TIMEOUT_OPTIONS],
                str(timeout),
            )
            if res is not None:
                updates["vad_silence_timeout"] = res
        elif choice == "vad_min_speech":
            res = _radio(
                "Minimum Speech Duration",
                "Segments shorter than this are discarded as noise.\n"
                "Lower = picks up short words; higher = ignores brief sounds.",
                [(v, f"  {_slider_bar(float(v), 0.3, 1.0)}  {lbl}") for v, lbl in VAD_MIN_SPEECH_OPTIONS],
                str(minspeech),
            )
            if res is not None:
                updates["vad_min_speech"] = res


# ── Main entry point ────────────────────────────────────────────


def run_settings(current: dict[str, Any]) -> dict[str, Any] | None:
    """Show the unified settings dialog.  Blocking.

    *current* is the full config dict from ``load_persistent_config()``.
    Returns a dict of updated keys, or ``None`` if no changes were made.
    """
    updates: dict[str, Any] = {}

    while True:
        # Build main menu items with current-value summaries
        from leuk.cli.theme import DEFAULT_THEME

        cur_theme = _theme_display(_effective("theme", updates, current, DEFAULT_THEME))
        cur_stt_backend = _effective("stt_backend", updates, current, "local")
        cur_tts_backend = _effective("tts_backend", updates, current, "local")
        cur_tts_lang = _lang_display(_effective("tts_language", updates, current, "en"))
        cur_tts_voice = _effective("tts_voice", updates, current, "alloy")

        voice_summary = f"in: {cur_stt_backend}, out: {cur_tts_backend}"
        if cur_tts_backend == "openai":
            voice_summary += f"/{cur_tts_voice}"
        else:
            voice_summary += f"/{cur_tts_lang}"

        tab = _radio(
            "Settings",
            "Select a category to configure.\n"
            "Press <b>Esc</b> to save and exit.",
            [
                (
                    "general",
                    f"  General            theme: {cur_theme}",
                ),
                (
                    "voice",
                    f"  Voice              {voice_summary}",
                ),
            ],
            None,
        )

        if tab is None:
            # Esc at main menu → save and exit
            return updates if updates else None

        if tab == "general":
            _run_general_tab(current, updates)
        elif tab == "voice":
            _run_voice_tab(current, updates)
