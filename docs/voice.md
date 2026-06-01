[Home](README.md) › Voice

# Voice

Voice needs the `[voice]` extra (`uv sync --extra voice`): PyTorch, transformers,
sounddevice, omegaconf. Availability is detected via `VOICE_AVAILABLE` in
`src/leuk/voice/__init__.py`.

## Input — STT + VAD

`/voice` toggles hands-free input. A neural **Silero VAD**
(`src/leuk/voice/recorder.py:ContinuousVAD`) watches the mic and detects speech;
on a pause, the clip is trimmed and transcribed.

| STT backend | Engine | Offline |
|-------------|--------|---------|
| `local` *(default)* | HuggingFace Whisper (`turbo`) | yes |
| `openai` | OpenAI Whisper API | no |

## Output — TTS

`/speak` reads responses aloud, sentence-by-sentence as text streams
(`StreamingTTSSpeaker`).

| TTS backend | Engine | Offline |
|-------------|--------|---------|
| `local` *(default)* | Silero TTS (multilingual, dual-model) | yes |
| `openai` | OpenAI TTS API | no |

Configure backends/voices/VAD via `/settings` (persisted to `config.json`).

### Spoken-form normalization (numbers & acronyms)

Silero's character vocabulary has **no digits**, so a raw number would be dropped
and come out silent. Before synthesis, `normalize_for_speech` (in
`src/leuk/voice/tts.py`) rewrites text into a fully pronounceable form:

- **Numbers → words** via [`num2words`](https://pypi.org/project/num2words/):
  `3` → "three" / "три" / "drei" …, including decimals (`3.14` → "three point one
  four") and thousands grouping (`1,000` → "one thousand"). Each number is spelled
  in the language **detected from the text around it** — the script of its
  neighbouring words — not the configured voice. So an English reply spoken by a
  Russian-configured voice still reads its numbers in English, and vice-versa; a
  bare number with no nearby words falls back to the configured language.
  Languages num2words doesn't implement fall back to a close relative
  (CIS → Russian) then English.
- **ALL-CAPS acronyms → spoken letter names**: `API` → "ay pee eye", `ФСБ` →
  "эф эс бэ". Each letter maps to a pronounceable name (Latin → English names,
  Cyrillic → Russian names — the two alphabets the dual-model handles), because a
  bare spaced "a p i" comes out as slurred phonemes. Mixed-case words (`iOS`) and
  lone capitals (the article `A`) are left alone.

This runs before the per-script model routing, so a number among Cyrillic words
is spelled in the user's language while a Latin acronym's letters still route to
the English model. The OpenAI backend already handles numbers/acronyms natively.

## Half-duplex: mic pauses while the agent speaks

Voice input is **half-duplex**. While the agent is speaking (TTS), the mic VAD is
**paused** and resumed once playback finishes (`_run_agent_turn` in
`src/leuk/cli/repl.py`). This avoids a speaker → mic → VAD feedback loop where the
agent would otherwise hear and transcribe its own voice. You speak, the agent
replies aloud, then you speak again — turn by turn.

Interrupting the agent *while it speaks* is done from the keyboard with **Ctrl-C**,
not by voice: a single Ctrl-C cancels the turn — interrupting generation **and**
stopping TTS playback at once — and returns to the prompt. Hands-free
voice barge-in was removed: reliably telling the user's voice apart from the
agent's own loud-speaker echo proved infeasible without dedicated hardware echo
cancellation, and the half-measures caused self-triggering, chopped playback, and
hangs. Use **headphones** if you want to talk while the agent is speaking without
it hearing itself.

Playback uses a thread-owned `sounddevice.OutputStream` and is interrupted by a
`threading.Event` checked between audio blocks — **never** a cross-thread
`sd.stop()`, which double-frees PortAudio and crashes the process.

Two STT content filters still run on every transcription so background noise
doesn't become a spurious prompt:

- **Hallucination filter** (`_is_hallucination`) — drops Whisper's stock
  non-speech hallucinations: degenerate repetition (a token/char repeated) *and*
  the canned phrases it emits on background noise ("Thank you.", "We'll see you
  in the next video.", "Thanks for watching!", and multilingual equivalents).

## See also

- [Multimodal](multimodal.md) · [Configuration](configuration.md) · [CLI & UI](cli-and-ui.md)
