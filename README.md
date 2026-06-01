<div align="center">

# leuk

**A persistent AI agent that lives in your terminal.**

Multi-provider LLMs · sub-agent orchestration · voice in/out · desktop & browser
control · native multimodal (images, audio, video) · shell, files, web & MCP.

[Getting Started](docs/getting-started.md) ·
[Documentation](docs/README.md) ·
[Configuration](docs/configuration.md) ·
[Architecture](docs/architecture.md)

</div>

---

## Quickstart

```bash
uv sync          # install
uv run leuk      # start the REPL
```

The default provider is **OpenCode Zen** (`big-pickle`) — free, no API key, works
out of the box. Want a different one? Run **`/auth`** inside the REPL to set up
Anthropic, OpenAI, Google Gemini, OpenRouter, or a local Ollama/vLLM endpoint.

Everything else is configured from **`/settings`** (written to a single
`~/.config/leuk/config.json`) — no shell environment variables required.

```text
❯ leuk
  leuk · zen/big-pickle                       ctx ~1.2k/200k 1%
❯ summarize today's git log and open a PR with the notes
  …
```

## Why leuk

- **It remembers.** Sessions are SQLite-backed, lazily created, auto-named, and
  resumable — pick up exactly where you left off.
- **It sees and hears.** Screenshots, images, audio, and video are sent to the
  model **natively** (never base64-pasted as text); voice gives you hands-free
  speech in and out.
- **It does things.** Run shell commands, read and patch files, fetch the web,
  drive a real browser, and control the desktop keyboard & mouse.
- **It stays safe.** Rule-based review policies, per-tool permissions, persistent
  approvals, and channel-native approval prompts gate every risky action.
- **It scales out.** Spawn sub-agents for parallel work, schedule background
  tasks, and reach it from Telegram, Slack, or Discord.

## Highlights

| | |
|---|---|
| **Providers** | Anthropic (incl. OAuth), OpenAI, Google Gemini, OpenRouter, OpenCode Zen, local Ollama/vLLM — context window & capabilities are **queried**, never hardcoded |
| **Multimodal** | Images, audio, and video sent through native provider blocks; non-vision models degrade gracefully |
| **Tools** | Shell, file read/[patch-edit](docs/tools.md), web fetch, sub-agents, memory, plus optional [browser](docs/tools/browser.md) and [desktop control](docs/tools/input_control.md) |
| **Voice** | Local or remote STT & TTS with neural VAD; numbers and acronyms are spoken correctly per language |
| **Context** | Tiered compaction (truncate → mask → summarize → drop) sized to each model's real context window |
| **Reach** | [MCP](docs/mcp.md) servers, a background [scheduler](docs/scheduler.md), and [channels](docs/channels.md) (Telegram/Slack/Discord) |

## Documentation

The **[wiki](docs/README.md)** is the authoritative, ground-truth documentation,
kept in sync with the code:

- [Getting Started](docs/getting-started.md) — install, extras, first run, `/auth`
- [REPL & Commands](docs/repl-commands.md) · [Configuration](docs/configuration.md) · [Environment Variables](docs/reference/environment.md)
- [Architecture](docs/architecture.md) · [Providers](docs/providers.md) · [Context Management](docs/context-management.md)
- [Tools](docs/tools.md) · [Safety](docs/safety.md) · [Voice](docs/voice.md) · [Multimodal](docs/multimodal.md)
- [Channels](docs/channels.md) · [Scheduler](docs/scheduler.md) · [Sessions & Persistence](docs/sessions-and-persistence.md) · [MCP](docs/mcp.md)
- [Development](docs/development.md)

## Acknowledgments

leuk's offline voice stack builds on:

- **[Silero VAD](https://github.com/snakers4/silero-vad)** (MIT) — voice-activity detection
- **[Silero Models](https://github.com/snakers4/silero-models)** — neural text-to-speech (see citation & licensing below)
- **[OpenAI Whisper](https://github.com/openai/whisper)** (MIT) — speech-to-text

### Citing Silero

```bibtex
@misc{Silero Models,
  author = {Silero Team},
  title = {Silero Models: pre-trained text-to-speech models made embarrassingly simple},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snakers4/silero-models}},
  commit = {insert_some_commit_here},
  email = {hello@silero.ai}
}
```

### Silero model licenses & commercial use

leuk is **AGPL-3.0**. The Silero TTS models are downloaded at runtime (not
bundled in this repo), so they do not alter leuk's license — but their **own**
licenses constrain how you may use the synthesized output:

| Component | License | Commercial use |
|-----------|---------|----------------|
| Silero **VAD** | MIT | ✅ yes |
| Silero **TTS — CIS** (`v5_cis_base`: ru, uk, kk, tt, uz, ky, ba, xal) | MIT | ✅ yes |
| Silero **TTS — other** (`v3_en`, `v3_de`, `v3_es`, `v3_fr`, `v4_indic`) | CC BY-NC | ❌ non-commercial only |

So Russian and CIS-language TTS is free for commercial use; **English, German,
Spanish, French, and Indic TTS via Silero is non-commercial only**. For
commercial use of those languages, switch to the **OpenAI TTS** backend (or a
permissively-licensed engine such as [Piper](https://github.com/OHF-Voice/piper1-gpl)).
See [Voice](docs/voice.md) for details.

## License

[AGPL-3.0-or-later](LICENSE)
