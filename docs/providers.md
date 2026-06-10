[Home](README.md) › Providers

# Providers

leuk talks to LLMs through the `LLMProvider` protocol
(`src/leuk/providers/base.py`): `generate()`, `stream()`, `close()`, and
`model_info()` (queries the model's metadata). Providers are built by
`create_provider()` (`src/leuk/providers/catalog.py`) from `LLMConfig`.

| Provider | Module | Notes |
|----------|--------|-------|
| OpenCode Zen (default) | `zen.py` | OpenAI-compatible gateway, curated free models (`big-pickle`) |
| Anthropic | `anthropic.py` | API key **or** OAuth bearer; auto token refresh on 401 |
| OpenAI | `openai.py` | also the base for `local`, `zen`, `openrouter` |
| Google Gemini | `google.py` | `google-genai` SDK; native image + audio |
| OpenRouter | `openrouter.py` | OpenAI provider with OpenRouter base URL |
| Local (Ollama/vLLM) | via `openai.py` | OpenAI-compatible local endpoint |

## Authentication

- **Anthropic OAuth** — `/auth → Anthropic → OAuth login` runs a PKCE flow
  (`src/leuk/cli/auth.py`); the token + refresh token go in `credentials.json`
  and are refreshed automatically.
- **API keys** — set per provider via `/auth` or
  [env vars](reference/environment.md).

## Multimodal

All three first-party providers send images natively, and tool screenshots are
delivered to the model (not just as text). Audio goes to providers that accept it
(Gemini, OpenAI audio models). See [Multimodal](multimodal.md).

## Thinking / reasoning stream

Extended reasoning is supported **by default — no setting**. Providers request
it and stream it as `THINKING_DELTA` events, viewable live in the
[TUI](repl-commands.md) with **Ctrl-T** and stored on the assistant
`Message.thinking`:

- **Anthropic** — `thinking: {type: enabled, budget_tokens: …}` with a budget
  derived from `max_tokens` (half, capped at 8192, min 1024). Skipped when the
  request carries a temperature (the API only allows `temperature == 1` with
  thinking) or when `max_tokens` is too small to fit thinking plus an answer.
  Thinking blocks (with signatures) are replayed on tool-use continuations, as
  the API requires.
- **Google Gemini** — `thinking_config.include_thoughts = true`; thought-summary
  parts stream as reasoning.
- **OpenAI-compatible** (OpenRouter/Zen/local) — the request carries
  `reasoning: {}` (OpenRouter and compatible gateways only include reasoning
  when asked), and DeepSeek-style `reasoning_content`/`reasoning` deltas are
  surfaced whenever the backend sends them.

Capability discovery is **live, not guessed**: if the active model rejects the
thinking/reasoning parameter, the API's own error triggers one retry without it
and the provider remembers the rejection for the rest of the session — no
model-name lists.

**No reasoning showing?** Run `/status` — its `Thinking:` line says whether the
parameter is being requested, was rejected by the endpoint, or is off because
`llm.temperature` is set (Anthropic allows thinking only at the default
temperature) or `llm.max_tokens` is too small. Two cases stream nothing by
design:

- Some endpoints **withhold the reasoning text** and return only a signed
  placeholder (observed on the Claude-subscription OAuth path with newer
  models): the model *does* think — that's the pause before the answer — but
  there is no text to display. `/status` detects and reports this. The signed
  blocks are still replayed on tool-use continuations as the API requires.
- A gateway that simply doesn't expose reasoning shows "requested" but streams
  none.

## Queried model metadata (`model_info`)

Capabilities are **queried from each provider's API**, never guessed from the
model name (which goes stale and is wrong for fast-moving local stacks). Each
provider's `model_info()` returns a [`ModelInfo`](../src/leuk/providers/model_info.py)
— `context_window`, `supports_vision`, `supports_audio` — reading whatever the
API exposes; anything it doesn't report stays `None` ("unknown"):

| Provider | Source | Reports |
|----------|--------|---------|
| OpenAI-compatible (OpenRouter/Zen/vLLM) | `GET /v1/models` | `context_length`/`max_model_len`; OpenRouter also `architecture.input_modalities` |
| Local (Ollama) | native `POST /api/show` | `capabilities` (e.g. `vision`) + `*.context_length` |
| Google Gemini | `models.get` | `input_token_limit` |
| Anthropic / plain OpenAI | — (API exposes neither) | all unknown |

**Context window** (for the usage gauge) is resolved by `resolve_context_window()`
(`src/leuk/providers/context_window.py`): **live query → `LLMConfig.context_window`
override → unknown**. When unknown the gauge shows the raw token estimate (no
made-up %); set `LEUK_LLM_CONTEXT_WINDOW` (or `config.json`) for providers that
don't report it (Anthropic, plain OpenAI).

**Vision**: images/video go through the model's native channel. Media is stripped
(with a note) **only** when a provider's query reports vision is *definitely*
absent; when unknown it's sent natively and the API decides (its error surfaces
to you) — no name-based guessing.

## Authorizing providers — `/auth`

`/auth` (`src/leuk/cli/auth.py`) is the single place to manage provider
credentials, **separate from `/model`**. It lists the providers; picking one by
number opens that provider's authorization (add/replace/delete its key, and switch
to it) — there are no separate add/edit/delete commands. For Anthropic, the
configure step offers the Claude subscription OAuth login or an API key.

## Switching models

`/model` (`src/leuk/cli/models.py`) lists models for **authorized** providers
(those with credentials, plus local), fetched at runtime. If the active provider
isn't authorized it redirects to `/auth` rather than showing an empty dialog. The
dialog is themed and dismissable with **Esc** or **q**. Selecting a model from
another provider switches providers and re-resolves the context window. You can
also set `LEUK_LLM_MODEL` directly.

The list is always **live** — no curated/hardcoded models. Under a **Claude
subscription (OAuth)**, if `/v1/models` returns `401` because the access token
expired, the catalog refreshes it (`refresh_anthropic_token()`) and retries once —
the same refresh-on-401 flow as generation (`AnthropicProvider._try_refresh_token`).
If the refresh itself fails (e.g. no stored refresh token), re-run `/auth`.

## See also

- [Configuration](configuration.md) · [Multimodal](multimodal.md) · [Context Management](context-management.md)
