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
