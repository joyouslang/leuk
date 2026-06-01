[Home](README.md) › Multimodal

# Multimodal (images, audio & video)

leuk sends images, audio, and video **natively** to multimodal models, so the
model can analyse pictures, screenshots, audio, and clips — not just text.

## Pieces

- `MediaPart` (`src/leuk/types.py`) — `kind` (`image`/`audio`/`video`),
  `media_type`, base64 `data`. `Message.attachments` holds them.
- `src/leuk/media.py` — `extract_media()` pulls inline tags
  (`[screenshot:…]`, `[image:…]`, `[audio:…]`, `[video:…]`) out of tool results
  into `MediaPart`s; `media_to_tag()` is the inverse; `load_media_file()` loads
  an image/audio/video file (type auto-detected from extension/MIME).

## Attaching your own files

`/file <path>` stages an image, audio, or video file for your **next** message
(e.g. `/file ~/diagram.png` then ask about it). Threaded via
`Agent.pending_attachments` onto the user `Message`.

## Tool screenshots are seen by the model

The [Browser](tools/browser.md) and [Input Control](tools/input_control.md) tools
emit `[screenshot:…]` tags (also verification screenshots after failures). These
become real image blocks for the active provider — the model sees them.

**Media survives context compaction.** A screenshot's base64 is large, so the
naive truncation/masking that bounds tool-result *text* would otherwise chop the
tag mid-base64 (breaking it) or count it as ~100k text tokens and mask it away —
either way the model would receive a garbled string and *hallucinate* seeing
something. Context management therefore treats inline media specially
(`src/leuk/agent/context.py`): only the surrounding **text** is truncated/masked,
media tags are kept intact, and each media part is budgeted as a flat
native-block cost rather than its base64 length.

## Per-provider handling ([Providers](providers.md))

| Provider | Images | Tool screenshots | Audio | Video |
|----------|--------|------------------|-------|-------|
| Anthropic | user content blocks | inside `tool_result` content | not supported → text note | not supported |
| OpenAI | `image_url` data-URIs | follow-up user message | `input_audio` (audio models) | not supported |
| Google | `inline_data` Blobs | follow-up user content | `inline_data` (native) | `inline_data` (Gemini) |

Video is forwarded only to providers that accept it (Gemini via `inline_data`);
Anthropic/OpenAI silently drop video parts (they're filtered to images).

## Models without vision

Media is **always** sent through the model's native image/video channel — never
as base64 text for the model to "reason" over. If the active model has **no
vision support**, leuk strips the media before the call and leaves a short note
(`[N media item(s) omitted — the active model '<model>' has no vision support…]`)
so the model gets a clear, graceful message instead of a base64 blob (which it
would hallucinate over) or an API error.

Vision capability is **queried from the provider's API** (`provider.model_info()`,
see [Providers](providers.md)) — never guessed from the model name. Media is
stripped only when the query reports vision is *definitely* absent (e.g. an
OpenRouter model whose `input_modalities` lack `image`, or an Ollama model whose
`/api/show` `capabilities` lack `vision`). When support is **unknown** (the API
doesn't report it, as with Anthropic/plain OpenAI), the media is sent natively and
the API decides. The stripping happens in `Agent._prepare_context`.

## Persistence

Attachments round-trip through SQLite (stored in message metadata under
`_attachments`), so multi-turn image analysis survives session reloads.

## See also

- [Providers](providers.md) · [Tools](tools.md) · [Voice](voice.md) · [Sessions & Persistence](sessions-and-persistence.md)
