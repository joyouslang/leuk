[Home](README.md) › Context Management

# Context Management

`src/leuk/agent/context.py:compact()` keeps the conversation under a **compaction
budget** that is **derived from the model's own [context window](providers.md)**
(queried from the provider, reserving room for the reply) — not a hardcoded
value. `max_context_tokens` is an optional *override*; when unset (the default)
the budget = `compaction_budget(window, reserve=max_output_tokens)`. It runs
before every provider call and is a tiered pipeline:

1. **Truncate** — tool results larger than `max_tool_result_tokens` are shortened
   in place.
2. **Mask observations** — past ~60% of budget, older tool-result *bodies* are
   replaced with one-line placeholders. Reasoning and tool-call arguments are
   never touched.
3. **Structured summarize** — still over budget: the oldest messages are archived
   to disk (`src/leuk/agent/archive.py`, when enabled) and summarized by the LLM
   into a persistent structured summary (goal, files, decisions, state, pending).
   Prior summaries merge incrementally.
4. **Emergency drop** — if summarization fails, the oldest non-system messages are
   dropped with a placeholder.

## Overflow recovery — no turn dies on "context exceeded"

If the server still rejects a request as too large (e.g. llama-server's
`request (N tokens) exceeds the available context size (M tokens)` — possible
when the window couldn't be queried, or because the char-based token estimator
undercounts vs. the server's real tokenizer), the agent does **not** fail the
turn: it parses the server-reported limit, clamps its effective window to it
(with a growing safety margin on repeated attempts), re-runs compaction, and
retries — up to 3 times per turn, with a status line in the transcript. The
learned clamp persists for the session so later turns pre-compact instead of
re-hitting the limit. Nothing is lost: compaction archives + summarizes, and
the [`history` tool](#nothing-is-ever-unreachable--the-history-tool) keeps the
originals reachable.

For llama.cpp's `llama-server`, the serving context (`-c`) is also queried up
front from its `/props` endpoint, so the budget is right from the first turn.

## Nothing is ever unreachable — the `history` tool

Compaction shrinks the *in-context view*, never the record: every message stays
in SQLite. The built-in **`history` tool** (`src/leuk/tools/history.py`) gives
the model read-only access to the **entire** stored conversation —
`action='search'` finds earlier messages by text (returning stable indices +
snippets) and `action='read'` re-reads originals around an index. The summary
and drop placeholders explicitly tell the model to use it, so after compaction
the agent has both the summary *and* on-demand access to the full history.

## Tool-pair safety

Compaction never splits a `tool_use` from its `tool_result`. `_safe_split_index`,
the group-aware emergency drop, and `_fix_orphaned_pairs` guarantee providers
never see an unpaired tool call. The agent also heals orphaned pairs in
`run_stream` after an interrupted turn.

## Usage gauge

The footer shows **`ctx ~<used>/<window> <pct>%`** and `/status` spells it out
(`~N est. prompt tokens of W window (P%)`). The figures:

- **used** — an *estimate* (the `~`) of the whole prompt leuk sends the model
  (system prompt + conversation + tool results), via `estimate_total_tokens()`
  (`chars / 4`, with inline media counted as a flat native-block cost, not its
  base64 length).
- **window** — the model's **maximum** context length, resolved by
  `resolve_context_window()`:
  1. a live query of the provider's API (`provider.model_info().context_window`
     — OpenRouter/Zen/vLLM `context_length`/`max_model_len`, Ollama `/api/show`,
     Gemini `input_token_limit`),
  2. the `LEUK_LLM_CONTEXT_WINDOW` override / `config.json`,
  3. otherwise **unknown** — the gauge shows the raw token estimate (no fake %).
     For providers whose API doesn't report it (Anthropic, plain OpenAI), set the
     override so the gauge has a denominator.

`/status` also shows the **compaction budget** (`Compact: trims at ~N tokens`) —
where leuk starts trimming, derived from the model window with headroom for the
reply. The gauge turns yellow at 70% and red at 90%.

## See also

- [Providers](providers.md) · [Architecture Overview](architecture.md)
