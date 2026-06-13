[Home](README.md) › Steering

# Steering

Steering makes **weaker / local models behave more like frontier models**. Strong
models (Sonnet/Opus-class) finish long, tool-heavy tasks on their own. Small local
models (Ollama/vLLM) often have all the context and tools they need yet **give
up**: they stop mid-task, refuse outright, or abandon the task after a single tool
error. Steering is the extra discipline leuk supplies so they keep going.

Implemented in `src/leuk/agent/steering.py` (pure helpers) and wired into
`src/leuk/agent/core.py:Agent`.

## When it's active

Gating is by config **and the provider signal** — never by guessing the model's
identity from its name. `steering.enabled`:

| Value | Effect |
|-------|--------|
| `auto` *(default)* | Active **only** when `llm.provider == "local"` |
| `on` | Active for every provider |
| `off` | Never active |

When inactive, steering is a true no-op: the system prompt is unchanged and the
loop behaves exactly as before (so frontier models are never affected).

Toggle live with [`/steering [auto\|on\|off]`](repl-commands.md) or in `/settings →
General → Model steering`. The choice persists to `config.json`
(`steering.enabled`).

## What it does

Five cooperating pieces, all behind the single switch above:

1. **System-prompt steering** — appends a short *operating discipline* block
   (persist until done; never refuse a task you have tools for; on an error
   re-observe state and try a *different* approach; verify before moving on; say
   so explicitly when done). It augments your prompt — your custom
   `system_prompt` stays first. Add your own lines with
   `steering.extra_instructions`. When the [desktop-control tool](tools/input_control.md)
   is enabled, it also tells the model to click by **percentage** (`xpct`/`ypct`)
   rather than pixels read off the scaled-down screenshot — weak models otherwise
   click the raw screenshot pixel, which misses.
2. **Mid-loop reminders** — small models lose the system prompt over long
   contexts, so a short reminder is re-injected every `reminder_interval` tool
   rounds (and the round after a tool error). It is **ephemeral**: added only to
   the context sent to the model, never persisted or compacted.
3. **Self-reflection persistence guard** — the core. When the model stops with
   **no tool calls**, leuk asks the model itself whether the task is actually
   complete (a cheap, separate check). If it says *continue*, leuk injects a
   nudge (carrying the model's own next step) and loops again. Bounded by
   `max_continuations` per turn, so it can never loop forever; after the cap it
   accepts the stop.
4. **Tool-error recovery** — errored tool results get a short recovery hint
   appended, so the model adapts instead of abandoning the task.
5. **Truncation fast-path** — if a reply is cut off (`finish_reason == "length"`),
   leuk injects `continue` immediately, **without** spending a reflection call.
6. **Circle-breaker** — the opposite failure from giving up: a model that *spins*,
   re-issuing the same tool calls without progress. In a **lengthy** session
   (`loop_min_rounds`+), leuk detects a tight loop — the last few tool-call
   signatures identical, or an ABAB 2-cycle — and redirects it ("stop repeating;
   use what you've gathered or take a *different* step"). If it keeps circling
   past `loop_max_interventions`, leuk forces a **tools-off consolidation reply**
   (gather what's been calculated, then conclude) instead of wasting rounds up to
   `max_tool_rounds`. A model that *varies* its calls after a nudge is not
   penalised.
7. **Text tool-call salvage** — weak models sometimes write a tool call as plain
   *text* instead of using the native function-calling API, e.g.
   `<tool_call><function=browser><parameter=action>navigate<parameter=url>…`. leuk
   would otherwise see a reply with no tool call and the round ends with nothing
   run. Salvage parses these back into real, executable calls (pseudo-XML,
   Hermes-style JSON, and bare `{"name","arguments"}` are all handled), validated
   against the *registered* tool names so ordinary prose is never misread.

## Bounds (no infinite loops)

Steering both *rescues* premature stops and *breaks* spins, with hard bounds either
way:

- `steering.max_continuations` — inner cap on reflection "continue" nudges per turn.
- `steering.loop_max_interventions` — redirect nudges before a circling loop is
  force-consolidated.
- `agent.max_tool_rounds` — outer ceiling the loop re-enters on every continue.
- the existing max-rounds forced-final reply.

The reflection check is also skipped for casual chat by default
(`reflect_only_after_tool_use = true`) — it runs only once a tool has been used in
the turn, so a plain Q&A reply is never second-guessed. First-turn refusals are
handled by the system-prompt steering instead. Circle detection only arms after
`loop_min_rounds` tool rounds, so short, legitimate repeats are never flagged.

## Configuration

All fields live under the `steering` section (`SteeringConfig`,
`env_prefix = LEUK_STEERING_`). See [Environment Variables](reference/environment.md#steering--leuk_steering_)
for the env-var form.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `auto` | `auto` (local only) / `on` / `off` |
| `extra_instructions` | `""` | Your steering text, appended after the built-in block |
| `reminder_interval` | `8` | Reminder every N tool rounds (0 = never); also after an error |
| `max_continuations` | `3` | Max self-reflection nudges per turn |
| `reflection_max_tokens` | `256` | Token budget for each reflection check |
| `reflect_only_after_tool_use` | `true` | Only reflect when a tool ran this turn |
| `nudge_on_truncation` | `true` | Auto-`continue` on `finish_reason=length` |
| `enrich_tool_errors` | `true` | Append a recovery hint to errored tool results |
| `loop_detection` | `true` | Detect & break a spinning (repeated/cyclic) tool loop |
| `loop_min_rounds` | `4` | Only check for circling after this many tool rounds |
| `loop_max_interventions` | `2` | Redirect nudges before forcing a consolidation reply |
| `salvage_text_tool_calls` | `true` | Recover tool calls a model emitted as plain text |

Example `config.json`:

```json
{ "steering": { "enabled": "on", "reminder_interval": 6,
                "extra_instructions": "Prefer ripgrep over grep." } }
```

## Cost note

The self-reflection guard spends one extra (small) model call each time the model
stops while a task is in progress — bounded by `max_continuations`. Truncation
uses the free fast-path, and casual chat is skipped, so the overhead targets
exactly the agentic turns where giving up matters. The circle-breaker is the other
side of the ledger: detection is a near-free string comparison, and breaking a
spin at round ~4–6 instead of `max_tool_rounds` (50) **saves** many wasted LLM
calls and tool executions.

## See also

- [Providers](providers.md) · [Configuration](configuration.md) · [Architecture](architecture.md) · [Safety & Approvals](safety.md)
