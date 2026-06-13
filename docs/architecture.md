[Home](README.md) › Architecture Overview

# Architecture Overview

```
        ┌──────────────┐   keyboard / voice / channels
        │   CLI / REPL │◀──────────────────────────────┐
        └──────┬───────┘                                │
               │ push(text, attachments)                │
        ┌──────▼───────┐     ┌──────────────┐           │
        │ AgentSession │────▶│    Agent     │           │
        │ (asyncio task)│     │  run_stream  │           │
        └──────────────┘     └──┬───┬───┬───┘           │
                                │   │   │               events
              ┌─────────────────┘   │   └───────────┐   │
        ┌─────▼─────┐  ┌────────────▼───┐  ┌─────────▼──┴───┐
        │ Provider  │  │  Tool Registry │  │ Context Manager│
        │  (LLM)    │  │  + SafetyGuard │  │  (compaction)  │
        └───────────┘  └───────┬────────┘  └────────────────┘
                               │
   shell · file_read/edit · web_fetch · sub_agent · memory_write
   · browser · input_control · local_llm · MCP-bridged tools
                               │
                      ┌────────┴────────┐
                   SQLite            Hot store
                  (durable)         (fast resume)
```

## The agent loop

`src/leuk/agent/core.py:Agent.run_stream`:

1. Append the user message (with any [attachments](multimodal.md)) to history.
2. **Prepare context** — tiered compaction keeps it under budget
   ([Context Management](context-management.md)); for weak/local models a short
   [steering](steering.md) reminder is re-injected periodically (ephemeral).
3. Call the **provider** (streaming) with the context and tool specs. The system
   prompt is augmented with [steering](steering.md) discipline for weak/local
   models; a tool call a weak model emitted as plain text (not via the API) is
   salvaged into a real call.
4. For each tool call: pass through the **[SafetyGuard](safety.md)** (deny/ask/allow),
   then execute. Append the `ToolResult` (errored results get a recovery hint under
   steering). Repeat 2–4 up to `max_tool_rounds`. Under [steering](steering.md), a
   lengthy run that keeps repeating the same calls is detected as **circling** and
   redirected, then force-consolidated — rather than spinning to the ceiling.
5. When the model stops with **no tool calls**, the [steering](steering.md)
   persistence guard decides accept-vs-continue: a bounded self-reflection check
   (or a truncation fast-path) may inject a nudge and loop again; otherwise the
   turn ends.
6. Persist every message to [SQLite](sessions-and-persistence.md) as it's produced.
7. On interrupt/error, partial output and orphaned tool calls are healed so the
   next turn isn't corrupted.

## AgentSession

`src/leuk/agent/session.py:AgentSession` drives the `Agent` in a background
`asyncio.Task`, exchanging messages over input/event queues. It survives per-turn
errors (surfacing an `ERROR` event + `/retry`), supports graceful interruption,
and lets the REPL detach/reattach.

## Streaming & rendering

`src/leuk/cli/render.py:StreamRenderer` consumes the event stream and renders:
assistant text as live **Markdown**, tool calls as bordered status blocks (with
syntax-highlighted diffs), a thinking spinner, and replayed history. See
[CLI & UI](cli-and-ui.md).

## Sub-agents & teams

- **Sub-agents** (`src/leuk/agent/sub_agent.py`) — the `sub_agent` tool spawns an
  independent `Agent` with its own `Session` (`parent_session_id` set), running as
  an `asyncio.Task`. On completion the session is **archived** (kept for
  inspection via `/subagents`) and cascade-deleted with its parent.
- **Teams** (`src/leuk/agent/team.py`) — named roles for multi-agent orchestration.

## Key building blocks

| Concern | Page |
|---------|------|
| LLM backends | [Providers](providers.md) |
| Tools | [Tools](tools.md) |
| Approvals | [Safety & Approvals](safety.md) |
| Sessions & storage | [Sessions & Persistence](sessions-and-persistence.md) |
| External tools | [MCP](mcp.md) |
| Background jobs | [Scheduler](scheduler.md) |
| Remote chat | [Channels](channels.md) |

## See also

- [Context Management](context-management.md) · [CLI & UI](cli-and-ui.md) · [Development](development.md)
