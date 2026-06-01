[Home](README.md) › Sessions & Persistence

# Sessions & Persistence

## Session lifecycle

- **Lazy creation** — startup and `/new` produce an unpersisted *draft*. A session
  is created and started only when you send the first message or pick one. The
  footer shows `new session` while pending.
- **Auto-naming** — the first message is turned into a short title via the model
  (validated against refusals/prose, with a heuristic fallback). Shown in the
  footer, `/sessions`, and `/switch`.
- **Resume** — `/switch <id>` clears the screen and replays the conversation
  (Markdown + tool blocks) via `render_history`.
- **Delete** — `/delete` of the current session opens a modal to choose what to
  continue with; a fresh session is only auto-created when none remain.
- **Sub-agents** — child sessions (`parent_session_id` set) are **archived** when
  they finish (inspect via `/subagents <id>`) and **cascade-deleted** with their
  parent. They are hidden from `/sessions` (`top_level_only`).

## SQLite (durable) — `src/leuk/persistence/sqlite.py`

Stored at `~/.config/leuk/leuk.db`:

- **sessions** — `id`, `status`, `created_at`, `updated_at`, `system_prompt`,
  `metadata` (JSON), `parent_session_id`.
- **messages** — `session_id`, `role`, `content`, `tool_calls` (JSON),
  `tool_result` (JSON), `timestamp`, `metadata` (JSON; also carries multimodal
  `_attachments`).
- **tool_approvals** — persisted [approval rules](safety.md).

`delete_session` cascades to sub-agent children. `list_sessions(top_level_only=…)`
and `list_child_sessions(parent_id=…)` drive `/sessions` and `/subagents`.

## Hot store — `src/leuk/persistence/memory.py`

An in-memory `HotStore` caches recent context and tracks the active session id for
fast resume. Sessions are always also written to SQLite.

## See also

- [Architecture Overview](architecture.md) · [Multimodal](multimodal.md) · [Safety](safety.md)
