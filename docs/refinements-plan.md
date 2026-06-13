# leuk — refinements backlog

The original audit (`refactor-plan.md`) and TUI design (`repl-tui-design.md`)
are **done** and have been removed. The full-screen persistent-input TUI,
event-style approval (in-app overlay with contextual scoped allow/deny,
Tab→amend, Ctrl-E risk/explanation), live thinking stream, `/undo` with git
snapshots, inline diffs, crash recovery, channel quality (auto-discovery, pipe
channel, Telegram HTML/typing/edit-in-place, Slack mrkdwn), context-overflow
recovery, the `history` tool, and dynamic theming all shipped.

This file tracks only what was **deliberately deferred** from those plans, plus
notes for future work. Keep it in sync: delete an item when it lands.

## Deferred items

### 1. `ReplController` / `ReplState` refactor (was plan §4.1, §4.3)
`_run_repl()` is still a ~2k-line function with nonlocal session/agent/provider
state and inline command handlers. Decompose into a `ReplController` class with a
command dispatch table and a `ReplState` dataclass. Pure cleanup — **no
user-facing change**, high churn, and the REPL was just rebuilt around the TUI,
so this is intentionally last. Do it as an isolated pass with the existing tests
as the safety net.

### 2. WhatsApp channel (was plan §4.6 / §6.6)
Needs a backend decision (Twilio WhatsApp API vs. Meta Cloud API vs. an
unofficial `whatsapp-web.js` bridge) and credentials. The channel registry
auto-discovers `channels/*.py`, so it's a drop-in `channels/whatsapp.py`
implementing the `Channel` protocol once a backend is chosen.

### 3. Discord rich embeds (was plan §6.5)
Discord renders standard Markdown natively, so replies already format. Native
*embeds* (cards/fields/colour bars) would be a polish item — add a `send_rich()`
capability on the `Channel` protocol and implement it for Discord (embeds) and
Slack (Block Kit, beyond the current mrkdwn text).

### 4. Mid-turn "interrupt and ask" (was plan §5.11)
Mid-turn *queueing* works (type while the agent streams; the message is answered
after the current turn). The richer variant — a key that **suspends** the
running turn and injects a correction immediately — is not implemented. Would
require pausing the agent loop mid-round and splicing a user message before the
next provider call.

### 5. SIGINT handler context manager (was plan §0.5 / §3.7)
The classic line-prompt fallback still installs/removes the SIGINT handler
manually (tiny window where a stray Ctrl-C could terminate). Near-moot now: the
default TUI handles Ctrl-C via a key binding, so this only affects the fallback
path. Wrap it in a context manager if revisiting that path.

## Notes / ideas (not yet planned)

- **Streaming token usage from provider metadata.** The footer/thinking counters
  estimate tokens (chars/4). When providers report real `usage`, surface exact
  prompt/completion counts.
- **Approval explanation via a fast LLM summariser.** The Ctrl-E panel currently
  shows a heuristic risk + reason. The original §5.10 idea was a lazy one-line
  `generate()` summary of *what the command does*; could be added behind a flag.
- **Multi-program shell scope.** "Always allow" for a compound command
  (`a && b`) currently scopes to the first program. Could persist one rule per
  distinct program.
