[Home](README.md) › Safety & Approvals

# Safety & Approvals

Every tool call passes through `src/leuk/safety.py:SafetyGuard` before executing.

## Review policies

`/policy <mode>` sets how aggressively tools require approval:

| Mode | Behavior |
|------|----------|
| `auto` | Never ask — all tools proceed |
| `agent` | Ask only on dangerous shell ops (`rm`, `sudo`, …) |
| `cautious` *(default)* | Ask on all writes (shell, file_edit); reads auto-allowed |
| `strict` | Also ask on reads (file_read, web_fetch) |
| `paranoid` | Ask for every tool call |

Always-on **deny rules** protect secrets/system paths (`.env`, `*.pem`, `~/.ssh`,
`/etc/**`) regardless of policy.

## Rules

`ToolRule(tool, pattern, action)` with **deny > ask > allow** priority; patterns
are globs matched against the tool's primary argument. User rules
(`SafetyConfig.rules`) are evaluated before policy rules.

## Approval prompt

When a tool needs approval the TUI shows an in-app overlay (the classic prompt
shows the equivalent themed dialog):

- **Enter** allow once · **Esc** deny once.
- **`a`** always allow · **`d`** always deny — scoped to a **meaningful pattern**,
  not the whole tool or the verbatim argument: `shell` → the program
  (`pkg-config *`), `file_edit`/`file_read` → the directory (`src/game/*`),
  `web_fetch`/`browser` → the host (`*docs.python.org*`), `input_control` → the
  action (`click*`). "Always allow" thus grants a *semantic* permission, and
  these saved rules take precedence over a policy "ask"
  (`SafetyGuard._match_saved_approval`).
- **Tab** — *amend*: edit the command/path inline and approve the edited version
  (the agent runs your version; carried as `ApprovalResult.amended_args`).
- **Ctrl-E** — toggle a risk level (low/medium/high, from the shared
  dangerous-command patterns) plus a one-line explanation.

**Always** choices persist to SQLite (`/approvals` to list, `/approvals clear`
to reset). Pattern derivation lives in `cli/approval.py:approval_scope`.

## File edits are patches, not rewrites

`file_edit` changes existing files with **targeted patches** (exact-string
replace) — it refuses to rewrite a whole existing file. Replacing a file's entire
contents requires an explicit `overwrite=true`, which is treated as destructive
and **prompts for approval** under every policy except `auto`.

## Read-only mode

`/readonly` blocks all write tools (shell, file_edit, sub_agent) outright.

## Desktop control

The [Input Control](tools/input_control.md) tool is **always-ask** by default
(even under `auto`/`agent`). The dedicated `/desktop-auto` switch (or
`input_control.auto_approve`) opts into auto-approval with a prominent warning; in
that mode the agent self-verifies actions and should escalate risky ones. On
[channels](channels.md) the approval request carries before/after screenshots.

## Channel approvals

On Telegram/Slack/Discord, approval requests appear as inline buttons routed to
the originating channel (`request_approval`); denial stops the agent.

## Steering & approvals

[Steering](steering.md) may inject `[STEERING]` guidance messages and add recovery
hints to errored tool results, but it runs **after** a turn and never bypasses the
SafetyGuard: every tool call it leads to is still gated, and a denied
(`[BLOCKED]`) call suppresses the steering nudge so it can't push against your
decision.

## See also

- [Tools](tools.md) · [Channels](channels.md) · [Steering](steering.md) · [Configuration](configuration.md)
