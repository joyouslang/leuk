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

When a tool needs approval the REPL asks `y / n / Y / N`
(allow / deny / always-allow / always-deny; Enter = deny). **Always** choices are
persisted to SQLite (`/approvals` to list, `/approvals clear` to reset).

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

## See also

- [Tools](tools.md) · [Channels](channels.md) · [Configuration](configuration.md)
