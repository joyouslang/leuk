[Home](README.md) › Skills

# Agent Skills (SKILL.md)

leuk can use **Agent Skills** — reusable, self-contained playbooks in the open
**SKILL.md** format (used by OpenClaw/ClawHub, Anthropic, and others). A skill is
a folder bundle:

```
my-skill/
├── SKILL.md          # YAML frontmatter (name, description) + markdown instructions
└── scripts/…         # optional helper scripts/assets
```

The model sees only each skill's **name + description** (cheap), and pulls the
full instructions on demand via the `skill` tool — *progressive disclosure*. leuk
is **instructions-only**: it never auto-runs a skill's scripts; the model carries
out the instructions through the normal, [SafetyGuard](safety.md)-gated shell/file
tools.

Enable skills with the **Agent skills** toggle in `/settings` (or
`{"skills": {"enabled": true}}` / `LEUK_SKILLS_ENABLED=true`).

## Trust gate

A skill is **inert until you trust it**. On import (and in the manager) leuk shows
the bundle's files and asks *"Do you trust this skill?"* — untrusted skills are
hidden from the model. This is the first of two safety layers; the second is the
per-call SafetyGuard on whatever shell/file actions the model takes. Only trust
skills from sources you trust.

## Managing skills — `/skills` · `src/leuk/skills/`

Run **`/skills`** for a menu UI (add, trust/untrust, enable/disable, view,
remove), or the CLI:

```bash
leuk skills search <query>                          # search ClawHub (needs the `clawhub` CLI)
leuk skills add <slug> --source clawhub [--trust]   # install a ClawHub skill
leuk skills add <url>  --source git                 # clone a SKILL.md repo
leuk skills add <path> --source local               # copy a local bundle
leuk skills list                                    # installed skills + trust/on-off
leuk skills trust <name> | untrust <name>
leuk skills enable <name> | disable <name>
leuk skills remove <name>
```

The `/skills` and `/mcp` menus (and `/settings`) render in your chosen colour
theme and select on **Enter** (no Tab needed).

Skills live in `~/.config/leuk/skills/` (global) and `./.leuk/skills/` (per
project). Trust and enable/disable state is stored in `config.json`. Changes apply
on the next session/restart.

## How the model uses a skill

- `src/leuk/skills/tool.py` (`SkillTool`) lists usable skills in its description
  and returns a skill's full `SKILL.md` (plus a manifest of bundle files) for
  `action="read"`.
- `src/leuk/skills/loader.py` (`SkillLoader`) discovers bundles, parses
  frontmatter, and enforces trust/enable filtering.

## See also

- [Tools](tools.md) · [Safety](safety.md) · [MCP](mcp.md) · [Configuration](configuration.md)
