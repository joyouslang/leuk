"""Interactive manager for skills and MCP connectors (the `/skills` and `/mcp` UI).

Built on the same themed, Enter-to-select dialog helpers as ``/settings``
(`_radio`/`_input`/`_message`), so it matches the user's chosen colour theme and
navigates on Enter (no Tab-then-Enter). The dialogs are blocking, so the REPL
runs the managers via ``asyncio.to_thread``; network calls use ``asyncio.run``.

The same loader/registry functions back the non-interactive CLI subcommands, so
there is one logic path with two front-ends.
"""

from __future__ import annotations

from html import escape as esc
from pathlib import Path

from leuk.cli.settings_dialog import _busy, _input, _message, _radio
from leuk.config import Settings, load_persistent_config
from leuk.mcp import registry
from leuk.skills import (
    SkillImportError,
    SkillLoader,
    import_clawhub,
    import_git,
    import_local,
    remove_skill,
    search_clawhub,
    set_skill_enabled,
    set_skill_trusted,
)


def _confirm(title: str, text: str) -> bool:
    return _radio(title, text, [("yes", "  Yes"), ("no", "  No")], "no") == "yes"


def _skills_state() -> tuple[set[str], set[str]]:
    sec = load_persistent_config().get("skills", {})
    sec = sec if isinstance(sec, dict) else {}
    return set(sec.get("trusted", [])), set(sec.get("disabled", []))


def _bundle_files(path: Path, limit: int = 30) -> list[str]:
    return [
        str(p.relative_to(path))
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != "SKILL.md"
    ][:limit]


# ── Skills ─────────────────────────────────────────────────────────


def run_skills_manager(settings: Settings) -> None:
    """Top-level skills menu loop (blocking)."""
    while True:
        trusted, disabled = _skills_state()
        loader = SkillLoader(settings.skills.directory, project_dir=".leuk/skills",
                             trusted=trusted, disabled=disabled)
        skills = loader.all_skills()
        values: list[tuple[str | None, str]] = []
        for m in skills:
            badge = "✓ trusted" if m.trusted else "⚠ untrusted"
            state = "on" if m.enabled else "off"
            values.append((m.slug, f"  {m.name} — {m.description[:44]}  [{badge}, {state}]"))
        values.append(("\x00add", "  ➕  Add a skill…"))
        values.append((None, "  ←  Done"))
        sel = _radio("Skills", f"Installed agent skills ({len(skills)}).", values, None)
        if sel is None:
            return
        if sel == "\x00add":
            _add_skill_flow(settings)
        else:
            _skill_actions(settings, sel)


def _skill_actions(settings: Settings, slug: str) -> None:
    trusted, disabled = _skills_state()
    loader = SkillLoader(settings.skills.directory, project_dir=".leuk/skills",
                         trusted=trusted, disabled=disabled)
    meta = loader.find(slug)
    if meta is None:
        return
    sel = _radio(
        esc(meta.name),
        esc(meta.description or "(no description)"),
        [
            ("trust", "  Untrust" if meta.trusted else "  Trust — allow the model to use it"),
            ("toggle", "  Disable" if meta.enabled else "  Enable"),
            ("view", "  View SKILL.md"),
            ("remove", "  Remove — delete the bundle"),
            (None, "  ← Back"),
        ],
        None,
    )
    if sel == "trust":
        if meta.trusted:
            set_skill_trusted(slug, False)
        else:
            listing = "\n".join(f"  • {f}" for f in _bundle_files(meta.path)) or "  (no extra files)"
            if _confirm(
                "Trust this skill?",
                esc(f"{meta.name}\n\nBundle files the agent could run:\n{listing}\n\n"
                    "Only trust skills from sources you trust. Trust it?"),
            ):
                set_skill_trusted(slug, True)
    elif sel == "toggle":
        set_skill_enabled(slug, not meta.enabled)
    elif sel == "view":
        try:
            body = (meta.path / "SKILL.md").read_text("utf-8")
        except OSError as exc:
            body = f"(could not read: {exc})"
        _message(esc(meta.name), esc(body[:4000]))
    elif sel == "remove":
        if _confirm("Remove skill?", esc(f"Delete the bundle for {meta.name!r}?")):
            remove_skill(slug, settings.skills.directory)


def _add_skill_flow(settings: Settings) -> None:
    source = _radio(
        "Add a skill",
        "Where from?",
        [
            ("clawhub", "  ClawHub — search & install"),
            ("git", "  Git URL — clone a SKILL.md repo"),
            ("local", "  Local folder — copy a SKILL.md bundle"),
            (None, "  ← Cancel"),
        ],
        None,
    )
    if source is None:
        return

    slug: str | None = None
    if source == "clawhub":
        query = _input("ClawHub", "Search skills:")
        if not query:
            return
        try:
            results = _busy("Searching", "Searching ClawHub…", lambda: search_clawhub(query.strip()))
        except SkillImportError as exc:
            _message("ClawHub", esc(str(exc)))
            return
        if not results:
            _message("No results", "No skills matched.")
            return
        pick = _radio(
            "ClawHub results", "Pick a skill:",
            [(s, f"  {n}  ({s})") for s, n in results] + [(None, "  ← Cancel")], None,
        )
        if pick is None:
            return
        try:
            slug = import_clawhub(pick, settings.skills.directory)
        except SkillImportError as exc:
            _message("Install failed", esc(str(exc)))
            return
    else:
        prompt = "Git repository URL:" if source == "git" else "Path to a local SKILL.md folder:"
        ident = _input("Add a skill", prompt)
        if not ident:
            return
        importer = import_git if source == "git" else import_local
        try:
            slug = importer(ident.strip(), settings.skills.directory)
        except SkillImportError as exc:
            _message("Import failed", esc(str(exc)))
            return

    bundle = Path(settings.skills.directory).expanduser() / slug
    listing = "\n".join(f"  • {f}" for f in _bundle_files(bundle)) or "  (no extra files)"
    trusted = _confirm(
        "Trust this skill?",
        esc(f"Installed {slug!r}.\n\nBundle files the agent could run:\n{listing}\n\n"
            "Only trust skills from sources you trust. Trust it now?"),
    )
    if trusted:
        set_skill_trusted(slug, True)
    _message(
        "Skill installed",
        esc(f"{slug} installed" + (" and trusted." if trusted else " (untrusted — trust it later to use).")),
    )


# ── MCP connectors ─────────────────────────────────────────────────


def run_mcp_manager(settings: Settings, status: dict[str, str] | None = None) -> None:
    """Top-level MCP-connector menu loop (blocking). *status* maps name → live state."""
    status = status or {}
    while True:
        servers = registry.list_connectors()
        values: list[tuple[str | None, str]] = []
        for s in servers:
            state = status.get(s.name, "—") if s.enabled else "off"
            values.append((s.name, f"  {s.name}  [{s.transport}, {state}]"))
        values.append(("\x00add", "  ➕  Add a connector…"))
        values.append((None, "  ←  Done"))
        sel = _radio(
            "MCP connectors / plugins",
            f"Saved MCP servers ({len(servers)}). Status is this session's live state.",
            values, None,
        )
        if sel is None:
            return
        if sel == "\x00add":
            _add_connector_flow(settings)
        else:
            _connector_actions(sel)


def _prompt_inputs(resolved: registry.ResolvedConnector) -> bool:
    """Prompt (themed dialogs) for each required arg/env the server needs. False = cancel."""
    if not resolved.inputs:
        return True
    values: dict[str, str] = {}
    for spec in resolved.inputs:
        label = spec.flag or spec.id
        prompt = label
        if spec.fmt and spec.fmt not in ("string", ""):
            prompt += f"  ({spec.fmt})"
        if spec.description:
            prompt += f"\n{spec.description}"
        if spec.secret:
            prompt += "\n(stored in config.json)"
        val = _input("Configure connector", esc(prompt), default=spec.default)
        if val is None:
            return False
        values[spec.id] = val.strip()
    registry.apply_inputs(resolved, values)
    return True


def _connector_actions(name: str) -> None:
    import shlex

    server = next((s for s in registry.list_connectors() if s.name == name), None)
    if server is None:
        return
    detail = server.url or f"{server.command} {' '.join(server.args)}".strip()
    edit_label = "  Edit URL" if server.transport == "sse" else "  Edit command & arguments"
    options: list[tuple[str | None, str]] = [
        ("toggle", "  Disable" if server.enabled else "  Enable"),
        ("edit", edit_label),
    ]
    if server.transport != "sse":
        options.append(("edit_env", "  Edit environment variables"))
    options += [("remove", "  Remove"), (None, "  ← Back")]

    sel = _radio(esc(name), esc(f"{server.transport}: {detail}"), options, None)
    if sel == "toggle":
        registry.set_connector_enabled(name, not server.enabled)
    elif sel == "remove":
        if _confirm("Remove connector?", esc(f"Delete {name!r}?")):
            registry.remove_connector(name)
    elif sel == "edit":
        if server.transport == "sse":
            new = _input("Edit URL", esc(f"Remote MCP server URL for {name}:"), default=server.url)
            if new and new.strip():
                registry.update_connector(name, url=new.strip())
        else:
            current = " ".join(shlex.quote(t) for t in [server.command, *server.args])
            new = _input(
                "Edit command & arguments",
                esc(f"Full command line for {name} (shell-quoted, e.g. add "
                    "--allowed-directories /your/path):"),
                default=current,
            )
            if new and new.strip():
                toks = shlex.split(new)
                registry.update_connector(name, command=toks[0], args=toks[1:])
    elif sel == "edit_env":
        current = " ".join(f"{k}={v}" for k, v in server.env.items())
        new = _input(
            "Edit environment",
            esc(f"Env vars for {name} as KEY=value pairs (space-separated):"),
            default=current,
        )
        if new is not None:
            env = dict(
                tok.split("=", 1) for tok in shlex.split(new) if "=" in tok
            )
            registry.update_connector(name, env=env)


def _add_connector_flow(settings: Settings) -> None:
    source = _radio(
        "Add a connector",
        "Where from?",
        [
            ("mcp", "  Official MCP registry — search"),
            ("url", "  Paste a remote MCP server URL"),
            (None, "  ← Cancel"),
        ],
        None,
    )
    if source is None:
        return

    description = ""
    try:
        if source == "url":
            url = _input("Add a connector", "Remote MCP server URL:")
            if not url:
                return
            resolved = registry.resolve(url.strip(), "url")
        else:
            query = _input("Search connectors", "Search the MCP registry:")
            if query is None:
                return
            hits = _busy(
                "Searching", "Searching the MCP registry…",
                lambda: registry.search(query.strip(), "mcp", registry_url=settings.mcp_registry.url),
            )
            if not hits:
                _message("No results", "No connectors matched.")
                return
            pick = _radio(
                "Search results", "Pick a connector:",
                [(h.id, f"  {h.name} — {h.description}") for h in hits] + [(None, "  ← Cancel")],
                None,
            )
            if pick is None:
                return
            hit = next((h for h in hits if h.id == pick), None)
            description = hit.description if hit else ""
            resolved = hit.resolved if (hit and hit.resolved) else _busy(
                "Resolving", "Fetching connector details…",
                lambda: registry.resolve(pick, "mcp", registry_url=settings.mcp_registry.url),
            )
    except Exception as exc:  # noqa: BLE001 — surface any resolve/network error
        _message("Could not resolve", esc(str(exc)))
        return

    if not _prompt_inputs(resolved):
        return
    cfg = resolved.config
    desc_note = f"\n\n{description}" if description else ""
    detail = cfg.url or f"{cfg.command} {' '.join(cfg.args)}".strip()
    if _confirm(
        "Add this connector?",
        esc(f"{resolved.summary}\nSaved as: {cfg.name}\n{detail}{desc_note}"),
    ):
        registry.add_connector(resolved)
        _message("Connector added", esc(f"{cfg.name} saved — it connects on the next start."))


# ── non-interactive CLI (`leuk skills …` / `leuk mcp …`) ───────────


def _prompt_inputs_cli(resolved: registry.ResolvedConnector) -> bool:
    """Prompt at the terminal for a connector's required args/env. False = abort."""
    import getpass
    import sys

    if not sys.stdin.isatty():
        needed = ", ".join(s.flag or s.id for s in resolved.inputs)
        print(f"error: this connector needs values ({needed}); run interactively to provide them")
        return False
    values: dict[str, str] = {}
    for spec in resolved.inputs:
        label = spec.flag or spec.id
        suffix = f" [{spec.default}]" if spec.default else ""
        desc = f" — {spec.description}" if spec.description else ""
        prompt = f"{label}{desc}{suffix}: "
        val = (getpass.getpass(prompt) if spec.secret else input(prompt)).strip()
        values[spec.id] = val or spec.default
    registry.apply_inputs(resolved, values)
    return True


def run_skills_cli(argv: list[str]) -> int:
    """Handle ``leuk skills <action> …``. Returns a process exit code."""
    import argparse

    from leuk.config import load_settings, migrate_legacy_config_env

    p = argparse.ArgumentParser(prog="leuk skills")
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    add = sub.add_parser("add")
    add.add_argument("ident", help="slug / git URL / local path")
    add.add_argument("--source", choices=["clawhub", "git", "local"], default="clawhub")
    add.add_argument("--trust", action="store_true", help="mark trusted immediately")
    sea = sub.add_parser("search")
    sea.add_argument("query")
    for act in ("enable", "disable", "trust", "untrust", "remove"):
        sub.add_parser(act).add_argument("name")
    args = p.parse_args(argv)

    migrate_legacy_config_env()
    settings = load_settings()
    directory = settings.skills.directory
    trusted, disabled = _skills_state()
    loader = SkillLoader(directory, project_dir=".leuk/skills", trusted=trusted, disabled=disabled)

    if args.action == "list":
        skills = loader.all_skills()
        if not skills:
            print("No skills installed.")
            return 0
        for m in skills:
            badge = "trusted" if m.trusted else "UNTRUSTED"
            state = "on" if m.enabled else "off"
            print(f"{m.slug:<24} [{badge}, {state}] {m.name} — {m.description}")
        return 0
    if args.action == "search":
        try:
            for slug, nm in search_clawhub(args.query):
                print(f"{slug:<24} {nm}")
        except SkillImportError as exc:
            print(f"error: {exc}")
            return 1
        return 0
    if args.action == "add":
        importer = {"clawhub": import_clawhub, "git": import_git, "local": import_local}[args.source]
        try:
            slug = importer(args.ident, directory)
        except SkillImportError as exc:
            print(f"error: {exc}")
            return 1
        if args.trust:
            set_skill_trusted(slug, True)
            print(f"installed and trusted {slug}")
        else:
            print(f"installed {slug} (untrusted — run `leuk skills trust {slug}` to use it)")
        return 0

    meta = loader.find(args.name)
    if meta is None:
        print(f"no such skill: {args.name}")
        return 1
    {
        "enable": lambda: set_skill_enabled(meta.slug, True),
        "disable": lambda: set_skill_enabled(meta.slug, False),
        "trust": lambda: set_skill_trusted(meta.slug, True),
        "untrust": lambda: set_skill_trusted(meta.slug, False),
        "remove": lambda: remove_skill(meta.slug, directory),
    }[args.action]()
    print(f"{args.action}: {meta.slug}")
    return 0


def run_mcp_cli(argv: list[str]) -> int:
    """Handle ``leuk mcp <action> …``. Returns a process exit code."""
    import argparse

    from leuk.config import load_settings, migrate_legacy_config_env

    p = argparse.ArgumentParser(prog="leuk mcp")
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    sea = sub.add_parser("search")
    sea.add_argument("query")
    add = sub.add_parser("add")
    add.add_argument("ident", help="registry id / URL")
    add.add_argument("--source", choices=["mcp", "url"], default="mcp")
    add.add_argument("--name")
    for act in ("enable", "disable", "remove"):
        sub.add_parser(act).add_argument("name")
    args = p.parse_args(argv)

    migrate_legacy_config_env()
    settings = load_settings()
    registry_url = settings.mcp_registry.url

    if args.action == "list":
        servers = registry.list_connectors()
        if not servers:
            print("No connectors saved.")
            return 0
        for s in servers:
            detail = s.url or f"{s.command} {' '.join(s.args)}".strip()
            print(f"{s.name:<24} [{'on' if s.enabled else 'off'}, {s.transport}] {detail}")
        return 0
    if args.action == "search":
        try:
            hits = registry.search(args.query, "mcp", registry_url=registry_url)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}")
            return 1
        for h in hits:
            print(f"{h.id} — {h.description}")
        return 0
    if args.action == "add":
        try:
            resolved = registry.resolve(
                args.ident, args.source, name=args.name, registry_url=registry_url
            )
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}")
            return 1
        if resolved.inputs and not _prompt_inputs_cli(resolved):
            return 1
        registry.add_connector(resolved)
        print(f"added {resolved.config.name} ({resolved.summary}); connects on next `leuk` start")
        return 0

    if args.action in ("enable", "disable"):
        ok = registry.set_connector_enabled(args.name, args.action == "enable")
    else:
        ok = registry.remove_connector(args.name)
    print(f"{args.action}: {args.name}" if ok else f"no such connector: {args.name}")
    return 0 if ok else 1
