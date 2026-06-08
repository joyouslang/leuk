"""SkillLoader: discover, parse, and import SKILL.md skill bundles.

A *skill* is a folder bundle containing a ``SKILL.md`` file (YAML frontmatter
with at least ``name`` and ``description``, then markdown instructions) and
optionally scripts/assets the agent may run via the existing shell/file tools.
This follows the open SKILL.md standard used by OpenClaw/ClawHub, Anthropic, and
others.

Skills are **inert until trusted**: a bundle is only surfaced to the model once
its slug is in the ``trusted`` set (the user reviews it on import). ``disabled``
slugs are kept on disk but hidden. leuk never executes a skill's scripts itself —
the model does, through the SafetyGuard-gated shell tool.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from leuk.config import load_persistent_config, save_persistent_config

_FRONTMATTER = "---"
_MAX_MANIFEST_FILES = 40


@dataclass(slots=True)
class SkillMeta:
    """One discovered skill bundle."""

    slug: str  # the bundle directory name (stable id)
    name: str
    description: str
    path: Path  # absolute bundle directory
    trusted: bool
    enabled: bool
    scope: str = "global"  # "global" or "project"


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a SKILL.md into (frontmatter fields, body).

    Minimal hand parser for the simple ``key: value`` frontmatter we need
    (``name``/``description``) — avoids a YAML dependency. Unknown/nested keys
    are ignored here; the model reads the full body verbatim anyway.
    """
    if not text.startswith(_FRONTMATTER):
        return {}, text
    lines = text.splitlines()
    # lines[0] == "---"; find the closing fence.
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == _FRONTMATTER:
            end = i
            break
    if end is None:
        return {}, text
    fields: dict[str, str] = {}
    for raw in lines[1:end]:
        if ":" not in raw or raw.lstrip().startswith("#"):
            continue
        key, _, val = raw.partition(":")
        fields[key.strip().lower()] = val.strip().strip("'").strip('"')
    body = "\n".join(lines[end + 1 :]).strip()
    return fields, body


class SkillLoader:
    """Discovers SKILL.md bundles and exposes them, honouring trust/enable state."""

    def __init__(
        self,
        skills_dir: str = "~/.config/leuk/skills",
        *,
        project_dir: str | None = None,
        trusted: set[str] | None = None,
        disabled: set[str] | None = None,
        max_index: int = 50,
    ) -> None:
        self.skills_dir = Path(skills_dir).expanduser()
        self.project_dir = Path(project_dir).expanduser() if project_dir else None
        self.trusted = trusted or set()
        self.disabled = disabled or set()
        self.max_index = max_index

    # ── discovery ──────────────────────────────────────────────────
    def _bundle_dirs(self) -> list[tuple[Path, str]]:
        """(bundle_dir, scope) for every dir holding a SKILL.md, global then project."""
        out: list[tuple[Path, str]] = []
        for base, scope in ((self.skills_dir, "global"), (self.project_dir, "project")):
            if base is None or not base.is_dir():
                continue
            for child in sorted(base.iterdir()):
                if child.is_dir() and (child / "SKILL.md").is_file():
                    out.append((child, scope))
        return out

    def all_skills(self) -> list[SkillMeta]:
        """Every installed skill (trusted or not, enabled or not) — for the manager."""
        metas: list[SkillMeta] = []
        for bundle, scope in self._bundle_dirs():
            slug = bundle.name
            try:
                fields, _body = _parse_frontmatter((bundle / "SKILL.md").read_text("utf-8"))
            except OSError:
                continue
            metas.append(
                SkillMeta(
                    slug=slug,
                    name=fields.get("name") or slug,
                    description=fields.get("description", ""),
                    path=bundle,
                    trusted=slug in self.trusted,
                    enabled=slug not in self.disabled,
                    scope=scope,
                )
            )
        return metas

    def usable(self) -> list[SkillMeta]:
        """Trusted **and** enabled skills, capped at ``max_index`` — for the tool."""
        usable = [m for m in self.all_skills() if m.trusted and m.enabled]
        return usable[: self.max_index]

    def find(self, name_or_slug: str) -> SkillMeta | None:
        key = name_or_slug.strip().lower()
        for m in self.all_skills():
            if key in (m.slug.lower(), m.name.lower()):
                return m
        return None

    def read(self, name_or_slug: str) -> str | None:
        """Full SKILL.md body + a manifest of the bundle's other files.

        Only trusted+enabled skills can be read. Returns ``None`` if not found
        or not usable.
        """
        meta = self.find(name_or_slug)
        if meta is None or not meta.trusted or not meta.enabled:
            return None
        try:
            _fields, body = _parse_frontmatter((meta.path / "SKILL.md").read_text("utf-8"))
        except OSError:
            return None
        files = [
            str(p.relative_to(meta.path))
            for p in sorted(meta.path.rglob("*"))
            if p.is_file() and p.name != "SKILL.md"
        ][:_MAX_MANIFEST_FILES]
        manifest = ""
        if files:
            listing = "\n".join(f"- {f}" for f in files)
            manifest = (
                f"\n\n---\nBundle directory: {meta.path}\n"
                f"Other files in this skill (run/read them with the shell/file tools "
                f"as the instructions direct):\n{listing}"
            )
        return f"# {meta.name}\n\n{body}{manifest}"


# ── importers (place a bundle on disk; trust is decided by the caller) ──


class SkillImportError(RuntimeError):
    """Raised when a skill import cannot complete."""


def _dest_dir(skills_dir: str) -> Path:
    d = Path(skills_dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, turning a timeout/OS error into a SkillImportError so it
    never propagates and crashes the REPL."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SkillImportError(
            f"{cmd[0]} timed out after {timeout}s — it may be offline or unresponsive"
        ) from None
    except OSError as exc:
        raise SkillImportError(f"could not run {cmd[0]}: {exc}") from exc


def import_local(path: str, skills_dir: str = "~/.config/leuk/skills") -> str:
    """Copy a local SKILL.md bundle folder into *skills_dir*. Returns its slug."""
    src = Path(path).expanduser()
    if not (src / "SKILL.md").is_file():
        raise SkillImportError(f"{src} has no SKILL.md")
    dest = _dest_dir(skills_dir) / src.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return src.name


def import_git(url: str, skills_dir: str = "~/.config/leuk/skills") -> str:
    """Shallow-clone a git repo whose root is a SKILL.md bundle. Returns its slug."""
    if shutil.which("git") is None:
        raise SkillImportError("git is not installed")
    slug = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    dest = _dest_dir(skills_dir) / slug
    if dest.exists():
        shutil.rmtree(dest)
    proc = _run(["git", "clone", "--depth", "1", url, str(dest)], timeout=120)
    if proc.returncode != 0:
        raise SkillImportError(proc.stderr.strip() or "git clone failed")
    shutil.rmtree(dest / ".git", ignore_errors=True)
    if not (dest / "SKILL.md").is_file():
        shutil.rmtree(dest, ignore_errors=True)
        raise SkillImportError("cloned repo has no SKILL.md at its root")
    return slug


def import_clawhub(slug: str, skills_dir: str = "~/.config/leuk/skills") -> str:
    """Install a ClawHub skill via the `clawhub` CLI into *skills_dir*.

    ClawHub's ``--dir`` is *relative to* ``--workdir``, so we point workdir at the
    skills dir's parent and dir at its name → it installs into ``<skills_dir>/<slug>``.
    Relies on the optional ``clawhub`` system CLI (see `/doctor`).
    """
    if shutil.which("clawhub") is None:
        raise SkillImportError(
            "the 'clawhub' CLI is not installed (see `leuk doctor` / `/doctor`)"
        )
    dest = _dest_dir(skills_dir)
    proc = _run(
        ["clawhub", "--no-input", "--workdir", str(dest.parent), "--dir", dest.name,
         "install", slug],
        timeout=90,
    )
    # clawhub sometimes exits non-zero with a spurious "Timeout" *after* doing the
    # work, so judge success by the installed bundle, not the return code.
    if (dest / slug / "SKILL.md").is_file():
        return slug
    raise SkillImportError(proc.stderr.strip() or f"clawhub install {slug} failed")


def search_clawhub(query: str, *, limit: int = 10) -> list[tuple[str, str]]:
    """Search ClawHub skills via the `clawhub` CLI. Returns ``[(slug, name), …]``.

    ClawHub's ``search`` has no JSON mode; it prints ``<slug>  <name>  (<score>)``
    rows, which we parse. Raises :class:`SkillImportError` if the CLI is missing.
    """
    if shutil.which("clawhub") is None:
        raise SkillImportError(
            "the 'clawhub' CLI is not installed (see `leuk doctor` / `/doctor`)"
        )
    proc = _run(["clawhub", "--no-input", "search", query, "--limit", str(limit)], timeout=30)
    # Results go to stdout; clawhub may then exit non-zero with a spurious
    # "Timeout", so parse rows first and only error if there were none.
    row = re.compile(r"^(?P<slug>\S+)\s+(?P<name>.+?)\s+\([\d.]+\)\s*$")
    out: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        m = row.match(line.strip())
        if m:
            out.append((m.group("slug"), m.group("name").strip()))
    if not out and proc.returncode != 0:
        raise SkillImportError(proc.stderr.strip() or "clawhub search failed")
    return out


# ── trust / enable / remove state (config.json ``skills`` section) ──


def _skills_section() -> dict:
    sec = load_persistent_config().get("skills", {})
    return sec if isinstance(sec, dict) else {}


def _save_skills_section(updates: dict) -> None:
    sec = _skills_section()
    sec.update(updates)
    save_persistent_config({"skills": sec})


def set_skill_trusted(slug: str, trusted: bool) -> None:
    """Add/remove *slug* from the trusted set in config.json."""
    cur = set(_skills_section().get("trusted", []))
    cur.add(slug) if trusted else cur.discard(slug)
    _save_skills_section({"trusted": sorted(cur)})


def set_skill_enabled(slug: str, enabled: bool) -> None:
    """Toggle a skill on/off via the ``disabled`` list in config.json."""
    cur = set(_skills_section().get("disabled", []))
    cur.discard(slug) if enabled else cur.add(slug)
    _save_skills_section({"disabled": sorted(cur)})


def remove_skill(slug: str, skills_dir: str = "~/.config/leuk/skills") -> bool:
    """Delete a skill bundle from disk and drop it from trusted/disabled lists."""
    bundle = Path(skills_dir).expanduser() / slug
    existed = bundle.is_dir()
    shutil.rmtree(bundle, ignore_errors=True)
    sec = _skills_section()
    _save_skills_section({
        "trusted": [s for s in sec.get("trusted", []) if s != slug],
        "disabled": [s for s in sec.get("disabled", []) if s != slug],
    })
    return existed
