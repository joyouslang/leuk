"""Tests for the Agent Skills (SKILL.md) runtime: loader, trust gate, tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from leuk.skills.loader import (
    SkillImportError,
    SkillLoader,
    _parse_frontmatter,
    import_local,
)
from leuk.skills.tool import SkillTool


def _make_skill(root: Path, slug: str, name: str, desc: str, *, body: str = "do things",
                extra: dict[str, str] | None = None) -> Path:
    d = root / slug
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n{body}\n")
    for rel, content in (extra or {}).items():
        f = d / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
    return d


class TestFrontmatter:
    def test_parses_name_description(self):
        fields, body = _parse_frontmatter("---\nname: PDF\ndescription: Read PDFs\n---\nbody here")
        assert fields["name"] == "PDF"
        assert fields["description"] == "Read PDFs"
        assert body == "body here"

    def test_strips_quotes(self):
        fields, _ = _parse_frontmatter('---\nname: "Quoted"\n---\nx')
        assert fields["name"] == "Quoted"

    def test_no_frontmatter(self):
        fields, body = _parse_frontmatter("just text")
        assert fields == {} and body == "just text"

    def test_unterminated_frontmatter_is_safe(self):
        fields, body = _parse_frontmatter("---\nname: X\nno close")
        assert fields == {} and body.startswith("---")


class TestLoader:
    def test_discovers_and_flags_trust_enable(self, tmp_path):
        _make_skill(tmp_path, "pdf", "PDF", "Read PDFs")
        _make_skill(tmp_path, "csv", "CSV", "Read CSVs")
        loader = SkillLoader(str(tmp_path), trusted={"pdf"}, disabled={"csv"})
        by_slug = {m.slug: m for m in loader.all_skills()}
        assert by_slug["pdf"].trusted and by_slug["pdf"].enabled
        assert not by_slug["csv"].trusted and not by_slug["csv"].enabled

    def test_usable_only_trusted_and_enabled(self, tmp_path):
        _make_skill(tmp_path, "pdf", "PDF", "Read PDFs")
        _make_skill(tmp_path, "csv", "CSV", "Read CSVs")
        _make_skill(tmp_path, "img", "IMG", "Images")
        loader = SkillLoader(str(tmp_path), trusted={"pdf", "csv"}, disabled={"csv"})
        usable = {m.slug for m in loader.usable()}
        assert usable == {"pdf"}  # csv trusted-but-disabled, img untrusted

    def test_usable_respects_max_index(self, tmp_path):
        for i in range(5):
            _make_skill(tmp_path, f"s{i}", f"S{i}", "d")
        loader = SkillLoader(str(tmp_path), trusted={f"s{i}" for i in range(5)}, max_index=2)
        assert len(loader.usable()) == 2

    def test_read_returns_body_and_manifest(self, tmp_path):
        _make_skill(tmp_path, "pdf", "PDF", "Read PDFs", body="step 1",
                    extra={"scripts/run.py": "print(1)"})
        loader = SkillLoader(str(tmp_path), trusted={"pdf"})
        out = loader.read("PDF")
        assert "step 1" in out
        assert "scripts/run.py" in out  # bundle manifest included
        assert str(tmp_path / "pdf") in out

    def test_read_untrusted_returns_none(self, tmp_path):
        _make_skill(tmp_path, "pdf", "PDF", "Read PDFs")
        loader = SkillLoader(str(tmp_path), trusted=set())
        assert loader.read("pdf") is None

    def test_find_by_name_or_slug(self, tmp_path):
        _make_skill(tmp_path, "pdf-tools", "PDF Tools", "Read PDFs")
        loader = SkillLoader(str(tmp_path), trusted={"pdf-tools"})
        assert loader.find("pdf-tools").name == "PDF Tools"
        assert loader.find("PDF TOOLS").slug == "pdf-tools"


class TestSkillTool:
    def test_spec_lists_only_usable(self, tmp_path):
        _make_skill(tmp_path, "pdf", "PDF", "Read PDFs")
        _make_skill(tmp_path, "secret", "Secret", "untrusted")
        tool = SkillTool(SkillLoader(str(tmp_path), trusted={"pdf"}))
        desc = tool.spec.description
        assert "PDF: Read PDFs" in desc
        assert "Secret" not in desc

    def test_spec_no_skills(self, tmp_path):
        tool = SkillTool(SkillLoader(str(tmp_path)))
        assert "No skills" in tool.spec.description

    @pytest.mark.asyncio
    async def test_execute_read(self, tmp_path):
        _make_skill(tmp_path, "pdf", "PDF", "Read PDFs", body="extract text")
        tool = SkillTool(SkillLoader(str(tmp_path), trusted={"pdf"}))
        out = await tool.execute({"action": "read", "name": "PDF"})
        assert "extract text" in out

    @pytest.mark.asyncio
    async def test_execute_read_untrusted(self, tmp_path):
        _make_skill(tmp_path, "pdf", "PDF", "Read PDFs")
        tool = SkillTool(SkillLoader(str(tmp_path)))  # not trusted
        out = await tool.execute({"action": "read", "name": "pdf"})
        assert "[ERROR]" in out and "untrusted" in out

    @pytest.mark.asyncio
    async def test_execute_list(self, tmp_path):
        _make_skill(tmp_path, "pdf", "PDF", "Read PDFs")
        tool = SkillTool(SkillLoader(str(tmp_path), trusted={"pdf"}))
        out = await tool.execute({"action": "list"})
        assert "PDF" in out


class TestImporters:
    def test_import_local_copies_bundle(self, tmp_path):
        src = _make_skill(tmp_path / "src", "pdf", "PDF", "Read PDFs",
                          extra={"scripts/run.sh": "echo hi"})
        dest_dir = tmp_path / "installed"
        slug = import_local(str(src), str(dest_dir))
        assert slug == "pdf"
        assert (dest_dir / "pdf" / "SKILL.md").is_file()
        assert (dest_dir / "pdf" / "scripts" / "run.sh").is_file()

    def test_import_local_rejects_non_skill(self, tmp_path):
        (tmp_path / "notaskill").mkdir()
        with pytest.raises(SkillImportError):
            import_local(str(tmp_path / "notaskill"), str(tmp_path / "installed"))

    def test_import_clawhub_missing_cli(self, tmp_path, monkeypatch):
        import leuk.skills.loader as ld

        monkeypatch.setattr(ld.shutil, "which", lambda name: None)
        with pytest.raises(SkillImportError, match="clawhub"):
            ld.import_clawhub("some-slug", str(tmp_path))

    def test_import_clawhub_uses_workdir_dir(self, tmp_path, monkeypatch):
        """`--dir` is relative to `--workdir`, so install must target <dir>/<slug>."""
        import subprocess

        import leuk.skills.loader as ld

        monkeypatch.setattr(ld.shutil, "which", lambda name: "/usr/bin/clawhub")
        captured = {}
        skills_dir = tmp_path / "leuk" / "skills"

        def _fake_run(cmd, **kw):
            captured["cmd"] = cmd
            # Simulate clawhub installing the bundle but exiting non-zero (its
            # spurious post-work "Timeout"): success must be judged by the files.
            bundle = skills_dir / "demo"
            bundle.mkdir(parents=True, exist_ok=True)
            (bundle / "SKILL.md").write_text("---\nname: Demo\n---\nx")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Timeout")

        monkeypatch.setattr(ld.subprocess, "run", _fake_run)
        assert ld.import_clawhub("demo", str(skills_dir)) == "demo"
        cmd = captured["cmd"]
        assert cmd[:2] == ["clawhub", "--no-input"]
        assert "--workdir" in cmd and str(skills_dir.parent) in cmd
        assert "--dir" in cmd and "skills" in cmd
        assert cmd[-2:] == ["install", "demo"]

    def test_search_clawhub_lenient_on_nonzero_exit(self, monkeypatch):
        """clawhub prints rows then exits non-zero with 'Timeout' — keep the rows."""
        import subprocess

        import leuk.skills.loader as ld

        monkeypatch.setattr(ld.shutil, "which", lambda name: "/usr/bin/clawhub")
        out = "- Searching\npdf  Pdf  (3.690)\n"
        monkeypatch.setattr(
            ld.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout=out, stderr="Timeout"),
        )
        assert ld.search_clawhub("pdf") == [("pdf", "Pdf")]

    def test_search_clawhub_parses_rows(self, monkeypatch):
        import subprocess

        import leuk.skills.loader as ld

        monkeypatch.setattr(ld.shutil, "which", lambda name: "/usr/bin/clawhub")
        out = "- Searching\npdf  Pdf  (3.690)\ndocument-pdf  Document PDF  (2.5)\n"
        monkeypatch.setattr(
            ld.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=out, stderr=""),
        )
        results = ld.search_clawhub("pdf")
        assert results == [("pdf", "Pdf"), ("document-pdf", "Document PDF")]
