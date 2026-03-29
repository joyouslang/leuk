"""Tests for the hierarchical memory system."""

from __future__ import annotations

from pathlib import Path


from leuk.memory.loader import MemoryLoader


class TestMemoryLoader:
    def test_empty_memory_dir(self, tmp_path: Path):
        loader = MemoryLoader(memory_dir=str(tmp_path), project_name="proj")
        assert loader.load() == ""

    def test_global_only(self, tmp_path: Path):
        (tmp_path / "GLOBAL.md").write_text("global content")
        loader = MemoryLoader(memory_dir=str(tmp_path), project_name="proj")
        result = loader.load()
        assert "global content" in result
        assert "Global Memory" in result

    def test_project_only(self, tmp_path: Path):
        proj_dir = tmp_path / "projects" / "myapp"
        proj_dir.mkdir(parents=True)
        (proj_dir / "MEMORY.md").write_text("project content")
        loader = MemoryLoader(memory_dir=str(tmp_path), project_name="myapp")
        result = loader.load()
        assert "project content" in result
        assert "Project Memory (myapp)" in result

    def test_both_global_and_project(self, tmp_path: Path):
        (tmp_path / "GLOBAL.md").write_text("global info")
        proj_dir = tmp_path / "projects" / "myapp"
        proj_dir.mkdir(parents=True)
        (proj_dir / "MEMORY.md").write_text("project info")
        loader = MemoryLoader(memory_dir=str(tmp_path), project_name="myapp")
        result = loader.load()
        assert "global info" in result
        assert "project info" in result
        # Global should appear before project
        assert result.index("global info") < result.index("project info")

    def test_token_budget_truncates_global(self, tmp_path: Path):
        # Create global content that exceeds the budget
        long_global = "x" * 5000
        (tmp_path / "GLOBAL.md").write_text(long_global)
        # Small budget
        loader = MemoryLoader(memory_dir=str(tmp_path), project_name="p", token_budget=100)
        result = loader.load()
        # Result must fit within budget (chars = tokens * 4)
        assert len(result) <= 100 * 4 + 200  # some header overhead tolerated

    def test_token_budget_preserves_project_over_global(self, tmp_path: Path):
        # Global is huge, project is small — project must survive
        (tmp_path / "GLOBAL.md").write_text("G" * 10000)
        proj_dir = tmp_path / "projects" / "p"
        proj_dir.mkdir(parents=True)
        (proj_dir / "MEMORY.md").write_text("critical project note")
        loader = MemoryLoader(memory_dir=str(tmp_path), project_name="p", token_budget=200)
        result = loader.load()
        assert "critical project note" in result

    def test_detect_project_name_explicit(self, tmp_path: Path):
        loader = MemoryLoader(memory_dir=str(tmp_path), project_name="explicit")
        assert loader.detect_project_name() == "explicit"

    def test_estimate_tokens(self):
        assert MemoryLoader._estimate_tokens("a" * 400) == 100
        assert MemoryLoader._estimate_tokens("") == 0
