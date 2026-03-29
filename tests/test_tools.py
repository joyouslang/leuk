"""Tests for built-in tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from leuk.tools.file_edit import FileEditTool
from leuk.tools.file_read import FileReadTool
from leuk.tools.memory_write import MemoryWriteTool
from leuk.tools.shell import ShellTool
from leuk.tools.sub_agent import SubAgentTool
from leuk.tools.web_fetch import WebFetchTool, _html_to_text
from leuk.tools import create_default_registry


# ── Shell tool ──────────────────────────────────────────────────────

class TestShellTool:
    @pytest.fixture
    def tool(self):
        return ShellTool()

    @pytest.mark.asyncio
    async def test_simple_command(self, tool):
        result = await tool.execute({"command": "echo hello"})
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_exit_code(self, tool):
        result = await tool.execute({"command": "false"})
        assert "[exit code" in result

    @pytest.mark.asyncio
    async def test_timeout(self, tool):
        result = await tool.execute({"command": "sleep 10", "timeout": 1})
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_workdir(self, tool, tmp_path: Path):
        result = await tool.execute({"command": "pwd", "workdir": str(tmp_path)})
        assert str(tmp_path) in result

    def test_spec(self, tool):
        s = tool.spec
        assert s.name == "shell"
        assert "command" in s.parameters["properties"]


# ── File read tool ──────────────────────────────────────────────────

class TestFileReadTool:
    @pytest.fixture
    def tool(self):
        return FileReadTool()

    @pytest.mark.asyncio
    async def test_read_file(self, tool, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        result = await tool.execute({"path": str(f)})
        assert "line1" in result
        assert "3 lines total" in result

    @pytest.mark.asyncio
    async def test_read_with_offset_limit(self, tool, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("\n".join(f"line{i}" for i in range(10)))
        result = await tool.execute({"path": str(f), "offset": 2, "limit": 3})
        assert "line2" in result
        assert "line4" in result
        assert "line0" not in result

    @pytest.mark.asyncio
    async def test_file_not_found(self, tool):
        result = await tool.execute({"path": "/nonexistent/file.txt"})
        assert "[ERROR]" in result

    def test_spec(self, tool):
        s = tool.spec
        assert s.name == "file_read"


# ── File edit tool ──────────────────────────────────────────────────

class TestFileEditTool:
    @pytest.fixture
    def tool(self):
        return FileEditTool()

    @pytest.mark.asyncio
    async def test_create_file(self, tool, tmp_path: Path):
        f = tmp_path / "new.txt"
        result = await tool.execute({"path": str(f), "new_string": "hello world"})
        assert "Created" in result
        assert f.read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_edit_file(self, tool, tmp_path: Path):
        f = tmp_path / "edit.txt"
        f.write_text("foo bar baz")
        result = await tool.execute(
            {"path": str(f), "old_string": "bar", "new_string": "qux"}
        )
        assert "1 replacement" in result
        assert f.read_text() == "foo qux baz"

    @pytest.mark.asyncio
    async def test_edit_not_found(self, tool, tmp_path: Path):
        f = tmp_path / "edit.txt"
        f.write_text("hello")
        result = await tool.execute(
            {"path": str(f), "old_string": "xyz", "new_string": "abc"}
        )
        assert "[ERROR]" in result

    @pytest.mark.asyncio
    async def test_edit_ambiguous(self, tool, tmp_path: Path):
        f = tmp_path / "edit.txt"
        f.write_text("aaa aaa aaa")
        result = await tool.execute(
            {"path": str(f), "old_string": "aaa", "new_string": "bbb"}
        )
        assert "[ERROR]" in result
        assert "3 times" in result

    @pytest.mark.asyncio
    async def test_replace_all(self, tool, tmp_path: Path):
        f = tmp_path / "edit.txt"
        f.write_text("aaa aaa aaa")
        result = await tool.execute(
            {"path": str(f), "old_string": "aaa", "new_string": "bbb", "replace_all": True}
        )
        assert "3 replacement" in result
        assert f.read_text() == "bbb bbb bbb"

    def test_spec(self, tool):
        s = tool.spec
        assert s.name == "file_edit"


# ── Web fetch tool ──────────────────────────────────────────────────

class TestWebFetchTool:
    def test_spec(self):
        tool = WebFetchTool()
        s = tool.spec
        assert s.name == "web_fetch"
        assert "url" in s.parameters["properties"]

    def test_html_to_text(self):
        html = "<html><body><p>Hello world</p><script>evil()</script></body></html>"
        text = _html_to_text(html)
        assert "Hello world" in text
        assert "evil" not in text

    def test_html_to_text_with_selector(self):
        html = '<html><body><div class="content">Target</div><div>Other</div></body></html>'
        text = _html_to_text(html, selector=".content")
        assert "Target" in text
        assert "Other" not in text


# ── Sub-agent tool ──────────────────────────────────────────────────

class TestSubAgentTool:
    def test_spec(self):
        tool = SubAgentTool()
        s = tool.spec
        assert s.name == "sub_agent"
        assert "task" in s.parameters["properties"]

    @pytest.mark.asyncio
    async def test_no_manager(self):
        tool = SubAgentTool()
        result = await tool.execute({"task": "do something"})
        assert "[ERROR]" in result


# ── Memory write tool ───────────────────────────────────────────────

class TestMemoryWriteTool:
    def test_spec(self, tmp_path: Path):
        tool = MemoryWriteTool(memory_dir=str(tmp_path))
        s = tool.spec
        assert s.name == "memory_write"
        assert "scope" in s.parameters["properties"]
        assert "content" in s.parameters["properties"]

    @pytest.mark.asyncio
    async def test_write_global(self, tmp_path: Path):
        tool = MemoryWriteTool(memory_dir=str(tmp_path))
        result = await tool.execute({"scope": "global", "content": "remember this"})
        assert "global" in result
        target = tmp_path / "GLOBAL.md"
        assert target.exists()
        assert "remember this" in target.read_text()

    @pytest.mark.asyncio
    async def test_write_project(self, tmp_path: Path):
        tool = MemoryWriteTool(memory_dir=str(tmp_path), project_name="myproject")
        result = await tool.execute({"scope": "project", "content": "project note"})
        assert "project" in result
        target = tmp_path / "projects" / "myproject" / "MEMORY.md"
        assert target.exists()
        assert "project note" in target.read_text()

    @pytest.mark.asyncio
    async def test_write_project_no_name(self, tmp_path: Path):
        tool = MemoryWriteTool(memory_dir=str(tmp_path), project_name="")
        result = await tool.execute({"scope": "project", "content": "something"})
        assert "[ERROR]" in result

    @pytest.mark.asyncio
    async def test_append_global(self, tmp_path: Path):
        tool = MemoryWriteTool(memory_dir=str(tmp_path))
        await tool.execute({"scope": "global", "content": "first"})
        await tool.execute({"scope": "global", "content": "second"})
        text = (tmp_path / "GLOBAL.md").read_text()
        assert "first" in text
        assert "second" in text


# ── Registry ────────────────────────────────────────────────────────

def test_default_registry():
    reg = create_default_registry()
    names = {s.name for s in reg.specs()}
    assert names == {"shell", "file_read", "file_edit", "sub_agent", "web_fetch", "memory_write"}
    assert len(reg) == 6
