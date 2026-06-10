"""Tests for slash-command + path completion (refactor-plan §5.8)."""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit.document import Document

from leuk.cli.repl import COMMANDS, SlashCommandCompleter


def _comps(completer: SlashCommandCompleter, text: str) -> list[str]:
    doc = Document(text, len(text))
    return [c.text for c in completer.get_completions(doc, None)]


class TestCommandCompletion:
    def test_prefix_matches(self):
        c = SlashCommandCompleter(COMMANDS)
        assert "/undo" in _comps(c, "/un")
        assert "/settings" in _comps(c, "/se")

    def test_non_slash_offers_nothing(self):
        c = SlashCommandCompleter(COMMANDS)
        assert _comps(c, "hello") == []

    def test_argument_of_plain_command_offers_nothing(self):
        c = SlashCommandCompleter(COMMANDS)
        assert _comps(c, "/switch ab") == []


class TestPathCompletion:
    def _tree(self, tmp_path: Path) -> Path:
        (tmp_path / "src").mkdir()
        (tmp_path / "sounds").mkdir()
        (tmp_path / "song.mp3").write_text("x")
        (tmp_path / ".hidden").mkdir()
        return tmp_path

    def test_cd_completes_directories_only(self, tmp_path: Path, monkeypatch):
        root = self._tree(tmp_path)
        monkeypatch.chdir(root)
        c = SlashCommandCompleter(COMMANDS)
        out = _comps(c, "/cd s")
        assert "src/" in out and "sounds/" in out
        assert "song.mp3" not in out  # dirs only for /cd

    def test_file_completes_files_too(self, tmp_path: Path, monkeypatch):
        root = self._tree(tmp_path)
        monkeypatch.chdir(root)
        c = SlashCommandCompleter(COMMANDS)
        out = _comps(c, "/file s")
        assert "song.mp3" in out and "src/" in out

    def test_hidden_entries_require_explicit_dot(self, tmp_path: Path, monkeypatch):
        root = self._tree(tmp_path)
        monkeypatch.chdir(root)
        c = SlashCommandCompleter(COMMANDS)
        assert ".hidden/" not in _comps(c, "/cd ")
        assert ".hidden/" in _comps(c, "/cd .h")

    def test_subdirectory_fragment(self, tmp_path: Path, monkeypatch):
        root = self._tree(tmp_path)
        (root / "src" / "leuk").mkdir()
        monkeypatch.chdir(root)
        c = SlashCommandCompleter(COMMANDS)
        assert "leuk/" in _comps(c, "/cd src/le")

    def test_nonexistent_directory_is_silent(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        c = SlashCommandCompleter(COMMANDS)
        assert _comps(c, "/cd nowhere/at/all/") == []
