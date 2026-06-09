"""Tests for the shared scrollback block model (cli/blocks.py)."""

from __future__ import annotations

from rich.text import Text

from leuk.cli.blocks import Block, build_blocks, rich_to_ansi
from leuk.types import Message, Role


def _convo() -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content="sys"),
        Message(role=Role.USER, content="hello"),
        Message(role=Role.ASSISTANT, content="**hi** there"),
    ]


class TestRichToAnsi:
    def test_returns_string(self):
        out = rich_to_ansi(Text("hello"), width=40)
        assert isinstance(out, str)
        assert "hello" in out

    def test_no_trailing_newline(self):
        assert not rich_to_ansi(Text("x"), width=40).endswith("\n")


class TestBuildBlocks:
    def test_skips_system_messages(self):
        blocks = build_blocks(_convo())
        # user + assistant blocks only (system skipped)
        assert len(blocks) == 2
        assert all(isinstance(b, Block) for b in blocks)

    def test_empty_conversation(self):
        assert build_blocks([]) == []

    def test_assistant_block_not_expandable(self):
        blocks = build_blocks(_convo())
        # The user line and assistant markdown are static (not expandable).
        assert all(not b.expandable for b in blocks)
