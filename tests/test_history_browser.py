"""Tests for the interactive history browser (block model + expansion)."""

from __future__ import annotations

from leuk.cli.history_browser import _HistoryBrowser, build_blocks
from leuk.types import Message, Role, ToolCall, ToolResult


def _convo() -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content="sys"),  # skipped
        Message(role=Role.USER, content="hello"),
        Message(
            role=Role.ASSISTANT,
            content="Checking.",
            tool_calls=[ToolCall(id="t1", name="shell", arguments={"command": "ls"})],
        ),
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(
                tool_call_id="t1", name="shell", content="a\nb\n" + "z" * 600, is_error=False
            ),
        ),
        Message(role=Role.ASSISTANT, content="All done."),
    ]


class TestBuildBlocks:
    def test_skips_system_and_classifies(self):
        blocks = build_blocks(_convo())
        # user, assistant, tool, assistant (system dropped)
        assert len(blocks) == 4
        # only the tool block is expandable
        assert [b.expandable for b in blocks] == [False, False, True, False]

    def test_empty_conversation(self):
        assert build_blocks([]) == []


class TestBrowserState:
    def _text(self, b: _HistoryBrowser) -> str:
        return "".join(t for _s, t, *_ in b._ft)

    def test_selection_starts_at_latest(self):
        b = _HistoryBrowser(build_blocks(_convo()))
        assert b.selected == 3  # last block

    def test_navigation_clamps(self):
        b = _HistoryBrowser(build_blocks(_convo()))
        b._move(-100)
        assert b.selected == 0
        b._move(100)
        assert b.selected == 3

    def test_expand_collapse_changes_output(self):
        b = _HistoryBrowser(build_blocks(_convo()))
        b.selected = 2  # the tool block
        collapsed = self._text(b)
        assert "Tab → history" in collapsed  # truncation/expand marker present
        b._toggle()
        expanded = self._text(b)
        # The full output is now shown (rich word-wraps it, so count the chars).
        assert "Tab → history" not in expanded
        assert expanded.count("z") > collapsed.count("z")
        assert expanded.count("z") >= 590  # ~all 600 of the result's z's
        b._toggle()
        assert "Tab → history" in self._text(b)  # collapsed again

    def test_toggle_noop_on_non_expandable(self):
        b = _HistoryBrowser(build_blocks(_convo()))
        b.selected = 1  # assistant block (not expandable)
        before = self._text(b)
        b._toggle()
        assert self._text(b) == before
        assert 1 not in b.expanded

    def test_app_builds(self):
        b = _HistoryBrowser(build_blocks(_convo()))
        app = b.build_app()
        assert app.full_screen is True


class TestScrolling:
    def test_wheel_moves_cursor_and_is_consumed(self):
        """Wheel events move the cursor line and are consumed (return None) so the
        Window doesn't double-scroll; clicks select the block (return None)."""
        from types import SimpleNamespace

        from prompt_toolkit.mouse_events import MouseEventType

        b = _HistoryBrowser(build_blocks(_convo()))
        b._total_lines = 100
        b._cursor_line = 50
        handler = b._mouse_handler(0)
        assert handler(SimpleNamespace(event_type=MouseEventType.SCROLL_DOWN)) is None
        assert b._cursor_line == 53  # nudged down by 3
        assert handler(SimpleNamespace(event_type=MouseEventType.SCROLL_UP)) is None
        assert b._cursor_line == 50  # back up by 3
        assert handler(SimpleNamespace(event_type=MouseEventType.MOUSE_UP)) is None
        assert b.selected == 0  # the click selected block 0

    def test_scroll_clamps_to_bounds(self):
        b = _HistoryBrowser(build_blocks(_convo()))
        b._total_lines = 20
        b._cursor_line = 0
        b._scroll(-100)
        assert b._cursor_line == 0  # clamped at top
        b._scroll(100)
        assert b._cursor_line == 19  # clamped at last line

    def _fake_window(self, height: int):
        from types import SimpleNamespace

        return SimpleNamespace(render_info=SimpleNamespace(window_height=height))

    def test_scroll_page_moves_cursor(self):
        b = _HistoryBrowser(build_blocks(_convo()))
        b._total_lines = 100
        b._cursor_line = 0
        b._window = self._fake_window(height=10)
        b._scroll_page(1)
        assert b._cursor_line == 9  # one page down (height - 1)
        b._scroll_page(-1)
        assert b._cursor_line == 0  # back up, clamped at 0

    def test_cursor_position_clamped(self):
        b = _HistoryBrowser(build_blocks(_convo()))
        b._total_lines = 10
        b._cursor_line = 999
        assert b._cursor_position().y == 9  # never past the last line

    def test_no_window_is_safe(self):
        b = _HistoryBrowser(build_blocks(_convo()))
        b._window = None
        b._scroll_page(1)  # must not raise (falls back to a default height)
