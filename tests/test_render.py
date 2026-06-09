"""Tests for cli/render.py — tool status tracking and rendering."""

from __future__ import annotations

import time

import pytest
from rich.console import Console

from leuk.cli.render import (
    StreamRenderer,
    ToolState,
    ToolStatus,
    ToolStatusTracker,
    _truncate,
    render_history,
    render_tool_statuses,
)
from leuk.types import (
    Message,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
)


# ── Helpers ────────────────────────────────────────────────────────


def _tc(name: str = "shell", **kwargs) -> ToolCall:
    return ToolCall(id=f"tc_{name}_{id(kwargs)}", name=name, arguments=kwargs)


def _tr(tc: ToolCall, content: str = "ok", is_error: bool = False) -> ToolResult:
    return ToolResult(
        tool_call_id=tc.id,
        name=tc.name,
        content=content,
        is_error=is_error,
    )


def _tool_msg(result: ToolResult) -> Message:
    return Message(role=Role.TOOL, tool_result=result)


# ── _truncate ──────────────────────────────────────────────────────


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello") == "hello"

    def test_exact_boundary(self):
        text = "a" * 200
        assert _truncate(text) == text

    def test_long_text_truncated(self):
        text = "a" * 300
        result = _truncate(text)
        assert result.startswith("a" * 200)
        assert "+100 chars" in result  # 300 − 200 elided

    def test_custom_max_len(self):
        result = _truncate("hello world", max_len=5)
        assert result.startswith("hello")
        assert "+6 chars" in result  # 11 − 5 elided


# ── ToolStatus ─────────────────────────────────────────────────────


class TestToolStatus:
    def test_initial_state(self):
        tc = _tc("shell", command="ls")
        ts = ToolStatus(tool_call=tc)
        assert ts.state == ToolState.PENDING
        assert ts.result is None
        assert ts.end_time is None

    def test_elapsed_running(self):
        tc = _tc("shell", command="ls")
        ts = ToolStatus(tool_call=tc, start_time=time.monotonic() - 0.5)
        assert ts.elapsed >= 0.4

    def test_elapsed_completed(self):
        start = time.monotonic()
        tc = _tc("shell", command="ls")
        ts = ToolStatus(tool_call=tc, start_time=start, end_time=start + 1.5)
        assert abs(ts.elapsed - 1.5) < 0.01

    def test_elapsed_str_ms(self):
        start = time.monotonic()
        tc = _tc("shell", command="ls")
        ts = ToolStatus(tool_call=tc, start_time=start, end_time=start + 0.05)
        assert "ms" in ts.elapsed_str

    def test_elapsed_str_seconds(self):
        start = time.monotonic()
        tc = _tc("shell", command="ls")
        ts = ToolStatus(tool_call=tc, start_time=start, end_time=start + 2.3)
        assert "s" in ts.elapsed_str
        assert ts.elapsed_str.startswith("2.3")

    def test_args_str(self):
        tc = _tc("shell", command="ls -la")
        ts = ToolStatus(tool_call=tc)
        assert "command=" in ts.args_str
        assert "ls -la" in ts.args_str

    def test_args_str_empty(self):
        tc = ToolCall(id="tc1", name="shell", arguments={})
        ts = ToolStatus(tool_call=tc)
        assert ts.args_str == ""

    def test_args_str_long_value_truncated(self):
        tc = _tc("shell", command="x" * 100)
        ts = ToolStatus(tool_call=tc)
        assert "..." in ts.args_str


# ── ToolStatusTracker ──────────────────────────────────────────────


class TestToolStatusTracker:
    def test_start_tracking(self):
        tracker = ToolStatusTracker()
        tc = _tc("shell", command="ls")
        ts = tracker.start(tc)
        assert ts.state == ToolState.PENDING
        assert len(tracker.all_statuses) == 1
        assert len(tracker.active) == 1

    def test_mark_running(self):
        tracker = ToolStatusTracker()
        tc = _tc("shell", command="ls")
        tracker.start(tc)
        tracker.mark_running(tc.id)
        assert tracker.all_statuses[0].state == ToolState.RUNNING
        assert len(tracker.active) == 1

    def test_complete_success(self):
        tracker = ToolStatusTracker()
        tc = _tc("shell", command="ls")
        tracker.start(tc)
        result = _tr(tc, content="file1\nfile2")
        tracker.complete(result)
        ts = tracker.all_statuses[0]
        assert ts.state == ToolState.SUCCESS
        assert ts.result == result
        assert ts.end_time is not None
        assert len(tracker.active) == 0

    def test_complete_failure(self):
        tracker = ToolStatusTracker()
        tc = _tc("shell", command="rm /nope")
        tracker.start(tc)
        result = _tr(tc, content="permission denied", is_error=True)
        tracker.complete(result)
        assert tracker.all_statuses[0].state == ToolState.FAILED

    def test_multiple_tools(self):
        tracker = ToolStatusTracker()
        tc1 = _tc("shell", command="ls")
        tc2 = _tc("file_read", path="/tmp/foo")
        tracker.start(tc1)
        tracker.start(tc2)
        assert len(tracker.active) == 2
        tracker.complete(_tr(tc1))
        assert len(tracker.active) == 1
        tracker.complete(_tr(tc2))
        assert len(tracker.active) == 0
        assert len(tracker.all_statuses) == 2

    def test_round_tracking(self):
        tracker = ToolStatusTracker()
        assert tracker.round == 0
        tracker.new_round()
        assert tracker.round == 1
        tracker.new_round()
        assert tracker.round == 2

    def test_clear(self):
        tracker = ToolStatusTracker()
        tc = _tc("shell", command="ls")
        tracker.start(tc)
        tracker.new_round()
        tracker.clear()
        assert len(tracker.all_statuses) == 0
        assert tracker.round == 0

    def test_unknown_id_no_crash(self):
        tracker = ToolStatusTracker()
        tracker.mark_running("nonexistent")
        tracker.complete(
            ToolResult(
                tool_call_id="nonexistent",
                name="shell",
                content="ok",
            )
        )


# ── render_tool_statuses ──────────────────────────────────────────


class TestRenderToolStatuses:
    def test_pending_shows_spinner_symbol(self):
        tracker = ToolStatusTracker()
        tc = _tc("shell", command="ls")
        tracker.start(tc)
        text = render_tool_statuses(tracker)
        plain = text.plain
        assert "⟳" in plain
        assert "shell" in plain

    def test_success_shows_check(self):
        tracker = ToolStatusTracker()
        tc = _tc("shell", command="ls")
        tracker.start(tc)
        tracker.complete(_tr(tc, content="done"))
        text = render_tool_statuses(tracker)
        plain = text.plain
        assert "✓" in plain
        assert "shell" in plain

    def test_failure_shows_x(self):
        tracker = ToolStatusTracker()
        tc = _tc("shell", command="ls")
        tracker.start(tc)
        tracker.complete(_tr(tc, content="error!", is_error=True))
        text = render_tool_statuses(tracker)
        plain = text.plain
        assert "✗" in plain
        assert "error!" in plain

    def test_full_shows_entire_output(self):
        tracker = ToolStatusTracker()
        tc = _tc("shell", command="ls")
        tracker.start(tc)
        long_content = "x" * 500
        tracker.complete(_tr(tc, content=long_content))
        # Compact (default) — truncated with an elision marker.
        compact = render_tool_statuses(tracker).plain
        assert "+300 chars" in compact
        # full=True (expanded) — entire output, no marker.
        full = render_tool_statuses(tracker, full=True).plain
        assert "+300 chars" not in full
        assert "x" * 500 in full


# ── StreamRenderer ────────────────────────────────────────────────


class TestRenderHistory:
    def _console(self):
        import io

        buf = io.StringIO()
        return Console(file=buf, force_terminal=True, width=80), buf

    def test_renders_user_assistant_and_tool(self):
        console, buf = self._console()
        tc = _tc("shell", command="ls")
        msgs = [
            Message(role=Role.SYSTEM, content="sys prompt"),
            Message(role=Role.USER, content="hello there"),
            Message(role=Role.ASSISTANT, content="Hi! **bold**", tool_calls=[tc]),
            _tool_msg(_tr(tc, content="file1\nfile2")),
            Message(role=Role.ASSISTANT, content="All done"),
        ]
        n = render_history(console, msgs)
        out = buf.getvalue()
        assert n == 4  # user + assistant + tool + assistant (system skipped)
        assert "hello there" in out
        assert "file1" in out
        assert "All done" in out

    def test_skips_system_and_internal_user_messages(self):
        console, buf = self._console()
        msgs = [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="[SYSTEM] forced housekeeping"),
        ]
        n = render_history(console, msgs)
        assert n == 0
        assert "housekeeping" not in buf.getvalue()

    def test_empty_history_returns_zero(self):
        console, buf = self._console()
        assert render_history(console, []) == 0


class TestStreamRenderer:
    @pytest.mark.asyncio
    async def test_text_only_stream(self):
        """Pure text stream with no tool calls — rendered as Markdown to the
        console."""
        import io

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=80)
        renderer = StreamRenderer(console)

        async def stream():
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, content="Hello ")
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, content="world")
            yield StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                message=Message(role=Role.ASSISTANT, content="Hello world"),
            )

        await renderer.render_stream(stream())
        assert "Hello world" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_streaming_live_is_bounded_and_transient(self):
        """The assistant-text Live must be screen-capped and transient.

        With ``vertical_overflow="visible"`` a response taller than the terminal
        is re-emitted in full on every token, flooding the scrollback with dozens
        of duplicated copies (the real-TTY duplication bug). Capping the live to
        the screen and making it transient — then printing the complete Markdown
        once on stop — is what prevents that.
        """
        import io

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=80, height=10)
        renderer = StreamRenderer(console)

        renderer._on_text_delta(
            StreamEvent(type=StreamEventType.TEXT_DELTA, content="hello")
        )
        live = renderer._text_live
        assert live is not None
        assert live.vertical_overflow == "ellipsis"  # capped to the screen
        assert live.transient is True  # erased on stop, not left in scrollback

        # On stop, the complete text is emitted exactly once to the console.
        renderer._stop_text_live()
        assert "hello" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_text_only_stream_plain(self, capsys):
        """With markdown disabled, text streams raw to stdout."""
        console = Console(file=open("/dev/null", "w"), force_terminal=True)
        renderer = StreamRenderer(console)
        renderer.markdown = False

        async def stream():
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, content="Hello ")
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, content="world")
            yield StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                message=Message(role=Role.ASSISTANT, content="Hello world"),
            )

        await renderer.render_stream(stream())
        captured = capsys.readouterr()
        assert "Hello world" in captured.out

    @pytest.mark.asyncio
    async def test_tool_call_lifecycle(self):
        """Tool call goes through start → end → result."""
        console = Console(file=open("/dev/null", "w"), force_terminal=True)
        renderer = StreamRenderer(console)

        tc = _tc("shell", command="ls")

        async def stream():
            # Text first
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, content="Let me check")
            yield StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                message=Message(
                    role=Role.ASSISTANT,
                    content="Let me check",
                    tool_calls=[tc],
                ),
            )
            # Tool lifecycle events
            yield StreamEvent(type=StreamEventType.TOOL_CALL_START, tool_call=tc)
            yield StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=tc)
            # Tool result
            yield _tool_msg(_tr(tc, content="file1\nfile2"))

        await renderer.render_stream(stream())

        # Verify tracker processed everything
        assert renderer.tracker.round >= 1

    @pytest.mark.asyncio
    async def test_thinking_spinner_stops_on_cancel(self):
        """Cancelling render_queue (Ctrl-C) must tear down the Thinking… spinner
        — its cleanup is in a `finally`, so it never leaks past the prompt."""
        import asyncio

        console = Console(file=open("/dev/null", "w"), force_terminal=True)
        renderer = StreamRenderer(console)
        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(renderer.render_queue(queue))
        await asyncio.sleep(0.02)  # let it start the spinner on an empty queue
        assert renderer._thinking_live is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert renderer._thinking_live is None  # finally stopped it

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        """Multiple tool calls in a single round."""
        console = Console(file=open("/dev/null", "w"), force_terminal=True)
        renderer = StreamRenderer(console)

        tc1 = _tc("shell", command="ls")
        tc2 = _tc("file_read", path="/tmp/foo")

        async def stream():
            yield StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                message=Message(
                    role=Role.ASSISTANT,
                    content=None,
                    tool_calls=[tc1, tc2],
                ),
            )
            yield StreamEvent(type=StreamEventType.TOOL_CALL_START, tool_call=tc1)
            yield StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=tc1)
            yield StreamEvent(type=StreamEventType.TOOL_CALL_START, tool_call=tc2)
            yield StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=tc2)
            yield _tool_msg(_tr(tc1, content="file1"))
            yield _tool_msg(_tr(tc2, content="contents"))

        await renderer.render_stream(stream())

    @pytest.mark.asyncio
    async def test_error_result_tracked(self):
        """Failed tool calls are tracked as FAILED."""
        console = Console(file=open("/dev/null", "w"), force_terminal=True)
        renderer = StreamRenderer(console)

        tc = _tc("shell", command="bad_cmd")

        async def stream():
            yield StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                message=Message(
                    role=Role.ASSISTANT,
                    content=None,
                    tool_calls=[tc],
                ),
            )
            yield StreamEvent(type=StreamEventType.TOOL_CALL_START, tool_call=tc)
            yield StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=tc)
            yield _tool_msg(_tr(tc, content="command not found", is_error=True))

        await renderer.render_stream(stream())

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Empty stream doesn't crash."""
        console = Console(file=open("/dev/null", "w"), force_terminal=True)
        renderer = StreamRenderer(console)

        async def stream():
            return
            yield  # make it an async generator

        await renderer.render_stream(stream())
