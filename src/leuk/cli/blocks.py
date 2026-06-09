"""Shared scrollback **block model** and the Rich→ANSI bridge.

A *block* is one renderable entry in a vertical conversation view — a user
turn, an assistant reply, a tool / sub-agent call, or a media attachment. Each
block knows how to render itself to an ANSI string at a given width (tool blocks
render compact or full depending on an *expanded* flag).

This is the single source of truth for the **persistent-input TUI**
(``cli/tui.py``, design in ``docs/repl-tui-design.md``): its scrollback uses
these blocks and the ``rich_to_ansi`` bridge both for live-streamed turns and
for rebuilding the transcript from history on re-entry. ``rich`` stays the
*content* renderer; this module only bridges it to ANSI for prompt_toolkit.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from leuk.cli.render import ToolState, ToolStatus, _code_theme, render_tool_block
from leuk.cli.theme import LEUK_THEME
from leuk.media import extract_media, open_external
from leuk.media_render import render_media
from leuk.types import MediaPart, Message, Role, ToolCall, ToolResult


@dataclass
class Block:
    """One renderable entry in a scrollback list."""

    expandable: bool
    # render(full, width) -> ANSI string for the block body.
    render: Callable[[bool, int], str]
    # When set, this is a media block: Enter/click "activates" it (opens/plays)
    # instead of expanding text.
    on_activate: Callable[[], object] | None = None


def rich_to_ansi(renderable: object, width: int) -> str:
    """Render a Rich renderable to an ANSI string at *width* columns."""
    buf = io.StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        color_system="standard",
        width=max(20, width),
        theme=LEUK_THEME,
    )
    console.print(renderable)
    return buf.getvalue().rstrip("\n")


def render_static(renderable: object, full: bool, width: int) -> str:
    """Block body for a fixed renderable (user/assistant); ignores *full*."""
    return rich_to_ansi(renderable, width)


def render_tool(ts: ToolStatus, full: bool, width: int) -> str:
    """Block body for a tool/sub-agent call — compact, or full when expanded."""
    return rich_to_ansi(render_tool_block(ts, full=full), width)


def render_media_body(part: MediaPart, mode: str, full: bool, width: int) -> str:
    """Block body for an image/audio/video attachment (already-ANSI string)."""
    return render_media(part, mode, width=min(max(8, width - 2), 40))


def static_ansi_block(ansi: str) -> Block:
    """A non-expandable block that renders a fixed, already-ANSI string.

    Used for content that is captured as ANSI elsewhere (the startup banner,
    a slash-command's captured terminal output) and just passed through.
    """

    def _render(full: bool, width: int) -> str:  # noqa: ARG001 — fixed content
        return ansi

    return Block(False, _render)


def media_block(part: MediaPart, mode: str) -> Block:
    return Block(
        expandable=False,
        render=partial(render_media_body, part, mode),
        on_activate=partial(open_external, part),
    )


def tool_result_blocks(
    tool_call: ToolCall, tool_result: ToolResult, *, media_mode: str = "metadata"
) -> list[Block]:
    """Blocks for one tool result: a compact tool block + any media thumbnails.

    Media (e.g. screenshots) is extracted from the result content so the block
    shows clean text instead of a raw base64 blob; each media part becomes its
    own thumbnail/metadata block (shared by the live TUI and history rebuild).
    """
    clean, media = extract_media(tool_result.content or "")
    tr = tool_result
    if media:  # render the tool block without the raw base64 blob
        tr = ToolResult(
            tool_call_id=tr.tool_call_id, name=tr.name, content=clean,
            metadata=tr.metadata, is_error=tr.is_error,
        )
    ts = ToolStatus(
        tool_call=tool_call,
        state=ToolState.FAILED if tr.is_error else ToolState.SUCCESS,
        result=tr,
    )
    ts.end_time = None
    blocks: list[Block] = [Block(True, partial(render_tool, ts))]
    blocks.extend(media_block(part, media_mode) for part in media)
    return blocks


def build_blocks(messages: list[Message], *, media_mode: str = "metadata") -> list[Block]:
    """Turn a conversation into scrollback blocks (user / assistant / tool / media)."""
    calls_by_id: dict[str, ToolCall] = {}
    for m in messages:
        for tc in m.tool_calls or []:
            calls_by_id[tc.id] = tc

    blocks: list[Block] = []
    for m in messages:
        if m.role is Role.SYSTEM:
            continue
        if m.role is Role.USER:
            content = (m.content or "").strip()
            if content and not content.startswith("[SYSTEM]"):
                line = Text()
                line.append("❯ ", style="user.label")
                line.append(content, style="primary")
                blocks.append(Block(False, partial(render_static, line)))
            for att in m.attachments or []:
                blocks.append(media_block(att, media_mode))
        elif m.role is Role.ASSISTANT:
            if m.content and m.content.strip():
                md = Markdown(m.content, code_theme=_code_theme())
                blocks.append(Block(False, partial(render_static, md)))
        elif m.role is Role.TOOL and m.tool_result:
            clean, media = extract_media(m.tool_result.content or "")
            tr = m.tool_result
            if media:  # render the tool block without the raw base64 blob
                tr = ToolResult(
                    tool_call_id=tr.tool_call_id, name=tr.name, content=clean,
                    metadata=tr.metadata, is_error=tr.is_error,
                )
            tc = calls_by_id.get(tr.tool_call_id) or ToolCall(
                id=tr.tool_call_id, name=tr.name, arguments={}
            )
            ts = ToolStatus(
                tool_call=tc,
                state=ToolState.FAILED if tr.is_error else ToolState.SUCCESS,
                result=tr,
            )
            ts.end_time = None
            blocks.append(Block(True, partial(render_tool, ts)))
            for part in media:
                blocks.append(media_block(part, media_mode))
    return blocks
