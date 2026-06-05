"""Render :class:`~leuk.types.MediaPart` blocks for the terminal history browser.

Two modes (set via ``ui.media_render``):

* ``metadata`` — a compact one-line summary (kind, type, dimensions/size); never
  materializes the binary. The safe default for headless/SSH use.
* ``inline``   — a small 256-color ANSI thumbnail for images (audio/video fall
  back to the metadata line). Pressing ``Enter`` on the block opens/plays it via
  :func:`leuk.media.open_external`. Requires Pillow; without it, falls back to
  ``metadata``.
"""

from __future__ import annotations

import base64
import io

from leuk.media import png_size
from leuk.types import MediaPart

_MAX_THUMB_COLS = 40


def metadata_line(part: MediaPart) -> str:
    """A compact, binary-free description of a media part."""
    try:
        raw = base64.b64decode(part.data, validate=False)
        nbytes = len(raw)
    except Exception:  # noqa: BLE001
        raw, nbytes = b"", (len(part.data) * 3) // 4
    dims = ""
    if part.kind == "image":
        size = png_size(raw)
        if size:
            dims = f" {size[0]}×{size[1]}px"
    size_str = f"{nbytes / 1024:.1f}KB" if nbytes >= 1024 else f"{nbytes}B"
    return f"[{part.kind} · {part.media_type}{dims} · {size_str}]"


def _rgb_to_ansi256(r: int, g: int, b: int) -> int:
    """Nearest xterm-256 colour index for an RGB triple."""
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return 232 + (r - 8) * 24 // 247
    cube = [round(c / 255 * 5) for c in (r, g, b)]
    return 16 + 36 * cube[0] + 6 * cube[1] + cube[2]


def ansi_thumbnail(part: MediaPart, max_cols: int = _MAX_THUMB_COLS) -> str | None:
    """A 256-colour half-block ANSI thumbnail of an image, or None if unavailable.

    Each character cell shows two vertically stacked pixels via ``▀`` (upper half
    = foreground, lower half = background), doubling vertical resolution.
    """
    if part.kind != "image":
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(io.BytesIO(base64.b64decode(part.data))).convert("RGB")
    except Exception:  # noqa: BLE001 — corrupt/partial image
        return None
    w, h = img.size
    if w == 0 or h == 0:
        return None
    cols = max(1, min(max_cols, w))
    rows = max(1, round(cols * h / w / 2))  # /2: a cell is ~2x taller than wide
    img = img.resize((cols, rows * 2))
    data = img.tobytes()  # raw RGB, row-major, 3 bytes/pixel

    def _px(col: int, row: int) -> tuple[int, int, int]:
        i = (row * cols + col) * 3
        return data[i], data[i + 1], data[i + 2]

    lines: list[str] = []
    for r in range(rows):
        cells: list[str] = []
        for c in range(cols):
            top = _rgb_to_ansi256(*_px(c, 2 * r))
            bot = _rgb_to_ansi256(*_px(c, 2 * r + 1))
            cells.append(f"\x1b[38;5;{top}m\x1b[48;5;{bot}m▀")
        cells.append("\x1b[0m")
        lines.append("".join(cells))
    return "\n".join(lines)


def render_media(part: MediaPart, mode: str = "metadata", *, width: int = _MAX_THUMB_COLS) -> str:
    """Render a media part per *mode*. Always falls back to the metadata line."""
    meta = metadata_line(part)
    if mode != "inline":
        return meta
    thumb = ansi_thumbnail(part, max_cols=width)
    if thumb is None:
        return meta  # audio/video, or no Pillow → metadata only
    return f"{meta}\n{thumb}\n[Enter] open in viewer"
