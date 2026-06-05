"""Multimodal media helpers — parsing inline media tags and loading files.

Tools emit screenshots/images/videos as inline tags in their text result, e.g.::

    [screenshot:image/png;base64,iVBORw0KGgo...]
    [image:image/jpeg;base64,...]
    [audio:audio/wav;base64,...]
    [video:video/mp4;base64,...]

:func:`extract_media` pulls those out as :class:`~leuk.types.MediaPart` objects
(so providers can send them as native image/audio/video blocks) and returns the
text with the tags removed. :func:`media_to_tag` is the inverse. :func:`load_media_file`
reads an image/audio/video file from disk into a ``MediaPart``.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import re

from leuk.types import MediaPart, Message, ToolResult

# [screenshot:<media-type>;base64,<data>] / [image:...] / [audio:...] / [video:...]
_TAG_RE = re.compile(
    r"\[(screenshot|image|audio|video):([\w.+-]+/[\w.+-]+);base64,([A-Za-z0-9+/=\s]+?)\]",
    re.DOTALL,
)

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm"}
_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".mpeg", ".mpg", ".m4v"}


def _kind_for(tag: str, media_type: str) -> str:
    if tag == "audio" or media_type.startswith("audio/"):
        return "audio"
    if tag == "video" or media_type.startswith("video/"):
        return "video"
    return "image"


def extract_media(text: str) -> tuple[str, list[MediaPart]]:
    """Split *text* into (clean_text, media_parts), removing inline media tags."""
    if not text or "[" not in text:
        return text, []
    parts: list[MediaPart] = []

    def _sub(m: re.Match[str]) -> str:
        tag, media_type, data = m.group(1), m.group(2), m.group(3)
        kind = _kind_for(tag, media_type)
        parts.append(MediaPart(kind=kind, media_type=media_type, data="".join(data.split())))
        # Leave a short placeholder so the text still references the media.
        return f"[{tag} attached]"

    clean = _TAG_RE.sub(_sub, text)
    return clean, parts


def media_to_tag(part: MediaPart) -> str:
    """Serialize a :class:`MediaPart` back to an inline tag (inverse of
    :func:`extract_media`), so media can survive text processing intact."""
    return f"[{part.kind}:{part.media_type};base64,{part.data}]"


def strip_media(messages: list[Message], *, note: str) -> list[Message]:
    """Remove all media (inline tool tags and user attachments) from *messages*,
    leaving a short *note* in their place.

    Used when the active model has no vision support: rather than send a base64
    blob the model would "read" as text (and hallucinate over) or an image block
    the API would reject, the model gets a clear note that the media was omitted.
    Originals are not mutated.
    """
    out: list[Message] = []
    for msg in messages:
        new_tr = msg.tool_result
        new_atts = msg.attachments
        new_content = msg.content

        if msg.tool_result and "base64," in (msg.tool_result.content or ""):
            clean, media = extract_media(msg.tool_result.content)
            if media:
                clean += f"\n[{len(media)} media item(s) omitted — {note}]"
                new_tr = ToolResult(
                    tool_call_id=msg.tool_result.tool_call_id,
                    name=msg.tool_result.name,
                    content=clean,
                    metadata=msg.tool_result.metadata,
                    is_error=msg.tool_result.is_error,
                )

        if msg.attachments:
            new_content = (msg.content or "") + (
                f"\n[{len(msg.attachments)} attachment(s) omitted — {note}]"
            )
            new_atts = None

        if new_tr is msg.tool_result and new_atts is msg.attachments and new_content is msg.content:
            out.append(msg)
        else:
            out.append(
                Message(
                    role=msg.role,
                    content=new_content,
                    tool_calls=msg.tool_calls,
                    tool_result=new_tr,
                    timestamp=msg.timestamp,
                    metadata=msg.metadata,
                    attachments=new_atts,
                )
            )
    return out


def png_size(data: bytes) -> tuple[int, int] | None:
    """Read (width, height) from a PNG's IHDR header, or None if not a PNG."""
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    return None


def open_external(part: MediaPart) -> str:
    """Write a media part to a temp file and open it in the OS default handler.

    Returns the temp-file path. Used by the history browser to view/play media.
    """
    ext = mimetypes.guess_extension(part.media_type) or {
        "image": ".png", "audio": ".wav", "video": ".mp4"
    }.get(part.kind, ".bin")
    fd, path = tempfile.mkstemp(suffix=ext, prefix="leuk-media-")
    with os.fdopen(fd, "wb") as fh:
        fh.write(base64.b64decode(part.data))
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]  # noqa: S606 — Windows only
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen(
            [opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    return path


def load_media_file(path: str) -> MediaPart:
    """Load an image/audio/video file from disk into a :class:`MediaPart`.

    Raises ``FileNotFoundError`` / ``ValueError`` on problems.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {p}")
    ext = p.suffix.lower()
    if ext in _IMAGE_EXT:
        kind = "image"
    elif ext in _VIDEO_EXT:
        kind = "video"
    elif ext in _AUDIO_EXT:
        kind = "audio"
    else:
        raise ValueError(f"Unsupported media type for {p.name!r} (ext {ext!r})")
    media_type, _ = mimetypes.guess_type(str(p))
    if not media_type:
        media_type = {"image": "image/png", "audio": "audio/wav", "video": "video/mp4"}[kind]
    data = base64.b64encode(p.read_bytes()).decode()
    return MediaPart(kind=kind, media_type=media_type, data=data)
