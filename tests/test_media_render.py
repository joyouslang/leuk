"""Tests for media rendering in the history browser (metadata vs inline)."""

from __future__ import annotations

import base64
import io

import pytest

from leuk.media_render import ansi_thumbnail, metadata_line, render_media
from leuk.types import MediaPart


def _png(width: int, height: int, color=(0, 80, 160)) -> str:
    pytest.importorskip("PIL")
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestMetadataLine:
    def test_image_includes_dims_and_size(self):
        part = MediaPart(kind="image", media_type="image/png", data=_png(64, 32))
        line = metadata_line(part)
        assert "image" in line and "image/png" in line and "64×32px" in line

    def test_audio_has_no_dims(self):
        part = MediaPart(kind="audio", media_type="audio/wav", data=base64.b64encode(b"x" * 500).decode())
        line = metadata_line(part)
        assert "audio" in line and "px" not in line

    def test_handles_bad_base64(self):
        part = MediaPart(kind="image", media_type="image/png", data="!!!not base64!!!")
        # Must not raise.
        assert "image" in metadata_line(part)


class TestRenderMedia:
    def test_metadata_mode_has_no_binary(self):
        part = MediaPart(kind="image", media_type="image/png", data=_png(40, 20))
        out = render_media(part, "metadata")
        assert out == metadata_line(part)
        assert "\x1b[" not in out  # no ANSI escapes / thumbnail

    def test_inline_image_has_thumbnail(self):
        part = MediaPart(kind="image", media_type="image/png", data=_png(40, 20))
        out = render_media(part, "inline")
        assert "\x1b[38;5;" in out  # ANSI 256-color thumbnail
        # The metadata line (kind/dimensions) is always included above it.
        assert "image" in out

    def test_inline_audio_falls_back_to_metadata(self):
        part = MediaPart(kind="audio", media_type="audio/wav", data=base64.b64encode(b"x" * 100).decode())
        out = render_media(part, "inline")
        assert out == metadata_line(part)  # no thumbnail for audio

    def test_inline_without_pillow_falls_back(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _no_pil(name, *a, **k):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("no pillow")
            return real_import(name, *a, **k)

        part = MediaPart(kind="image", media_type="image/png", data=_png(40, 20))
        monkeypatch.setattr(builtins, "__import__", _no_pil)
        out = render_media(part, "inline")
        assert out == metadata_line(part)


class TestThumbnail:
    def test_thumbnail_dimensions(self):
        part = MediaPart(kind="image", media_type="image/png", data=_png(80, 40))
        thumb = ansi_thumbnail(part, max_cols=20)
        assert thumb is not None
        # 20 cols → 20*40/80/2 = 5 rows.
        assert len(thumb.splitlines()) == 5

    def test_non_image_returns_none(self):
        part = MediaPart(kind="video", media_type="video/mp4", data=base64.b64encode(b"x").decode())
        assert ansi_thumbnail(part) is None
