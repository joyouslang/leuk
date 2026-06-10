"""Tests for graceful provider-error handling: clean messages + image shrinking."""

from __future__ import annotations

import base64
import io

import pytest

from leuk.agent.session import _error_text
from leuk.media import _MAX_IMAGE_BYTES, load_media_file, shrink_image


class TestErrorText:
    def test_extracts_api_error_message(self):
        # Mirrors anthropic.BadRequestError: .body carries the structured error.
        class FakeAPIError(Exception):
            body = {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "messages.0.content.1.image.source.base64: "
                    "image exceeds 10 MB maximum: 11552136 bytes > 10485760 bytes",
                },
                "request_id": "req_x",
            }

        exc = FakeAPIError("Error code: 400 - {'type': 'error', 'error': {...}}")
        out = _error_text(exc)
        assert out.startswith("FakeAPIError: ")
        assert "image exceeds 10 MB maximum" in out
        assert "request_id" not in out  # no raw JSON wall

    def test_flat_message_body(self):
        class FlatError(Exception):
            body = {"message": "quota exceeded"}

        assert _error_text(FlatError("raw")) == "FlatError: quota exceeded"

    def test_plain_exception_falls_back(self):
        assert _error_text(ValueError("boom")) == "ValueError: boom"


def _png_bytes(w: int, h: int, *, alpha: bool = False, noisy: bool = False) -> bytes:
    pytest.importorskip("PIL")
    import random

    from PIL import Image

    mode = "RGBA" if alpha else "RGB"
    img = Image.new(mode, (w, h))
    if noisy:  # random pixels defeat PNG compression → large file
        rnd = random.Random(42)
        img.putdata(
            [
                tuple(rnd.randrange(256) for _ in range(4 if alpha else 3))
                for _ in range(w * h)
            ]
        )
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class TestShrinkImage:
    def test_small_image_untouched(self):
        raw = _png_bytes(50, 50)
        out, mt = shrink_image(raw, "image/png")
        assert out is raw and mt == "image/png"

    def test_oversized_image_downscaled(self):
        raw = _png_bytes(2400, 2400, noisy=True)
        assert len(raw) > _MAX_IMAGE_BYTES  # sanity: actually oversized
        out, mt = shrink_image(raw, "image/png")
        assert len(out) < _MAX_IMAGE_BYTES
        assert mt == "image/jpeg"  # no alpha → photo path
        from PIL import Image

        img = Image.open(io.BytesIO(out))
        assert max(img.size) <= 1568

    def test_alpha_preserved_when_possible(self):
        raw = _png_bytes(2400, 2400, alpha=True, noisy=True)
        out, mt = shrink_image(raw, "image/png")
        assert len(out) < len(raw)
        # Either stayed PNG (alpha kept) or fell back to JPEG if still too big.
        assert mt in ("image/png", "image/jpeg")

    def test_load_media_file_shrinks(self, tmp_path):
        raw = _png_bytes(2400, 2400, noisy=True)
        f = tmp_path / "huge.png"
        f.write_bytes(raw)
        part = load_media_file(str(f))
        assert part.kind == "image"
        assert len(base64.b64decode(part.data)) < _MAX_IMAGE_BYTES
