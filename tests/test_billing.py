"""Tests for the billing header generation."""

from __future__ import annotations

import hashlib

from leuk.billing import (
    CC_VERSION,
    _BILLING_SALT,
    _compute_version_hash,
    _first_user_message_text,
    _sample_js_char,
    billing_header,
)
from leuk.types import Message, Role


class TestSampleJsChar:
    def test_ascii(self):
        assert _sample_js_char("hello world", 0) == "h"
        assert _sample_js_char("hello world", 4) == "o"

    def test_out_of_range(self):
        assert _sample_js_char("hi", 10) == "0"

    def test_empty_string(self):
        assert _sample_js_char("", 0) == "0"

    def test_bmp_unicode(self):
        # BMP characters: each is 1 UTF-16 code unit, same as Python index
        assert _sample_js_char("café", 3) == "é"

    def test_astral_plane(self):
        # 💻 (U+1F4BB) is 2 UTF-16 code units.  Index 0 should yield the
        # high surrogate decoded as a replacement or the first code unit char.
        text = "💻abc"
        # In JS, text[0] returns the high surrogate of the emoji.
        # Index 2 (after the emoji's 2 code units) should be 'a'.
        assert _sample_js_char(text, 2) == "a"


class TestComputeVersionHash:
    def test_known_hash(self):
        """Verify the hash matches the reference JS implementation."""
        text = "Hello, Claude!"
        # JS indices: text[4]='o', text[7]='C', text[20]='0' (out of range)
        sampled = "oC0"
        expected = hashlib.sha256(
            f"{_BILLING_SALT}{sampled}{CC_VERSION}".encode()
        ).hexdigest()[:3]
        assert _compute_version_hash(text, CC_VERSION) == expected

    def test_empty_message(self):
        sampled = "000"
        expected = hashlib.sha256(
            f"{_BILLING_SALT}{sampled}{CC_VERSION}".encode()
        ).hexdigest()[:3]
        assert _compute_version_hash("", CC_VERSION) == expected

    def test_short_message(self):
        text = "Hi"
        # indices 4, 7, 20 all out of range -> "000"
        sampled = "000"
        expected = hashlib.sha256(
            f"{_BILLING_SALT}{sampled}{CC_VERSION}".encode()
        ).hexdigest()[:3]
        assert _compute_version_hash(text, CC_VERSION) == expected


class TestFirstUserMessageText:
    def test_finds_first_user(self):
        msgs = [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="hello"),
            Message(role=Role.USER, content="world"),
        ]
        assert _first_user_message_text(msgs) == "hello"

    def test_no_user_message(self):
        msgs = [Message(role=Role.SYSTEM, content="sys")]
        assert _first_user_message_text(msgs) == ""


class TestBillingHeader:
    def test_format(self):
        msgs = [Message(role=Role.USER, content="Hello, Claude!")]
        header = billing_header(msgs)
        assert header.startswith("x-anthropic-billing-header: cc_version=")
        assert f"cc_version={CC_VERSION}." in header
        assert "cc_entrypoint=" in header
        assert "cch=00000;" in header

    def test_empty_messages(self):
        header = billing_header([])
        assert "cc_version=" in header
        # Hash should still be present (based on empty text)
        assert f"{CC_VERSION}." in header

    def test_version_hash_changes_with_content(self):
        h1 = billing_header([Message(role=Role.USER, content="aaaa")])
        h2 = billing_header([Message(role=Role.USER, content="zzzzzzzzzzzzzzzzzzzzzzzz")])
        # Different messages should produce different hashes
        assert h1 != h2
