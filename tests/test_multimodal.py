"""Tests for multimodal support: media parsing, provider conversion, persistence."""

from __future__ import annotations

import base64

import pytest

from leuk.media import extract_media, load_media_file, strip_media
from leuk.providers.anthropic import AnthropicProvider
from leuk.providers.model_info import context_window_from_obj, modalities_from_obj
from leuk.providers.openai import OpenAIProvider
from leuk.types import MediaPart, Message, Role, ToolResult


class TestModelInfoParsing:
    """Parsing of queried model metadata — no name-based guessing."""

    def test_context_window_from_fields(self):
        assert context_window_from_obj({"context_length": 200000}) == 200000
        assert context_window_from_obj({"max_model_len": 32768}) == 32768  # vLLM
        assert context_window_from_obj({"id": "x"}) is None  # not reported

    def test_modalities_input_list(self):
        obj = {"architecture": {"input_modalities": ["text", "image"]}}
        assert modalities_from_obj(obj) == (True, False)
        obj2 = {"architecture": {"input_modalities": ["text", "image", "audio"]}}
        assert modalities_from_obj(obj2) == (True, True)

    def test_modalities_modality_string(self):
        obj = {"architecture": {"modality": "text+image->text"}}
        assert modalities_from_obj(obj) == (True, False)

    def test_modalities_unknown_when_absent(self):
        assert modalities_from_obj({"id": "x"}) == (None, None)


class TestOllamaModelInfo:
    @pytest.mark.asyncio
    async def test_queries_api_show_for_vision_and_context(self, monkeypatch):
        """Ollama capabilities are read from /api/show, not guessed from the name."""
        from leuk.config import LLMConfig

        provider = OpenAIProvider(
            LLMConfig(provider="local", model="llava"),
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "capabilities": ["completion", "vision", "tools"],
                    "model_info": {"llama.context_length": 131072},
                }

        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json):
                assert url.endswith("/api/show")
                return _Resp()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", _Client)
        info = await provider._ollama_model_info()
        assert info.supports_vision is True
        assert info.context_window == 131072


class TestStripMedia:
    def test_strips_tool_result_media_and_leaves_note(self):
        msg = Message(
            role=Role.TOOL,
            tool_result=ToolResult("t1", "input_control", "look [screenshot:image/png;base64,AAAA] ok"),
        )
        out = strip_media([msg], note="no vision")
        content = out[0].tool_result.content
        assert "AAAA" not in content  # base64 gone
        assert "omitted" in content and "no vision" in content

    def test_strips_user_attachments(self):
        msg = Message(
            role=Role.USER, content="see this", attachments=[MediaPart("image", "image/png", "BBBB")]
        )
        out = strip_media([msg], note="no vision")
        assert out[0].attachments is None
        assert "omitted" in (out[0].content or "")

    def test_leaves_plain_messages_untouched(self):
        msg = Message(role=Role.ASSISTANT, content="hello")
        out = strip_media([msg], note="x")
        assert out[0] is msg  # unchanged identity


class TestExtractMedia:
    def test_screenshot_tag(self):
        clean, parts = extract_media("before [screenshot:image/png;base64,AAAA] after")
        assert parts and parts[0].kind == "image" and parts[0].media_type == "image/png"
        assert parts[0].data == "AAAA"
        assert "[screenshot attached]" in clean and "AAAA" not in clean

    def test_audio_tag(self):
        _, parts = extract_media("[audio:audio/wav;base64,BBBB]")
        assert parts[0].kind == "audio"

    def test_multiple_and_whitespace(self):
        clean, parts = extract_media(
            "[image:image/jpeg;base64,AA AA] x [screenshot:image/png;base64,BB]"
        )
        assert len(parts) == 2
        assert parts[0].data == "AAAA"  # whitespace stripped

    def test_video_tag(self):
        _, parts = extract_media("[video:video/mp4;base64,CCCC]")
        assert parts and parts[0].kind == "video" and parts[0].media_type == "video/mp4"

    def test_no_tags(self):
        clean, parts = extract_media("just text")
        assert clean == "just text" and parts == []

    def test_media_to_tag_roundtrip(self):
        from leuk.media import media_to_tag

        original = "[image:image/png;base64,ZZZZ][video:video/mp4;base64,YYYY]"
        _clean, parts = extract_media(original)
        rebuilt = "".join(media_to_tag(p) for p in parts)
        _c2, parts2 = extract_media(rebuilt)
        assert [(p.kind, p.media_type, p.data) for p in parts2] == [
            ("image", "image/png", "ZZZZ"),
            ("video", "video/mp4", "YYYY"),
        ]


class TestLoadMediaFile:
    def test_load_png(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(b"\x89PNGdata")
        part = load_media_file(str(p))
        assert part.kind == "image" and part.media_type == "image/png"
        assert base64.b64decode(part.data) == b"\x89PNGdata"

    def test_load_wav(self, tmp_path):
        p = tmp_path / "x.wav"
        p.write_bytes(b"RIFFdata")
        assert load_media_file(str(p)).kind == "audio"

    def test_load_mp4_video(self, tmp_path):
        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00\x00\x00\x18ftypmp4")
        part = load_media_file(str(p))
        assert part.kind == "video" and part.media_type == "video/mp4"

    def test_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_media_file(str(tmp_path / "nope.png"))

    def test_unsupported(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("hi")
        with pytest.raises(ValueError):
            load_media_file(str(p))


class TestAnthropicConversion:
    def test_user_image(self):
        msgs = [Message(role=Role.USER, content="what?", attachments=[MediaPart("image", "image/png", "ABC")])]
        _, out = AnthropicProvider._to_anthropic_messages(msgs)
        blocks = out[0]["content"]
        assert blocks[0] == {"type": "text", "text": "what?"}
        assert blocks[1]["type"] == "image" and blocks[1]["source"]["data"] == "ABC"

    def test_tool_screenshot_becomes_image_block(self):
        msgs = [Message(role=Role.TOOL, tool_result=ToolResult(
            tool_call_id="1", name="browser", content="done [screenshot:image/png;base64,XYZ]"))]
        _, out = AnthropicProvider._to_anthropic_messages(msgs)
        tr = out[0]["content"][0]["content"]
        assert isinstance(tr, list)
        assert any(b.get("type") == "image" for b in tr)

    def test_audio_dropped_with_note(self):
        msgs = [Message(role=Role.USER, content="hear", attachments=[MediaPart("audio", "audio/wav", "AA")])]
        _, out = AnthropicProvider._to_anthropic_messages(msgs)
        text_block = out[0]["content"][0]
        assert "audio attachment" in text_block["text"]


class TestOpenAIConversion:
    def test_user_image_data_uri(self):
        msgs = [Message(role=Role.USER, content="hi", attachments=[MediaPart("image", "image/jpeg", "ABC")])]
        out = OpenAIProvider._to_openai_messages(msgs)
        block = out[0]["content"][1]
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == "data:image/jpeg;base64,ABC"

    def test_tool_screenshot_followup_user(self):
        msgs = [Message(role=Role.TOOL, tool_result=ToolResult(
            tool_call_id="1", name="input_control", content="moved [screenshot:image/png;base64,XYZ]"))]
        out = OpenAIProvider._to_openai_messages(msgs)
        assert out[0]["role"] == "tool" and "XYZ" not in out[0]["content"]
        assert out[1]["role"] == "user"
        assert out[1]["content"][1]["type"] == "image_url"

    def test_audio_input_block(self):
        msgs = [Message(role=Role.USER, content="", attachments=[MediaPart("audio", "audio/wav", "AA")])]
        out = OpenAIProvider._to_openai_messages(msgs)
        blocks = out[0]["content"]
        assert blocks[0]["type"] == "input_audio"
        assert blocks[0]["input_audio"]["format"] == "wav"


class TestPersistence:
    @pytest.mark.asyncio
    async def test_attachment_round_trip(self, tmp_path):
        from leuk.config import SQLiteConfig
        from leuk.persistence.sqlite import SQLiteStore

        store = SQLiteStore(SQLiteConfig(path=str(tmp_path / "t.db")))
        await store.init()
        from leuk.types import Session

        sess = Session()
        await store.create_session(sess)
        msg = Message(
            role=Role.USER,
            content="see this",
            attachments=[MediaPart("image", "image/png", "ZZZ")],
        )
        await store.append_message(sess.id, msg)
        loaded = await store.get_messages(sess.id)
        assert loaded[0].attachments is not None
        assert loaded[0].attachments[0].data == "ZZZ"
        assert "_attachments" not in loaded[0].metadata
        await store.close()
