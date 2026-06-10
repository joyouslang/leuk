"""Anthropic Claude provider."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import anthropic

from leuk.billing import CC_USER_AGENT, billing_header
from leuk.config import LLMConfig
from leuk.providers.model_info import ModelInfo
from leuk.types import Message, Role, StreamEvent, StreamEventType, ToolCall, ToolSpec

log = logging.getLogger(__name__)


class AnthropicProvider:
    """LLM provider backed by the Anthropic API."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = self._make_client(config)

    async def model_info(self) -> ModelInfo:
        # Anthropic's API doesn't expose context window or modality, so both are
        # "unknown": vision falls back to sending natively (Claude 3+ accept it;
        # older models surface an API error), and the context window comes from
        # the user's config override. No name-based guessing.
        return ModelInfo()

    # Beta flags required by the Anthropic API.
    _OAUTH_BETA = "oauth-2025-04-20"
    _BETAS = f"files-api-2025-04-14,{_OAUTH_BETA}"

    @staticmethod
    def _make_client(config: LLMConfig) -> anthropic.AsyncAnthropic:
        default_headers: dict[str, str] = {"User-Agent": CC_USER_AGENT}
        kwargs: dict[str, Any] = {}
        if config.anthropic_api_key:
            kwargs["api_key"] = config.anthropic_api_key
        elif config.anthropic_auth_token:
            kwargs["auth_token"] = config.anthropic_auth_token
            default_headers["anthropic-beta"] = AnthropicProvider._BETAS
        kwargs["default_headers"] = default_headers
        return anthropic.AsyncAnthropic(**kwargs)

    def _try_refresh_token(self) -> bool:
        """Attempt to refresh the OAuth credentials and rebuild the client.

        Returns True if the token was refreshed successfully.
        """
        from leuk.cli.auth import refresh_anthropic_token

        new_token = refresh_anthropic_token()
        if not new_token:
            return False

        log.info("Anthropic OAuth credentials refreshed")
        self._config.anthropic_auth_token = new_token
        self._client = self._make_client(self._config)
        return True

    # ------------------------------------------------------------------
    # Message format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_anthropic_messages(
        messages: list[Message],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Convert internal messages to Anthropic's format.

        Returns (system_blocks, messages_list).  ``system_blocks`` is a list
        of ``{"type": "text", "text": ...}`` dicts (the billing header is
        always the first block, followed by the user-provided system prompt).
        """
        system_text = ""
        out: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_text = msg.content or ""
                continue

            if msg.role == Role.TOOL and msg.tool_result is not None:
                from leuk.media import extract_media

                clean, media = extract_media(msg.tool_result.content)
                # Anthropic tool_result content may be a list of text + image
                # blocks, so screenshots from tools are seen natively by Claude.
                tr_content: Any = clean
                images = [m for m in media if m.kind == "image"]
                if images:
                    tr_content = [{"type": "text", "text": clean}] + [
                        AnthropicProvider._image_block(m) for m in images
                    ]
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_result.tool_call_id,
                                "content": tr_content,
                                "is_error": msg.tool_result.is_error,
                            }
                        ],
                    }
                )
                continue

            if msg.role == Role.ASSISTANT and msg.tool_calls:
                content_blocks: list[dict[str, Any]] = []
                # With extended thinking enabled, the API requires the turn's
                # thinking blocks (with signatures) to be replayed ahead of the
                # tool_use blocks on the next request.
                for tb in msg.metadata.get("_thinking_blocks") or []:
                    if tb.get("thinking") and tb.get("signature"):
                        content_blocks.append(
                            {
                                "type": "thinking",
                                "thinking": tb["thinking"],
                                "signature": tb["signature"],
                            }
                        )
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                out.append({"role": "assistant", "content": content_blocks})
                continue

            # User (or other) message — attach images natively if present.
            if msg.attachments:
                images = [a for a in msg.attachments if a.kind == "image"]
                dropped = [a for a in msg.attachments if a.kind != "image"]
                blocks: list[dict[str, Any]] = []
                text = msg.content or ""
                if dropped:
                    text += (
                        f"\n[{len(dropped)} audio attachment(s) omitted — "
                        "Claude does not support audio input]"
                    )
                if text:
                    blocks.append({"type": "text", "text": text})
                blocks.extend(AnthropicProvider._image_block(m) for m in images)
                out.append({"role": msg.role.value, "content": blocks or ""})
            else:
                out.append({"role": msg.role.value, "content": msg.content or ""})

        # Build system prompt blocks: billing header first, then user system prompt.
        bh = billing_header(messages)
        system_blocks: list[dict[str, Any]] = []
        if bh:
            system_blocks.append({"type": "text", "text": bh})
        if system_text:
            system_blocks.append({"type": "text", "text": system_text})

        return system_blocks, out

    @staticmethod
    def _image_block(part: "Any") -> dict[str, Any]:
        """Anthropic base64 image content block from a MediaPart."""
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": part.media_type,
                "data": part.data,
            },
        }

    def _thinking_param(self) -> dict[str, Any] | None:
        """The request's thinking parameter, when the user opted in."""
        if not self._config.thinking:
            return None
        return {"type": "enabled", "budget_tokens": self._config.thinking_budget}

    @staticmethod
    def _to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Message:
        system_blocks, msgs = self._to_anthropic_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": max_tokens or self._config.max_tokens,
            "messages": msgs,
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        if temperature is not None:
            kwargs["temperature"] = temperature
        elif self._config.temperature > 0:
            kwargs["temperature"] = self._config.temperature
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(tools)
        if thinking := self._thinking_param():
            kwargs["thinking"] = thinking

        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError:
            if not self._try_refresh_token():
                raise
            response = await self._client.messages.create(**kwargs)

        # Parse response into our Message type
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        thinking_blocks: list[dict[str, str]] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_blocks.append(
                    {"thinking": block.thinking, "signature": block.signature}
                )
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )

        meta: dict[str, Any] = {}
        if thinking_blocks:
            meta["_thinking_blocks"] = thinking_blocks
        return Message(
            role=Role.ASSISTANT,
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
            metadata=meta,
            thinking="\n".join(tb["thinking"] for tb in thinking_blocks) or None,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        system_blocks, msgs = self._to_anthropic_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": max_tokens or self._config.max_tokens,
            "messages": msgs,
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        if temperature is not None:
            kwargs["temperature"] = temperature
        elif self._config.temperature > 0:
            kwargs["temperature"] = self._config.temperature
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(tools)
        if thinking := self._thinking_param():
            kwargs["thinking"] = thinking

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        # Track in-progress tool calls by index
        current_tc: dict[str, Any] = {}
        # Completed thinking blocks (text + signature, for tool-use replay)
        # and the one currently streaming.
        thinking_blocks: list[dict[str, str]] = []
        current_thinking: dict[str, str] | None = None

        try:
            stream_cm = self._client.messages.stream(**kwargs)
            stream = await stream_cm.__aenter__()
        except anthropic.AuthenticationError:
            if not self._try_refresh_token():
                raise
            stream_cm = self._client.messages.stream(**kwargs)
            stream = await stream_cm.__aenter__()

        try:
            async for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "text":
                        pass  # Text deltas come separately
                    elif block.type == "thinking":
                        current_thinking = {"thinking": "", "signature": ""}
                    elif block.type == "tool_use":
                        current_tc = {"id": block.id, "name": block.name, "args_json": ""}
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_START,
                            tool_call=ToolCall(id=block.id, name=block.name, arguments={}),
                        )
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        text_parts.append(delta.text)
                        yield StreamEvent(type=StreamEventType.TEXT_DELTA, content=delta.text)
                    elif delta.type == "thinking_delta":
                        if current_thinking is not None:
                            current_thinking["thinking"] += delta.thinking
                        yield StreamEvent(
                            type=StreamEventType.THINKING_DELTA, content=delta.thinking
                        )
                    elif delta.type == "signature_delta":
                        if current_thinking is not None:
                            current_thinking["signature"] += delta.signature
                    elif delta.type == "input_json_delta":
                        if current_tc:
                            current_tc["args_json"] += delta.partial_json
                        yield StreamEvent(type=StreamEventType.TOOL_CALL_DELTA, content=delta.partial_json)
                elif event.type == "content_block_stop":
                    if current_thinking is not None:
                        thinking_blocks.append(current_thinking)
                        current_thinking = None
                    elif current_tc:
                        args = json.loads(current_tc["args_json"]) if current_tc["args_json"] else {}
                        tc = ToolCall(id=current_tc["id"], name=current_tc["name"], arguments=args)
                        tool_calls.append(tc)
                        yield StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=tc)
                        current_tc = {}
        finally:
            await stream_cm.__aexit__(None, None, None)

        meta: dict[str, Any] = {}
        if thinking_blocks:
            meta["_thinking_blocks"] = thinking_blocks
        final = Message(
            role=Role.ASSISTANT,
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
            metadata=meta,
            thinking="\n".join(tb["thinking"] for tb in thinking_blocks) or None,
        )
        yield StreamEvent(type=StreamEventType.MESSAGE_COMPLETE, message=final)

    async def close(self) -> None:
        await self._client.close()
