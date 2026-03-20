"""Anthropic Claude provider."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import anthropic

from leuk.billing import CC_USER_AGENT, billing_header
from leuk.config import LLMConfig
from leuk.types import Message, Role, StreamEvent, StreamEventType, ToolCall, ToolResult, ToolSpec

log = logging.getLogger(__name__)


class AnthropicProvider:
    """LLM provider backed by the Anthropic API."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = self._make_client(config)

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
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_result.tool_call_id,
                                "content": msg.tool_result.content,
                                "is_error": msg.tool_result.is_error,
                            }
                        ],
                    }
                )
                continue

            if msg.role == Role.ASSISTANT and msg.tool_calls:
                content_blocks: list[dict[str, Any]] = []
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

        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError:
            if not self._try_refresh_token():
                raise
            response = await self._client.messages.create(**kwargs)

        # Parse response into our Message type
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )

        return Message(
            role=Role.ASSISTANT,
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
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

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        # Track in-progress tool calls by index
        current_tc: dict[str, Any] = {}

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
                    elif delta.type == "input_json_delta":
                        if current_tc:
                            current_tc["args_json"] += delta.partial_json
                        yield StreamEvent(type=StreamEventType.TOOL_CALL_DELTA, content=delta.partial_json)
                elif event.type == "content_block_stop":
                    if current_tc:
                        args = json.loads(current_tc["args_json"]) if current_tc["args_json"] else {}
                        tc = ToolCall(id=current_tc["id"], name=current_tc["name"], arguments=args)
                        tool_calls.append(tc)
                        yield StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=tc)
                        current_tc = {}
        finally:
            await stream_cm.__aexit__(None, None, None)

        final = Message(
            role=Role.ASSISTANT,
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
        )
        yield StreamEvent(type=StreamEventType.MESSAGE_COMPLETE, message=final)

    async def close(self) -> None:
        await self._client.close()
