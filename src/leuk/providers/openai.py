"""OpenAI-compatible provider (also used for vLLM and Ollama)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import openai

from leuk.config import LLMConfig
from leuk.types import Message, Role, StreamEvent, StreamEventType, ToolCall, ToolSpec


class OpenAIProvider:
    """LLM provider backed by the OpenAI API (or any compatible endpoint)."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._config = config
        self._client = openai.AsyncOpenAI(
            api_key=api_key or config.openai_api_key or None,
            base_url=base_url,
        )

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == Role.TOOL and msg.tool_result is not None:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_result.tool_call_id,
                        "content": msg.tool_result.content,
                    }
                )
                continue

            if msg.role == Role.ASSISTANT and msg.tool_calls:
                d: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                out.append(d)
                continue

            out.append({"role": msg.role.value, "content": msg.content or ""})
        return out

    @staticmethod
    def _to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
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
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": self._to_openai_messages(messages),
        }
        if max_tokens or self._config.max_tokens:
            kwargs["max_tokens"] = max_tokens or self._config.max_tokens
        temp = temperature if temperature is not None else self._config.temperature
        if temp > 0:
            kwargs["temperature"] = temp
        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] | None = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in msg.tool_calls
            ]

        return Message(
            role=Role.ASSISTANT,
            content=msg.content,
            tool_calls=tool_calls,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": self._to_openai_messages(messages),
            "stream": True,
        }
        if max_tokens or self._config.max_tokens:
            kwargs["max_tokens"] = max_tokens or self._config.max_tokens
        temp = temperature if temperature is not None else self._config.temperature
        if temp > 0:
            kwargs["temperature"] = temp
        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)

        text_parts: list[str] = []
        # Track tool calls being assembled: index -> {id, name, args_json}
        tc_accum: dict[int, dict[str, str]] = {}

        response = await self._client.chat.completions.create(**kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Text content
            if delta.content:
                text_parts.append(delta.content)
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, content=delta.content)

            # Tool calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_accum:
                        tc_accum[idx] = {
                            "id": tc_delta.id or "",
                            "name": tc_delta.function.name or "" if tc_delta.function else "",
                            "args_json": "",
                        }
                        if tc_accum[idx]["id"] and tc_accum[idx]["name"]:
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_START,
                                tool_call=ToolCall(
                                    id=tc_accum[idx]["id"],
                                    name=tc_accum[idx]["name"],
                                    arguments={},
                                ),
                            )
                    else:
                        # Update id/name if they arrive in subsequent chunks
                        if tc_delta.id:
                            tc_accum[idx]["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            tc_accum[idx]["name"] = tc_delta.function.name

                    if tc_delta.function and tc_delta.function.arguments:
                        tc_accum[idx]["args_json"] += tc_delta.function.arguments
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_DELTA,
                            content=tc_delta.function.arguments,
                        )

        # Finalize tool calls
        tool_calls: list[ToolCall] = []
        for _idx, acc in sorted(tc_accum.items()):
            args = json.loads(acc["args_json"]) if acc["args_json"] else {}
            tc = ToolCall(id=acc["id"], name=acc["name"], arguments=args)
            tool_calls.append(tc)
            yield StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=tc)

        final = Message(
            role=Role.ASSISTANT,
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
        )
        yield StreamEvent(type=StreamEventType.MESSAGE_COMPLETE, message=final)

    async def close(self) -> None:
        await self._client.close()
