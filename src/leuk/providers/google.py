"""Google Gemini provider."""

from __future__ import annotations

from typing import AsyncIterator

from google import genai
from google.genai import types as gtypes

from leuk.config import LLMConfig
from leuk.types import Message, Role, StreamEvent, StreamEventType, ToolCall, ToolSpec


class GoogleProvider:
    """LLM provider backed by the Google GenAI (Gemini) API."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = genai.Client(api_key=config.google_api_key or None)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gemini_contents(messages: list[Message]) -> tuple[str | None, list[gtypes.Content]]:
        """Convert to Gemini content format.

        Returns (system_instruction, contents).
        """
        system: str | None = None
        contents: list[gtypes.Content] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system = msg.content
                continue

            if msg.role == Role.USER:
                contents.append(
                    gtypes.Content(role="user", parts=[gtypes.Part(text=msg.content or "")])
                )
            elif msg.role == Role.ASSISTANT:
                parts: list[gtypes.Part] = []
                if msg.content:
                    parts.append(gtypes.Part(text=msg.content))
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        fc_kwargs: dict = {
                            "function_call": gtypes.FunctionCall(
                                name=tc.name, args=tc.arguments
                            ),
                        }
                        # Replay thought_signature if present (required by
                        # Gemini thinking models for tool call round-trips).
                        if tc.metadata.get("thought_signature"):
                            fc_kwargs["thought_signature"] = tc.metadata[
                                "thought_signature"
                            ]
                        if tc.metadata.get("thought"):
                            fc_kwargs["thought"] = tc.metadata["thought"]
                        parts.append(gtypes.Part(**fc_kwargs))
                contents.append(gtypes.Content(role="model", parts=parts))
            elif msg.role == Role.TOOL and msg.tool_result is not None:
                fr_kwargs: dict = {
                    "function_response": gtypes.FunctionResponse(
                        name=msg.tool_result.name,
                        response={"result": msg.tool_result.content},
                    ),
                }
                # Include thought_signature from the originating tool call
                # so Gemini can correlate the response.
                if msg.tool_result.metadata.get("thought_signature"):
                    fr_kwargs["thought_signature"] = msg.tool_result.metadata[
                        "thought_signature"
                    ]
                contents.append(
                    gtypes.Content(
                        role="user",
                        parts=[gtypes.Part(**fr_kwargs)],
                    )
                )

        return system, contents

    @staticmethod
    def _to_gemini_tools(tools: list[ToolSpec]) -> list[gtypes.Tool]:
        declarations = []
        for t in tools:
            declarations.append(
                gtypes.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                )
            )
        return [gtypes.Tool(function_declarations=declarations)]

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
        system_instruction, contents = self._to_gemini_contents(messages)

        config = gtypes.GenerateContentConfig(
            temperature=temperature if temperature is not None else self._config.temperature,
            max_output_tokens=max_tokens or self._config.max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction
        if tools:
            config.tools = self._to_gemini_tools(tools)

        response = await self._client.aio.models.generate_content(
            model=self._config.model,
            contents=contents,
            config=config,
        )

        # Parse response
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    text_parts.append(part.text)
                elif part.function_call:
                    fc = part.function_call
                    meta: dict = {}
                    if getattr(part, "thought_signature", None):
                        meta["thought_signature"] = part.thought_signature
                    if getattr(part, "thought", None):
                        meta["thought"] = part.thought
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{fc.name}_{len(tool_calls)}",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                            metadata=meta,
                        )
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
        system_instruction, contents = self._to_gemini_contents(messages)

        config = gtypes.GenerateContentConfig(
            temperature=temperature if temperature is not None else self._config.temperature,
            max_output_tokens=max_tokens or self._config.max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction
        if tools:
            config.tools = self._to_gemini_tools(tools)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        stream = await self._client.aio.models.generate_content_stream(
            model=self._config.model,
            contents=contents,
            config=config,
        )
        async for chunk in stream:
            if not chunk.candidates:
                continue
            for part in chunk.candidates[0].content.parts:
                if part.text:
                    text_parts.append(part.text)
                    yield StreamEvent(type=StreamEventType.TEXT_DELTA, content=part.text)
                elif part.function_call:
                    fc = part.function_call
                    meta = {}
                    if getattr(part, "thought_signature", None):
                        meta["thought_signature"] = part.thought_signature
                    if getattr(part, "thought", None):
                        meta["thought"] = part.thought
                    tc = ToolCall(
                        id=f"call_{fc.name}_{len(tool_calls)}",
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                        metadata=meta,
                    )
                    tool_calls.append(tc)
                    yield StreamEvent(type=StreamEventType.TOOL_CALL_START, tool_call=tc)
                    yield StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=tc)

        final = Message(
            role=Role.ASSISTANT,
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
        )
        yield StreamEvent(type=StreamEventType.MESSAGE_COMPLETE, message=final)

    async def close(self) -> None:
        pass  # google-genai client has no explicit close
