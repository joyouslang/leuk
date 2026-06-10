"""OpenAI-compatible provider (also used for vLLM and Ollama)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import openai

from leuk.config import LLMConfig
from leuk.providers.model_info import (
    ModelInfo,
    context_window_from_obj,
    modalities_from_obj,
)
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
        self._model_info_cache: ModelInfo | None = None

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _media_parts(parts) -> list[dict[str, Any]]:  # noqa: ANN001
        """OpenAI content parts for images/audio from MediaParts."""
        blocks: list[dict[str, Any]] = []
        for p in parts:
            if p.kind == "image":
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{p.media_type};base64,{p.data}"},
                    }
                )
            elif p.kind == "audio":
                fmt = p.media_type.split("/")[-1].split(";")[0] or "wav"
                blocks.append(
                    {"type": "input_audio", "input_audio": {"data": p.data, "format": fmt}}
                )
        return blocks

    @staticmethod
    def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
        from leuk.media import extract_media

        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == Role.TOOL and msg.tool_result is not None:
                # OpenAI tool messages are text-only; if the tool produced
                # images, send the text here and the images as a follow-up
                # user message so the model can actually see them.
                clean, media = extract_media(msg.tool_result.content)
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_result.tool_call_id,
                        "content": clean,
                    }
                )
                img_blocks = OpenAIProvider._media_parts([m for m in media if m.kind == "image"])
                if img_blocks:
                    out.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"(image output from {msg.tool_result.name})"},
                                *img_blocks,
                            ],
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

            if msg.attachments:
                blocks: list[dict[str, Any]] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                blocks.extend(OpenAIProvider._media_parts(msg.attachments))
                out.append({"role": msg.role.value, "content": blocks})
            else:
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
            # Newer OpenAI models require max_completion_tokens instead of
            # max_tokens.  Use the newer parameter name — the SDK handles
            # backwards compatibility for older models.
            kwargs["max_completion_tokens"] = max_tokens or self._config.max_tokens
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
            kwargs["max_completion_tokens"] = max_tokens or self._config.max_tokens
        temp = temperature if temperature is not None else self._config.temperature
        if temp > 0:
            kwargs["temperature"] = temp
        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        # Track tool calls being assembled: index -> {id, name, args_json}
        tc_accum: dict[int, dict[str, str]] = {}

        response = await self._client.chat.completions.create(**kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Reasoning content (DeepSeek-style `reasoning_content`, or
            # `reasoning` as used by some OpenAI-compatible gateways). Surfaced
            # whenever the backend sends it — no opt-in needed.
            reasoning = getattr(delta, "reasoning_content", None) or getattr(
                delta, "reasoning", None
            )
            if isinstance(reasoning, str) and reasoning:
                thinking_parts.append(reasoning)
                yield StreamEvent(type=StreamEventType.THINKING_DELTA, content=reasoning)

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
            thinking="".join(thinking_parts) or None,
        )
        yield StreamEvent(type=StreamEventType.MESSAGE_COMPLETE, message=final)

    async def model_info(self) -> ModelInfo:
        """Query the active model's metadata (context window + modalities).

        Reads the model object from the OpenAI-compatible ``/v1/models`` endpoint
        (OpenRouter/Zen/vLLM expose ``context_length``/``max_model_len`` and
        OpenRouter also ``architecture.input_modalities``). For Ollama we
        additionally query its native ``/api/show``, which reports a model's
        ``capabilities`` (e.g. ``vision``) and context length. Cached.
        """
        if self._model_info_cache is not None:
            return self._model_info_cache

        info = ModelInfo()
        if self._config.provider == "local":
            info = await self._ollama_model_info()  # richest source for Ollama

        if info.context_window is None or info.supports_vision is None:
            oai = await self._openai_model_info()
            info = ModelInfo(
                context_window=info.context_window or oai.context_window,
                supports_vision=(
                    info.supports_vision
                    if info.supports_vision is not None
                    else oai.supports_vision
                ),
                supports_audio=(
                    info.supports_audio
                    if info.supports_audio is not None
                    else oai.supports_audio
                ),
            )
        self._model_info_cache = info
        return info

    async def _openai_model_info(self) -> ModelInfo:
        """Read the model object from ``/v1/models`` (retrieve, then list)."""
        target = self._config.model
        obj: object | None = None
        try:
            try:
                obj = await self._client.models.retrieve(target)
            except Exception:
                models = await self._client.models.list()
                for m in getattr(models, "data", []) or []:
                    if getattr(m, "id", None) == target:
                        obj = m
                        break
        except Exception:
            return ModelInfo()
        if obj is None:
            return ModelInfo()
        vision, audio = modalities_from_obj(obj)
        return ModelInfo(
            context_window=context_window_from_obj(obj),
            supports_vision=vision,
            supports_audio=audio,
        )

    async def _ollama_model_info(self) -> ModelInfo:
        """Query Ollama's native ``/api/show`` for capabilities + context length."""
        import httpx

        base = str(self._client.base_url).rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3].rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(f"{base}/api/show", json={"model": self._config.model})
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            return ModelInfo()

        caps = data.get("capabilities")
        vision = audio = None
        if isinstance(caps, list):
            low = [str(c).lower() for c in caps]
            vision = "vision" in low
            audio = any(c in low for c in ("audio", "speech"))
        context = None
        mi = data.get("model_info")
        if isinstance(mi, dict):
            for key, val in mi.items():
                if key.endswith(".context_length") and isinstance(val, (int, float)) and val > 0:
                    context = int(val)
                    break
        return ModelInfo(context_window=context, supports_vision=vision, supports_audio=audio)

    async def close(self) -> None:
        await self._client.close()
