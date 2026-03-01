"""Abstract LLM provider protocol."""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from leuk.types import Message, StreamEvent, ToolSpec


class NoCredentialsError(RuntimeError):
    """Raised when a provider is used without the required credentials."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(
            f"No credentials configured for provider '{provider}'. "
            f"Run /auth to set up authentication."
        )


class LLMProvider(Protocol):
    """Interface that every LLM backend must satisfy."""

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Message:
        """Send messages to the LLM and return the assistant response.

        The returned Message may contain tool_calls if the model wants to invoke tools.
        """
        ...

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from the LLM, yielding events as they arrive.

        The final event is always MESSAGE_COMPLETE with the full assembled Message.
        """
        ...
        # Make this an async generator so Protocol is satisfied
        yield  # type: ignore[misc]

    async def close(self) -> None:
        """Release any held resources (HTTP connections, etc.)."""
        ...
