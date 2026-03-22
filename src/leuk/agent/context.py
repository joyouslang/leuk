"""Context window management: truncation and summarization.

Prevents context from exceeding model limits by applying a multi-strategy approach:
1. Tool result truncation -- large tool outputs are shortened
2. Sliding window -- oldest messages (except system prompt) are dropped
3. Summarization -- when dropping messages, a summary is injected
"""

from __future__ import annotations

import logging

from leuk.providers.base import LLMProvider
from leuk.types import Message, Role

logger = logging.getLogger(__name__)

# Rough token estimation: 1 token ~ 4 chars (conservative for most models)
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str | None) -> int:
    """Rough token count estimate from character length."""
    if not text:
        return 0
    return len(text) // _CHARS_PER_TOKEN


def estimate_message_tokens(msg: Message) -> int:
    """Estimate token usage for a single message."""
    total = _estimate_tokens(msg.content)
    if msg.tool_calls:
        for tc in msg.tool_calls:
            total += _estimate_tokens(tc.name)
            total += _estimate_tokens(str(tc.arguments))
    if msg.tool_result:
        total += _estimate_tokens(msg.tool_result.content)
        total += _estimate_tokens(msg.tool_result.name)
    total += 4  # role/overhead tokens
    return total


def estimate_total_tokens(messages: list[Message]) -> int:
    """Estimate total tokens across all messages."""
    return sum(estimate_message_tokens(m) for m in messages)


def truncate_tool_results(
    messages: list[Message],
    *,
    max_result_tokens: int = 8000,
) -> list[Message]:
    """Truncate individual tool results that exceed the limit.

    Returns a new list with truncated messages (originals are not mutated).
    """
    result: list[Message] = []
    max_chars = max_result_tokens * _CHARS_PER_TOKEN

    for msg in messages:
        if msg.tool_result and len(msg.tool_result.content) > max_chars:
            from leuk.types import ToolResult

            truncated_content = (
                msg.tool_result.content[:max_chars]
                + f"\n... [truncated from {len(msg.tool_result.content)} chars]"
            )
            new_result = ToolResult(
                tool_call_id=msg.tool_result.tool_call_id,
                name=msg.tool_result.name,
                content=truncated_content,
                is_error=msg.tool_result.is_error,
            )
            result.append(
                Message(
                    role=msg.role,
                    content=msg.content,
                    tool_calls=msg.tool_calls,
                    tool_result=new_result,
                    timestamp=msg.timestamp,
                    metadata=msg.metadata,
                )
            )
        else:
            result.append(msg)

    return result


def sliding_window(
    messages: list[Message],
    *,
    max_tokens: int = 100_000,
) -> list[Message]:
    """Apply a sliding window to keep messages under the token budget.

    Always preserves the system prompt (first message if role==SYSTEM)
    and the most recent messages. Drops oldest non-system messages first.

    When messages are dropped, a summary placeholder is inserted.
    """
    total = estimate_total_tokens(messages)
    if total <= max_tokens:
        return messages

    # Separate system prompt from the rest
    system_msgs: list[Message] = []
    rest: list[Message] = []
    for msg in messages:
        if msg.role == Role.SYSTEM and not rest:
            system_msgs.append(msg)
        else:
            rest.append(msg)

    # Drop from the front of `rest` until we fit
    dropped_count = 0
    while rest and estimate_total_tokens(system_msgs + rest) > max_tokens:
        # Don't drop tool results without their corresponding assistant tool_call
        rest.pop(0)
        dropped_count += 1

    if dropped_count > 0:
        logger.info("Context window: dropped %d oldest messages to fit budget", dropped_count)
        summary = Message(
            role=Role.USER,
            content=f"[SYSTEM NOTE: {dropped_count} earlier messages were trimmed from context to stay within limits.]",
        )
        return system_msgs + [summary] + rest

    return system_msgs + rest


async def summarize_and_compress(
    messages: list[Message],
    provider: LLMProvider,
    *,
    max_tokens: int = 100_000,
    summary_budget_tokens: int = 500,
) -> list[Message]:
    """Summarize older messages to compress context.

    If the context exceeds max_tokens, the oldest messages (excluding system)
    are summarized into a single message using the LLM provider, then the
    summary replaces them.

    Falls back to simple sliding_window if summarization itself fails.
    """
    total = estimate_total_tokens(messages)
    if total <= max_tokens:
        return messages

    # Split: system, old half, recent half
    system_msgs: list[Message] = []
    rest: list[Message] = []
    for msg in messages:
        if msg.role == Role.SYSTEM and not rest:
            system_msgs.append(msg)
        else:
            rest.append(msg)

    if len(rest) < 4:
        # Not enough to meaningfully summarize
        return sliding_window(messages, max_tokens=max_tokens)

    # Summarize the first half
    split = len(rest) // 2
    to_summarize = rest[:split]
    to_keep = rest[split:]

    # Build a summarization prompt
    summary_text_parts: list[str] = []
    for msg in to_summarize:
        prefix = msg.role.value.upper()
        if msg.content:
            summary_text_parts.append(f"{prefix}: {msg.content[:500]}")
        if msg.tool_result:
            summary_text_parts.append(
                f"TOOL({msg.tool_result.name}): {msg.tool_result.content[:200]}"
            )

    summarize_prompt = (
        "Summarize the following conversation excerpt in 2-3 concise sentences. "
        "Focus on key decisions, actions taken, and current state:\n\n"
        + "\n".join(summary_text_parts)
    )

    try:
        summary_response = await provider.generate(
            [
                Message(role=Role.SYSTEM, content="You are a concise summarizer."),
                Message(role=Role.USER, content=summarize_prompt),
            ],
            tools=None,
            max_tokens=summary_budget_tokens,
        )
        summary_content = summary_response.content or "Previous conversation context."
    except Exception:
        logger.warning("Summarization failed, falling back to sliding window")
        return sliding_window(messages, max_tokens=max_tokens)

    summary_msg = Message(
        role=Role.USER,
        content=f"[CONVERSATION SUMMARY: {summary_content}]",
    )

    result = system_msgs + [summary_msg] + to_keep

    # If still too large, apply sliding window as fallback
    if estimate_total_tokens(result) > max_tokens:
        return sliding_window(result, max_tokens=max_tokens)

    logger.info(
        "Context compressed: summarized %d messages, %d remain",
        len(to_summarize),
        len(result),
    )
    return result
