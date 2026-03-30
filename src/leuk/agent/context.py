"""Context window management: tiered compaction pipeline.

Prevents context from exceeding model limits via a multi-stage pipeline
(applied in order):

1. **Truncate** — oversized individual tool results are shortened in place.
2. **Mask observations** — when total tokens exceed a threshold (60% of
   budget), older tool-result bodies are replaced with one-line placeholders
   while the agent's reasoning and actions are preserved intact.
3. **Structured summarize** — when still over budget, the oldest messages
   are archived to disk, then summarized by the LLM into a persistent
   structured summary that merges incrementally.
4. **Emergency drop** — if summarization fails or the result is still too
   large, the oldest non-system messages are dropped outright.

Research basis (2025–2026):
- JetBrains Research found observation masking outperformed LLM
  summarization in 4/5 settings while being 52% cheaper.
- Factory.ai showed structured (sectioned) summaries scored 3.70/5
  vs 3.44 (Anthropic) and 3.35 (OpenAI) for agent context compression.
"""

from __future__ import annotations

import logging

from leuk.providers.base import LLMProvider
from leuk.types import Message, Role, ToolResult

logger = logging.getLogger(__name__)

# Rough token estimation: 1 token ≈ 4 chars (conservative for most models).
_CHARS_PER_TOKEN = 4

# Observation masking kicks in when context exceeds this fraction of the budget.
_MASK_THRESHOLD = 0.6


# ------------------------------------------------------------------
# Token estimation
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Stage 1: truncate oversized tool results
# ------------------------------------------------------------------


def truncate_tool_results(
    messages: list[Message],
    *,
    max_result_tokens: int = 8000,
) -> list[Message]:
    """Truncate individual tool results that exceed *max_result_tokens*.

    Returns a new list with truncated messages (originals are not mutated).
    """
    result: list[Message] = []
    max_chars = max_result_tokens * _CHARS_PER_TOKEN

    for msg in messages:
        if msg.tool_result and len(msg.tool_result.content) > max_chars:
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


# ------------------------------------------------------------------
# Stage 2: observation masking
# ------------------------------------------------------------------


def mask_observations(
    messages: list[Message],
    *,
    max_tokens: int = 100_000,
    threshold: float = _MASK_THRESHOLD,
) -> list[Message]:
    """Replace older tool-result bodies with one-line placeholders.

    Masking is adaptive: it only activates when the total token estimate
    exceeds *threshold* × *max_tokens*.  Tool results are masked from
    oldest to newest until the estimate fits or all eligible results have
    been processed.

    The agent's own reasoning (assistant messages, tool call names and
    arguments) is never touched — only ``ToolResult.content`` bodies.
    """
    total = estimate_total_tokens(messages)
    target = int(max_tokens * threshold)
    if total <= target:
        return messages

    # Find indices of tool-result messages, oldest first.
    tool_indices = [
        i for i, m in enumerate(messages)
        if m.tool_result and not m.tool_result.is_error
    ]

    result = list(messages)
    for idx in tool_indices:
        if estimate_total_tokens(result) <= target:
            break
        msg = result[idx]
        tr = msg.tool_result
        assert tr is not None  # guaranteed by filter above
        original_len = len(tr.content)
        if original_len <= 200:
            continue  # too small to bother masking

        preview = tr.content[:100].replace("\n", " ")
        placeholder = f"[{tr.name}: {preview}… ({original_len} chars masked)]"
        masked_result = ToolResult(
            tool_call_id=tr.tool_call_id,
            name=tr.name,
            content=placeholder,
            is_error=tr.is_error,
        )
        result[idx] = Message(
            role=msg.role,
            content=msg.content,
            tool_calls=msg.tool_calls,
            tool_result=masked_result,
            timestamp=msg.timestamp,
            metadata=msg.metadata,
        )

    masked_count = sum(
        1 for orig, new in zip(messages, result)
        if orig is not new
    )
    if masked_count:
        logger.info("Masked %d older tool observations", masked_count)

    return result


# ------------------------------------------------------------------
# Stage 3: structured anchored summarization
# ------------------------------------------------------------------

_SUMMARY_SYSTEM_PROMPT = """\
You are a context compressor for an AI agent.  Produce a structured summary
using EXACTLY the sections below.  Merge new information into existing
sections rather than rewriting from scratch.  Be concise — every token counts.

## Session Goal
One-line description of what the user originally asked.

## Files Modified
- path: what changed (one line each)

## Key Decisions
Numbered list of important choices made so far.

## Current State
Where we are right now — what just happened and what is next.

## Pending Actions
Bulleted list of remaining work, if any.
"""


async def _summarize_messages(
    to_summarize: list[Message],
    provider: LLMProvider,
    *,
    existing_summary: str | None = None,
    budget_tokens: int = 800,
) -> str:
    """Produce a structured summary of *to_summarize*.

    If *existing_summary* is provided, the LLM merges the new content into
    the existing sections rather than starting from scratch.
    """
    parts: list[str] = []
    if existing_summary:
        parts.append(f"EXISTING SUMMARY (merge into this):\n{existing_summary}\n")
    parts.append("NEW MESSAGES TO INCORPORATE:\n")
    for msg in to_summarize:
        prefix = msg.role.value.upper()
        if msg.content:
            parts.append(f"{prefix}: {msg.content[:500]}")
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args_preview = str(tc.arguments)[:200]
                parts.append(f"TOOL_CALL: {tc.name}({args_preview})")
        if msg.tool_result:
            parts.append(
                f"TOOL_RESULT({msg.tool_result.name}): {msg.tool_result.content[:200]}"
            )

    response = await provider.generate(
        [
            Message(role=Role.SYSTEM, content=_SUMMARY_SYSTEM_PROMPT),
            Message(role=Role.USER, content="\n".join(parts)),
        ],
        tools=None,
        max_tokens=budget_tokens,
    )
    return response.content or "Previous conversation context."


async def summarize_and_compress(
    messages: list[Message],
    provider: LLMProvider,
    *,
    max_tokens: int = 100_000,
    summary_budget_tokens: int = 800,
    session_id: str | None = None,
    archive_dir: str | None = None,
) -> list[Message]:
    """Structured anchored summarization with incremental merging.

    When context exceeds *max_tokens*, the oldest non-system messages are
    archived to disk and summarized into a persistent structured summary.
    If a prior summary already exists (from a previous compaction cycle),
    the new content is merged into its sections rather than regenerated.

    Falls back to emergency drop if summarization fails.
    """
    total = estimate_total_tokens(messages)
    if total <= max_tokens:
        return messages

    # Separate system prompt(s) from conversation.
    system_msgs: list[Message] = []
    rest: list[Message] = []
    for msg in messages:
        if msg.role == Role.SYSTEM and not rest:
            system_msgs.append(msg)
        else:
            rest.append(msg)

    if len(rest) < 4:
        return _emergency_drop(
            system_msgs, rest, max_tokens=max_tokens,
            session_id=session_id, archive_dir=archive_dir,
        )

    # Check for an existing summary from a previous compaction.
    existing_summary: str | None = None
    if (
        rest
        and rest[0].role == Role.USER
        and rest[0].content
        and rest[0].content.startswith("[CONVERSATION SUMMARY:")
    ):
        existing_summary = rest[0].content
        rest = rest[1:]

    # Determine how many messages to evict.  Take the oldest half, but at
    # least enough to get under budget after summarization.
    split = max(len(rest) // 2, 1)
    to_summarize = rest[:split]
    to_keep = rest[split:]

    # Archive before eviction.
    if session_id and archive_dir:
        from leuk.agent.archive import archive_conversation

        await archive_conversation(session_id, to_summarize, archive_dir)

    try:
        summary_content = await _summarize_messages(
            to_summarize,
            provider,
            existing_summary=existing_summary,
            budget_tokens=summary_budget_tokens,
        )
    except Exception:
        logger.warning("Structured summarization failed, falling back to emergency drop")
        return _emergency_drop(
            system_msgs, rest, max_tokens=max_tokens,
            session_id=session_id, archive_dir=archive_dir,
        )

    summary_msg = Message(
        role=Role.USER,
        content=f"[CONVERSATION SUMMARY:\n{summary_content}\n]",
    )

    result = system_msgs + [summary_msg] + to_keep

    # If still too large (rare), do another round of emergency dropping.
    if estimate_total_tokens(result) > max_tokens:
        return _emergency_drop(
            system_msgs, [summary_msg] + to_keep, max_tokens=max_tokens,
        )

    logger.info(
        "Context compressed: summarized %d messages, %d remain",
        len(to_summarize),
        len(result),
    )
    return result


# ------------------------------------------------------------------
# Stage 4: emergency drop (last resort)
# ------------------------------------------------------------------


def _emergency_drop(
    system_msgs: list[Message],
    rest: list[Message],
    *,
    max_tokens: int = 100_000,
    session_id: str | None = None,
    archive_dir: str | None = None,
) -> list[Message]:
    """Drop oldest non-system messages until under budget.

    This is a synchronous fallback — archiving must be done by the caller
    before invoking this function (or pass session_id/archive_dir for
    best-effort async archiving, though this function is sync).
    """
    dropped: list[Message] = []
    while rest and estimate_total_tokens(system_msgs + rest) > max_tokens:
        dropped.append(rest.pop(0))

    if dropped:
        logger.info("Emergency drop: removed %d oldest messages", len(dropped))
        note = Message(
            role=Role.USER,
            content=(
                f"[SYSTEM NOTE: {len(dropped)} earlier messages were removed "
                f"from context to stay within limits.]"
            ),
        )
        return system_msgs + [note] + rest

    return system_msgs + rest


# ------------------------------------------------------------------
# Unified pipeline
# ------------------------------------------------------------------


async def compact(
    messages: list[Message],
    provider: LLMProvider,
    *,
    max_tokens: int = 100_000,
    max_result_tokens: int = 8_000,
    summary_budget_tokens: int = 800,
    session_id: str | None = None,
    archive_dir: str | None = None,
) -> list[Message]:
    """Run the full tiered compaction pipeline.

    1. Truncate oversized tool results.
    2. Mask older tool observations (adaptive, budget-based).
    3. Structured summarization if still over budget.

    This is the single entry point for context management — callers
    should use this instead of individual functions.
    """
    # Stage 1: truncate
    messages = truncate_tool_results(messages, max_result_tokens=max_result_tokens)

    # Stage 2: mask observations (kicks in at 60% of budget)
    messages = mask_observations(messages, max_tokens=max_tokens)

    # Stage 3: summarize if still over budget
    if estimate_total_tokens(messages) > max_tokens:
        messages = await summarize_and_compress(
            messages,
            provider,
            max_tokens=max_tokens,
            summary_budget_tokens=summary_budget_tokens,
            session_id=session_id,
            archive_dir=archive_dir,
        )

    return messages
