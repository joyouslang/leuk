"""Tests for the steering persistence guard (the agent-loop behaviour).

Covers the self-reflection continuation, its bound, the truncation fast-path,
tool-error enrichment, mid-loop reminders, and finish_reason plumbing. The guard
is keyed on ``settings.llm.provider`` (a config signal), never on the provider
class — so these use MockProvider with the provider *name* set to drive gating.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from leuk.agent.core import Agent
from leuk.agent.steering import STEERING_REMINDER
from leuk.config import AgentConfig, LLMConfig, Settings, SQLiteConfig, SteeringConfig
from leuk.persistence.memory import MemoryStore
from leuk.persistence.sqlite import SQLiteStore
from leuk.providers.openai import OpenAIProvider
from leuk.tools import create_default_registry
from leuk.types import Message, Role, StreamEvent, StreamEventType, ToolCall

from tests.conftest import MockProvider


@pytest_asyncio.fixture
async def make_agent(tmp_path: Path):
    """Factory for fully-wired agents with custom steering/provider settings."""
    created: list[tuple[Agent, SQLiteStore]] = []

    async def _factory(
        *,
        steering: SteeringConfig,
        provider_name: str = "local",
        max_tool_rounds: int = 10,
        responses: list[Message] | None = None,
    ) -> tuple[Agent, MockProvider, SQLiteStore]:
        settings = Settings(
            llm=LLMConfig(provider=provider_name),
            sqlite=SQLiteConfig(path=str(tmp_path / f"t{len(created)}.db")),
            agent=AgentConfig(max_tool_rounds=max_tool_rounds),
            steering=steering,
        )
        provider = MockProvider(responses)
        sqlite = SQLiteStore(settings.sqlite)
        agent = Agent(
            settings=settings,
            provider=provider,
            tool_registry=create_default_registry(),
            sqlite=sqlite,
            hot_store=MemoryStore(),
        )
        await agent.init()
        created.append((agent, sqlite))
        return agent, provider, sqlite

    yield _factory

    for agent, sqlite in created:
        await agent.shutdown()
        await sqlite.close()


def _has_steering_nudge(agent: Agent) -> bool:
    return any((m.content or "").startswith("[STEERING]") for m in agent._messages)


async def _collect_stream(agent: Agent, text: str) -> tuple[list[Message], list[Message]]:
    completed: list[Message] = []
    tool_msgs: list[Message] = []
    async for item in agent.run_stream(text):
        if isinstance(item, StreamEvent):
            if item.type == StreamEventType.MESSAGE_COMPLETE and item.message is not None:
                completed.append(item.message)
        else:
            tool_msgs.append(item)
    return completed, tool_msgs


# ── self-reflection continuation ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_giveup_then_recover_run(make_agent):
    agent, provider, _ = await make_agent(
        steering=SteeringConfig(enabled="on", reflect_only_after_tool_use=False),
        responses=[
            Message(role=Role.ASSISTANT, content="I cannot do this."),  # give-up
            Message(role=Role.ASSISTANT, content="CONTINUE\nretry now"),  # reflection: continue
            Message(role=Role.ASSISTANT, content="All finished."),  # recovery answer
            Message(role=Role.ASSISTANT, content="DONE"),  # reflection: accept
        ],
    )

    messages = [m async for m in agent.run("do it")]

    # The give-up did NOT end the turn; the recovery answer was produced.
    assert [m.content for m in messages] == ["I cannot do this.", "All finished."]
    assert provider._call_count == 4  # 2 turns + 2 reflection checks
    assert _has_steering_nudge(agent)
    # The reflection's next-action hint rode into the nudge.
    assert any("retry now" in (m.content or "") for m in agent._messages)


@pytest.mark.asyncio
async def test_giveup_then_recover_stream(make_agent):
    agent, provider, _ = await make_agent(
        steering=SteeringConfig(enabled="on", reflect_only_after_tool_use=False),
        responses=[
            Message(role=Role.ASSISTANT, content="I cannot do this."),
            Message(role=Role.ASSISTANT, content="CONTINUE\nretry now"),
            Message(role=Role.ASSISTANT, content="All finished."),
            Message(role=Role.ASSISTANT, content="DONE"),
        ],
    )

    completed, _ = await _collect_stream(agent, "do it")

    assert [m.content for m in completed] == ["I cannot do this.", "All finished."]
    assert provider._call_count == 4
    assert _has_steering_nudge(agent)
    # The internal nudge is never surfaced as a streamed message.
    assert not any((m.content or "").startswith("[STEERING]") for m in completed)


@pytest.mark.asyncio
async def test_continuation_bound_enforced(make_agent):
    # Reflection always says CONTINUE; the cap must stop the loop deterministically.
    agent, provider, _ = await make_agent(
        steering=SteeringConfig(
            enabled="on", reflect_only_after_tool_use=False, max_continuations=2
        ),
        responses=[Message(role=Role.ASSISTANT, content="CONTINUE") for _ in range(12)],
    )

    _ = [m async for m in agent.run("loop forever?")]

    assert agent._continuations == 2  # exactly the cap, no more
    # 3 give-up turns + 2 reflection checks (the 3rd stop hits the cap, no check).
    assert provider._call_count == 5


# ── truncation fast-path ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_truncation_continues_without_reflection_call(make_agent):
    agent, provider, _ = await make_agent(
        steering=SteeringConfig(enabled="on", nudge_on_truncation=True),
        responses=[
            Message(
                role=Role.ASSISTANT,
                content="partial output",
                metadata={"finish_reason": "length"},
            ),
            Message(role=Role.ASSISTANT, content="complete answer"),
        ],
    )

    messages = [m async for m in agent.run("write a lot")]

    assert [m.content for m in messages] == ["partial output", "complete answer"]
    assert agent._continuations == 1
    # Exactly 2 generate calls — the truncation path spends NO reflection call.
    assert provider._call_count == 2
    assert any((m.content or "") == "continue" for m in agent._messages)


# ── no-op / gating ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_disabled_for_strong_model_is_noop(make_agent):
    # auto + non-local → steering inactive → today's behaviour (stop immediately).
    agent, provider, _ = await make_agent(
        steering=SteeringConfig(enabled="auto"),
        provider_name="anthropic",
        responses=[Message(role=Role.ASSISTANT, content="I refuse.")],
    )

    messages = [m async for m in agent.run("do it")]

    assert [m.content for m in messages] == ["I refuse."]
    assert provider._call_count == 1
    assert agent._continuations == 0
    assert not _has_steering_nudge(agent)


@pytest.mark.asyncio
async def test_reflect_only_after_tool_use_skips_chat(make_agent):
    # A refusal-looking reply with no tool use this turn is accepted (no check).
    agent, provider, _ = await make_agent(
        steering=SteeringConfig(enabled="on", reflect_only_after_tool_use=True),
        responses=[Message(role=Role.ASSISTANT, content="I'm not able to help with that.")],
    )

    messages = [m async for m in agent.run("hi there")]

    assert [m.content for m in messages] == ["I'm not able to help with that."]
    assert provider._call_count == 1  # no reflection call
    assert not _has_steering_nudge(agent)


# ── tool-error enrichment ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_error_enriched_when_active(make_agent):
    agent, _, _ = await make_agent(
        steering=SteeringConfig(enabled="on", enrich_tool_errors=True),
    )
    result = await agent._execute_tool(ToolCall(id="x", name="does_not_exist", arguments={}))
    assert result.is_error
    assert "[recovery hint]" in result.content


@pytest.mark.asyncio
async def test_tool_error_not_enriched_when_disabled(make_agent):
    agent, _, _ = await make_agent(
        steering=SteeringConfig(enabled="on", enrich_tool_errors=False),
    )
    result = await agent._execute_tool(ToolCall(id="x", name="does_not_exist", arguments={}))
    assert result.is_error
    assert "[recovery hint]" not in result.content


# ── mid-loop reminders (ephemeral) ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_reminder_injected_periodically(make_agent):
    agent, provider, _ = await make_agent(
        steering=SteeringConfig(enabled="on", reminder_interval=1, max_continuations=0),
        responses=[
            Message(
                role=Role.ASSISTANT,
                tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo hi"})],
            ),
            Message(role=Role.ASSISTANT, content="done"),
        ],
    )

    _ = [m async for m in agent.run("run something")]

    # Round 1's context (the last generate call) carried the reminder…
    assert any((m.content or "") == STEERING_REMINDER for m in provider.last_messages)
    # …but it was never persisted into the real history (ephemeral).
    assert not _has_steering_nudge(agent)


@pytest.mark.asyncio
async def test_reminder_injected_after_tool_error(make_agent):
    agent, provider, _ = await make_agent(
        steering=SteeringConfig(enabled="on", reminder_interval=0, max_continuations=0),
        responses=[
            Message(
                role=Role.ASSISTANT,
                tool_calls=[ToolCall(id="c1", name="does_not_exist", arguments={})],
            ),
            Message(role=Role.ASSISTANT, content="done"),
        ],
    )

    _ = [m async for m in agent.run("trigger an error")]

    # Periodic reminders are off (interval 0), but the round after an error still
    # gets one.
    assert any((m.content or "") == STEERING_REMINDER for m in provider.last_messages)


# ── circle-breaker (anti-spin) ─────────────────────────────────────────────


def _shell(cmd: str) -> Message:
    return Message(
        role=Role.ASSISTANT,
        content="working",
        tool_calls=[ToolCall(id="c", name="shell", arguments={"command": cmd})],
    )


def _steering_nudges(agent: Agent) -> list[str]:
    return [m.content or "" for m in agent._messages if (m.content or "").startswith("[STEERING]")]


@pytest.mark.asyncio
async def test_circle_breaker_nudges_then_escalates(make_agent):
    # The model repeats the same tool call forever; the breaker redirects once,
    # then forces a tools-off consolidation reply instead of spinning to the cap.
    agent, provider, _ = await make_agent(
        steering=SteeringConfig(
            enabled="on",
            loop_detection=True,
            loop_min_rounds=4,
            loop_max_interventions=1,
            max_continuations=0,
        ),
        max_tool_rounds=20,
        responses=[_shell("echo loop") for _ in range(5)]
        + [Message(role=Role.ASSISTANT, content="consolidated answer")],
    )

    messages = [m async for m in agent.run("spin")]

    assert messages[-1].content == "consolidated answer"  # forced consolidation
    assert agent._loop_interventions == 1
    nudges = _steering_nudges(agent)
    assert len(nudges) == 2  # 1 redirect + 1 consolidation directive
    assert provider._call_count == 6  # 5 spins + 1 consolidation generate (not 20)


@pytest.mark.asyncio
async def test_circle_breaker_escalates_in_stream(make_agent):
    agent, provider, _ = await make_agent(
        steering=SteeringConfig(
            enabled="on",
            loop_detection=True,
            loop_min_rounds=4,
            loop_max_interventions=1,
            max_continuations=0,
        ),
        max_tool_rounds=20,
        responses=[_shell("echo loop") for _ in range(5)]
        + [Message(role=Role.ASSISTANT, content="consolidated answer")],
    )

    completed, _ = await _collect_stream(agent, "spin")

    assert completed[-1].content == "consolidated answer"
    assert agent._loop_interventions == 1


@pytest.mark.asyncio
async def test_circle_breaker_diversify_avoids_escalation(make_agent):
    # After one redirect the model varies its call → not a circle → no escalation.
    agent, provider, _ = await make_agent(
        steering=SteeringConfig(
            enabled="on",
            loop_detection=True,
            loop_min_rounds=4,
            loop_max_interventions=1,
            max_continuations=0,
        ),
        max_tool_rounds=20,
        responses=[
            _shell("echo a"),
            _shell("echo a"),
            _shell("echo a"),
            _shell("echo a"),
            _shell("echo b"),  # diversify after the redirect
            Message(role=Role.ASSISTANT, content="done diversified"),
        ],
    )

    messages = [m async for m in agent.run("spin then vary")]

    assert messages[-1].content == "done diversified"
    assert agent._loop_interventions == 1
    assert len(_steering_nudges(agent)) == 1  # only the redirect; no consolidation


@pytest.mark.asyncio
async def test_loop_detection_off_is_noop(make_agent):
    # With loop detection off, a spinning model runs to max_tool_rounds (no breaker).
    agent, _, _ = await make_agent(
        steering=SteeringConfig(enabled="on", loop_detection=False),
        max_tool_rounds=3,
        responses=[_shell("echo loop") for _ in range(5)]
        + [Message(role=Role.ASSISTANT, content="forced final")],
    )

    _ = [m async for m in agent.run("spin")]

    assert _steering_nudges(agent) == []  # no circle intervention
    assert agent._loop_interventions == 0


# ── finish_reason plumbing (OpenAI/local provider) ─────────────────────────


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeCompletions:
    def __init__(self, result):
        self._result = result

    async def create(self, **kwargs):
        return self._result


def _make_provider_with_client(result) -> OpenAIProvider:
    # api_key only satisfies the real client constructor; we swap _client below.
    provider = OpenAIProvider(LLMConfig(provider="local"), api_key="test")
    provider._client = _Obj(chat=_Obj(completions=_FakeCompletions(result)))
    return provider


@pytest.mark.asyncio
async def test_generate_plumbs_finish_reason():
    response = _Obj(
        choices=[_Obj(message=_Obj(content="hi", tool_calls=None), finish_reason="length")]
    )
    provider = _make_provider_with_client(response)
    msg = await provider.generate([Message(role=Role.USER, content="x")])
    assert msg.metadata.get("finish_reason") == "length"


@pytest.mark.asyncio
async def test_generate_finish_reason_absent_is_safe():
    response = _Obj(choices=[_Obj(message=_Obj(content="hi", tool_calls=None), finish_reason=None)])
    provider = _make_provider_with_client(response)
    msg = await provider.generate([Message(role=Role.USER, content="x")])
    assert msg.metadata.get("finish_reason") is None


class _FakeStreamResult:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c


@pytest.mark.asyncio
async def test_stream_plumbs_finish_reason():
    chunks = [
        _Obj(choices=[_Obj(delta=_Obj(content="hi", tool_calls=None), finish_reason=None)]),
        _Obj(choices=[_Obj(delta=_Obj(content=None, tool_calls=None), finish_reason="length")]),
    ]
    provider = _make_provider_with_client(_FakeStreamResult(chunks))
    events = [e async for e in provider.stream([Message(role=Role.USER, content="x")])]
    completes = [e for e in events if e.type == StreamEventType.MESSAGE_COMPLETE]
    assert completes and completes[-1].message is not None
    assert completes[-1].message.metadata.get("finish_reason") == "length"


# ── text tool-call salvage ─────────────────────────────────────────────────

_TEXT_CALL = (
    "<tool_call><function=shell><parameter=command>echo hi</parameter></function></tool_call>"
)


@pytest.mark.asyncio
async def test_salvages_text_tool_call_and_executes(make_agent):
    # The model wrote the call as text (no native tool_calls); salvage runs it.
    agent, provider, _ = await make_agent(
        steering=SteeringConfig(enabled="on", salvage_text_tool_calls=True, max_continuations=0),
        responses=[
            Message(role=Role.ASSISTANT, content=_TEXT_CALL),
            Message(role=Role.ASSISTANT, content="all done"),
        ],
    )

    messages = [m async for m in agent.run("open the page")]

    assert messages[0].tool_calls and messages[0].tool_calls[0].name == "shell"
    tool_msgs = [m for m in messages if m.role is Role.TOOL]
    assert tool_msgs and "hi" in tool_msgs[0].tool_result.content  # the salvaged call ran
    assert messages[-1].content == "all done"


@pytest.mark.asyncio
async def test_salvage_noop_when_steering_inactive(make_agent):
    agent, provider, _ = await make_agent(
        steering=SteeringConfig(enabled="auto"),  # auto + non-local → inactive
        provider_name="anthropic",
        responses=[Message(role=Role.ASSISTANT, content=_TEXT_CALL)],
    )

    messages = [m async for m in agent.run("open the page")]

    assert messages[0].tool_calls is None  # not salvaged
    assert not any(m.role is Role.TOOL for m in messages)  # nothing executed


@pytest.mark.asyncio
async def test_salvage_off_keeps_text_unparsed(make_agent):
    agent, provider, _ = await make_agent(
        steering=SteeringConfig(enabled="on", salvage_text_tool_calls=False, max_continuations=0),
        responses=[Message(role=Role.ASSISTANT, content=_TEXT_CALL)],
    )

    messages = [m async for m in agent.run("open the page")]

    assert messages[0].tool_calls is None
    assert not any(m.role is Role.TOOL for m in messages)
