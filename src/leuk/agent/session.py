"""AgentSession: autonomous background agent loop.

An ``AgentSession`` wraps an :class:`Agent` and drives it in a background
``asyncio.Task``.  The REPL (or any other consumer) interacts through two
async queues:

* **input_queue** – push user messages (text) here via :meth:`push`.
* **event_queue** – consume :class:`StreamEvent` / :class:`Message` objects
  that the agent produces.

Key capabilities:

* The agent loop runs **autonomously** — it processes all tool rounds for a
  given user message without waiting for further user input.
* Between turns (after the agent responds and has no more tool calls), the
  loop waits on the *input_queue* for the next message.
* The user can **interrupt** the agent mid-generation via :meth:`interrupt`,
  which cancels the current streaming task.
* The user can **detach** — the session keeps running in the background; a
  new REPL view can :meth:`attach` later and resume receiving events.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from leuk.agent.core import Agent
from leuk.types import (
    AgentState,
    Message,
    StreamEvent,
    StreamEventType,
)

logger = logging.getLogger(__name__)

# Sentinel pushed into event_queue on graceful shutdown.
_STOP_SENTINEL = object()


class AgentSession:
    """Autonomous background wrapper around an :class:`Agent`.

    Parameters
    ----------
    agent:
        A fully initialised Agent (``await agent.init()`` already called).
    maxsize:
        Maximum size of the event queue.  ``0`` means unbounded.
    """

    def __init__(self, agent: Agent, *, maxsize: int = 0) -> None:
        self.agent = agent
        self.input_queue: asyncio.Queue[str] = asyncio.Queue()
        self.event_queue: asyncio.Queue[StreamEvent | Message | object] = asyncio.Queue(
            maxsize=maxsize
        )
        self._state = AgentState.IDLE
        self._task: asyncio.Task[None] | None = None
        self._current_stream_task: asyncio.Task[None] | None = None
        self._interrupted = False

    # ── Public properties ─────────────────────────────────────────

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def session(self):
        """Shortcut to the underlying persistence Session."""
        return self.agent.session

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the background agent loop."""
        if self._task is not None and not self._task.done():
            logger.warning("AgentSession already running")
            return
        self._task = asyncio.create_task(self._loop(), name="agent-session-loop")

    async def stop(self) -> None:
        """Gracefully stop the background loop.

        Pushes a stop sentinel so the loop exits at the next idle point,
        then awaits the task.
        """
        if self._task is None:
            return
        # Cancel any in-flight stream first
        self._cancel_stream()
        # Push sentinel to unblock the queue wait
        await self.input_queue.put("")  # empty string acts as wakeup
        self._state = AgentState.STOPPED
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        await self.event_queue.put(_STOP_SENTINEL)

    def push(self, text: str) -> None:
        """Push a user message for the agent to process.

        Non-blocking; the message is enqueued and the background loop
        picks it up when it finishes the current turn.
        """
        self.input_queue.put_nowait(text)

    def interrupt(self) -> None:
        """Interrupt the current agent generation.

        Cancels the streaming task so the agent stops mid-response.
        The loop will emit a STATE_CHANGE → INTERRUPTED event, then
        resume waiting for the next user message.
        """
        self._interrupted = True
        self._cancel_stream()

    # ── Internal ──────────────────────────────────────────────────

    def _cancel_stream(self) -> None:
        if self._current_stream_task and not self._current_stream_task.done():
            self._current_stream_task.cancel()

    def _set_state(self, new_state: AgentState) -> None:
        if self._state == new_state:
            return
        self._state = new_state
        self.event_queue.put_nowait(
            StreamEvent(type=StreamEventType.STATE_CHANGE, content=new_state.value)
        )

    async def _loop(self) -> None:
        """The main background loop.

        1. Wait for a user message on input_queue.
        2. Run the agent's streaming loop, forwarding all events.
        3. When the agent turn completes, emit TURN_COMPLETE and go to 1.
        """
        try:
            while True:
                self._set_state(AgentState.IDLE)

                # Wait for user input
                text = await self.input_queue.get()
                if not text:
                    # Empty string can be a wakeup signal; check if we should stop
                    if self._state == AgentState.STOPPED:
                        break
                    continue

                self._interrupted = False
                await self._run_turn(text)

        except asyncio.CancelledError:
            logger.debug("AgentSession loop cancelled")
        except Exception as exc:
            logger.exception("AgentSession loop crashed")
            self._set_state(AgentState.STOPPED)
            self.event_queue.put_nowait(StreamEvent(type=StreamEventType.ERROR, content=str(exc)))
        finally:
            self._set_state(AgentState.STOPPED)

    async def _run_turn(self, user_input: str) -> None:
        """Execute one full agent turn (may span multiple tool rounds)."""
        self._set_state(AgentState.THINKING)

        try:
            # Create a task for the streaming so we can cancel it on interrupt
            self._current_stream_task = asyncio.current_task()
            async for event in self.agent.run_stream(user_input):
                if self._interrupted:
                    break

                if isinstance(event, StreamEvent):
                    # Track state transitions
                    if event.type == StreamEventType.TOOL_CALL_START:
                        self._set_state(AgentState.TOOL_RUNNING)
                    elif event.type == StreamEventType.MESSAGE_COMPLETE:
                        pass  # will go to IDLE after turn ends

                    self.event_queue.put_nowait(event)

                elif isinstance(event, Message):
                    # Tool result messages
                    self.event_queue.put_nowait(event)

                    # After tool results, we're about to think again
                    self._set_state(AgentState.THINKING)

        except asyncio.CancelledError:
            self._interrupted = True
            logger.debug("Agent stream cancelled (interrupt)")
        finally:
            self._current_stream_task = None

        if self._interrupted:
            self._set_state(AgentState.INTERRUPTED)
            self.event_queue.put_nowait(
                StreamEvent(type=StreamEventType.STATE_CHANGE, content=AgentState.INTERRUPTED.value)
            )

        # Turn is done
        self.event_queue.put_nowait(StreamEvent(type=StreamEventType.TURN_COMPLETE))
