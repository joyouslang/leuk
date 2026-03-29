"""Tests for the multi-channel messaging system."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from leuk.channels.base import Channel, ChannelMessage
from leuk.channels import ChannelRegistry, register_channel, _factories
from leuk.channels.repl import ReplChannel, _make_repl


# ── ChannelMessage ─────────────────────────────────────────────────────────


def test_channel_message_defaults():
    msg = ChannelMessage(text="hello", chat_id="42", sender="alice", channel="telegram")
    assert msg.text == "hello"
    assert msg.chat_id == "42"
    assert msg.sender == "alice"
    assert msg.channel == "telegram"
    assert msg.metadata == {}


def test_channel_message_with_metadata():
    msg = ChannelMessage(
        text="hi",
        chat_id="1",
        sender="bob",
        channel="slack",
        metadata={"ts": "123.456"},
    )
    assert msg.metadata["ts"] == "123.456"


# ── register_channel / _factories ─────────────────────────────────────────


def test_register_channel_adds_to_registry():
    sentinel = object()

    def _factory(config: Any) -> Any:
        return sentinel

    register_channel("_test_channel_", _factory)
    assert "_test_channel_" in _factories
    assert _factories["_test_channel_"] is _factory
    # Clean up
    del _factories["_test_channel_"]


# ── ReplChannel ────────────────────────────────────────────────────────────


def test_repl_factory_enabled():
    config = MagicMock()
    config.repl_enabled = True
    ch = _make_repl(config)
    assert isinstance(ch, ReplChannel)
    assert ch.name == "repl"


def test_repl_factory_disabled():
    config = MagicMock()
    config.repl_enabled = False
    ch = _make_repl(config)
    assert ch is None


@pytest.mark.asyncio
async def test_repl_send_writes_to_stdout(capsys):
    ch = ReplChannel(prompt="")
    await ch.send("default", "hello world")
    captured = capsys.readouterr()
    assert "hello world" in captured.out


@pytest.mark.asyncio
async def test_repl_send_appends_newline(capsys):
    ch = ReplChannel(prompt="")
    await ch.send("default", "no newline")
    captured = capsys.readouterr()
    assert captured.out.endswith("\n")


@pytest.mark.asyncio
async def test_repl_send_does_not_double_newline(capsys):
    ch = ReplChannel(prompt="")
    await ch.send("default", "already\n")
    captured = capsys.readouterr()
    assert captured.out == "already\n"


@pytest.mark.asyncio
async def test_repl_on_message_registers_callback():
    ch = ReplChannel(prompt="")
    cb = AsyncMock()
    ch.on_message(cb)
    assert ch._callback is cb


@pytest.mark.asyncio
async def test_repl_disconnect_cancels_task():
    ch = ReplChannel(prompt="")
    # connect() starts the background task; disconnect() should cancel it
    await ch.connect()
    assert ch._task is not None
    await ch.disconnect()
    assert ch._task is None


# ── Channel Protocol ───────────────────────────────────────────────────────


def test_channel_is_runtime_checkable():
    # ReplChannel satisfies the Channel protocol structurally
    ch = ReplChannel()
    assert isinstance(ch, Channel)


# ── ChannelRegistry ────────────────────────────────────────────────────────


class _FakeChannel:
    """A no-op channel for registry tests."""

    name = "fake"

    def __init__(self) -> None:
        self._callback = None
        self.connected = False
        self.disconnected = False
        self.sent: list[tuple[str, str]] = []

    async def connect(self) -> None:
        self.connected = True

    async def send(self, chat_id: str, text: str) -> None:
        self.sent.append((chat_id, text))

    def on_message(self, callback: Any) -> None:
        self._callback = callback

    async def disconnect(self) -> None:
        self.disconnected = True


class _FakeSession:
    """A minimal stub that satisfies the ChannelRegistry's session interface."""

    def __init__(self) -> None:
        self.input_queue: asyncio.Queue[str] = asyncio.Queue()
        self.event_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._started = False
        from leuk.types import AgentState
        self.state = AgentState.IDLE

    def start(self) -> None:
        self._started = True

    def push(self, text: str) -> None:
        self.input_queue.put_nowait(text)

    async def stop(self) -> None:
        pass


@pytest.mark.asyncio
async def test_registry_start_stop_with_no_factories():
    """Registry.start() with an empty factory table completes without error."""
    # Use a fresh registry with no real factories
    registry = ChannelRegistry(
        session_factory=AsyncMock(return_value=_FakeSession()),
        config=MagicMock(repl_enabled=False),
    )
    # Patch _factories to empty for this test
    import leuk.channels as ch_mod
    original = dict(ch_mod._factories)
    ch_mod._factories.clear()
    try:
        await registry.start()
        assert registry.active_channels == []
        assert registry.session_count == 0
        await registry.stop()
    finally:
        ch_mod._factories.update(original)


@pytest.mark.asyncio
async def test_registry_connect_channel():
    fake_ch = _FakeChannel()

    import leuk.channels as ch_mod
    original = dict(ch_mod._factories)
    ch_mod._factories.clear()
    ch_mod._factories["fake"] = lambda cfg: fake_ch

    sessions: list[_FakeSession] = []

    async def _factory(channel: str, chat_id: str) -> _FakeSession:
        s = _FakeSession()
        sessions.append(s)
        return s

    registry = ChannelRegistry(session_factory=_factory, config=MagicMock())
    try:
        await registry.start()
        assert "fake" in registry.active_channels
        assert fake_ch.connected
    finally:
        await registry.stop()
        ch_mod._factories.clear()
        ch_mod._factories.update(original)


@pytest.mark.asyncio
async def test_registry_factory_returns_none_skips_channel():
    """A factory returning None must not appear in active_channels."""
    import leuk.channels as ch_mod
    original = dict(ch_mod._factories)
    ch_mod._factories.clear()
    ch_mod._factories["none_channel"] = lambda cfg: None

    registry = ChannelRegistry(session_factory=AsyncMock(), config=MagicMock())
    try:
        await registry.start()
        assert "none_channel" not in registry.active_channels
        await registry.stop()
    finally:
        ch_mod._factories.clear()
        ch_mod._factories.update(original)


@pytest.mark.asyncio
async def test_registry_routes_message_to_session():
    """Incoming messages create a session and push the text."""
    fake_ch = _FakeChannel()
    session = _FakeSession()

    import leuk.channels as ch_mod
    original = dict(ch_mod._factories)
    ch_mod._factories.clear()
    ch_mod._factories["fake"] = lambda cfg: fake_ch

    async def _factory(channel: str, chat_id: str) -> _FakeSession:
        return session

    registry = ChannelRegistry(session_factory=_factory, config=MagicMock())
    try:
        await registry.start()
        msg = ChannelMessage(text="ping", chat_id="room1", sender="u1", channel="fake")
        await registry._handle_message(msg)

        assert session._started
        assert not session.input_queue.empty()
        received = session.input_queue.get_nowait()
        assert received == "ping"
    finally:
        await registry.stop()
        ch_mod._factories.clear()
        ch_mod._factories.update(original)


@pytest.mark.asyncio
async def test_registry_reuses_session_for_same_chat():
    """Same (channel, chat_id) must reuse the existing session."""
    fake_ch = _FakeChannel()
    created: list[_FakeSession] = []

    import leuk.channels as ch_mod
    original = dict(ch_mod._factories)
    ch_mod._factories.clear()
    ch_mod._factories["fake"] = lambda cfg: fake_ch

    async def _factory(channel: str, chat_id: str) -> _FakeSession:
        s = _FakeSession()
        created.append(s)
        return s

    registry = ChannelRegistry(session_factory=_factory, config=MagicMock())
    try:
        await registry.start()
        msg1 = ChannelMessage(text="first", chat_id="room1", sender="u1", channel="fake")
        msg2 = ChannelMessage(text="second", chat_id="room1", sender="u1", channel="fake")
        await registry._handle_message(msg1)
        await registry._handle_message(msg2)

        assert len(created) == 1  # only one session created
        assert registry.session_count == 1
    finally:
        await registry.stop()
        ch_mod._factories.clear()
        ch_mod._factories.update(original)


@pytest.mark.asyncio
async def test_registry_separate_sessions_for_different_chats():
    """Different chat_ids must get separate sessions."""
    fake_ch = _FakeChannel()
    created: list[_FakeSession] = []

    import leuk.channels as ch_mod
    original = dict(ch_mod._factories)
    ch_mod._factories.clear()
    ch_mod._factories["fake"] = lambda cfg: fake_ch

    async def _factory(channel: str, chat_id: str) -> _FakeSession:
        s = _FakeSession()
        created.append(s)
        return s

    registry = ChannelRegistry(session_factory=_factory, config=MagicMock())
    try:
        await registry.start()
        await registry._handle_message(
            ChannelMessage(text="hi", chat_id="room1", sender="u1", channel="fake")
        )
        await registry._handle_message(
            ChannelMessage(text="hey", chat_id="room2", sender="u2", channel="fake")
        )
        assert len(created) == 2
        assert registry.session_count == 2
    finally:
        await registry.stop()
        ch_mod._factories.clear()
        ch_mod._factories.update(original)


# ── Config ─────────────────────────────────────────────────────────────────


def test_channels_config_defaults():
    from leuk.config import ChannelsConfig

    cfg = ChannelsConfig()
    assert cfg.repl_enabled is True
    assert cfg.telegram_bot_token == ""
    assert cfg.slack_bot_token == ""
    assert cfg.slack_app_token == ""
    assert cfg.discord_bot_token == ""
    assert cfg.allowed_users == []


def test_settings_has_channels():
    from leuk.config import Settings

    s = Settings()
    assert hasattr(s, "channels")
    from leuk.config import ChannelsConfig
    assert isinstance(s.channels, ChannelsConfig)


# ── Optional channel factories return None without deps ────────────────────


def test_telegram_factory_no_token():
    from leuk.channels.telegram import _make_telegram

    config = MagicMock()
    config.telegram_bot_token = ""
    result = _make_telegram(config)
    assert result is None


def test_slack_factory_missing_tokens():
    from leuk.channels.slack import _make_slack

    config = MagicMock()
    config.slack_bot_token = ""
    config.slack_app_token = ""
    result = _make_slack(config)
    assert result is None


def test_discord_factory_no_token():
    from leuk.channels.discord import _make_discord

    config = MagicMock()
    config.discord_bot_token = ""
    result = _make_discord(config)
    assert result is None


# ── Allowlist ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_allowlist_blocks_unlisted_sender():
    """Messages from senders not in allowed_users must be silently dropped."""
    fake_ch = _FakeChannel()
    created: list[_FakeSession] = []

    import leuk.channels as ch_mod
    original = dict(ch_mod._factories)
    ch_mod._factories.clear()
    ch_mod._factories["fake"] = lambda cfg: fake_ch

    async def _factory(channel: str, chat_id: str) -> _FakeSession:
        s = _FakeSession()
        created.append(s)
        return s

    config = MagicMock()
    config.allowed_users = ["user_good"]
    registry = ChannelRegistry(session_factory=_factory, config=config)
    try:
        # Skip _import_channels to avoid spawning the real REPL stdin reader.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(ch_mod, "_import_channels", lambda: None)
            await registry.start()
        msg = ChannelMessage(text="hi", chat_id="c1", sender="user_bad", channel="fake")
        await registry._handle_message(msg)

        assert len(created) == 0
        assert registry.session_count == 0
    finally:
        await registry.stop()
        ch_mod._factories.clear()
        ch_mod._factories.update(original)


@pytest.mark.asyncio
async def test_allowlist_permits_listed_sender():
    """Messages from senders in allowed_users must be routed normally."""
    fake_ch = _FakeChannel()
    session = _FakeSession()

    import leuk.channels as ch_mod
    original = dict(ch_mod._factories)
    ch_mod._factories.clear()
    ch_mod._factories["fake"] = lambda cfg: fake_ch

    async def _factory(channel: str, chat_id: str) -> _FakeSession:
        return session

    config = MagicMock()
    config.allowed_users = ["user_good"]
    registry = ChannelRegistry(session_factory=_factory, config=config)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(ch_mod, "_import_channels", lambda: None)
            await registry.start()
        msg = ChannelMessage(text="hi", chat_id="c1", sender="user_good", channel="fake")
        await registry._handle_message(msg)

        assert session._started
        assert session.input_queue.get_nowait() == "hi"
    finally:
        await registry.stop()
        ch_mod._factories.clear()
        ch_mod._factories.update(original)


@pytest.mark.asyncio
async def test_empty_allowlist_permits_all():
    """An empty allowed_users list means unrestricted access."""
    fake_ch = _FakeChannel()
    session = _FakeSession()

    import leuk.channels as ch_mod
    original = dict(ch_mod._factories)
    ch_mod._factories.clear()
    ch_mod._factories["fake"] = lambda cfg: fake_ch

    async def _factory(channel: str, chat_id: str) -> _FakeSession:
        return session

    config = MagicMock()
    config.allowed_users = []
    registry = ChannelRegistry(session_factory=_factory, config=config)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(ch_mod, "_import_channels", lambda: None)
            await registry.start()
        msg = ChannelMessage(text="hi", chat_id="c1", sender="anyone", channel="fake")
        await registry._handle_message(msg)

        assert session._started
        assert session.input_queue.get_nowait() == "hi"
    finally:
        await registry.stop()
        ch_mod._factories.clear()
        ch_mod._factories.update(original)


@pytest.mark.asyncio
async def test_allowlist_exempts_repl_channel():
    """The REPL channel must bypass the allowlist — it is always local."""
    fake_ch = _FakeChannel()
    fake_ch.name = "repl"
    session = _FakeSession()

    import leuk.channels as ch_mod
    original = dict(ch_mod._factories)
    ch_mod._factories.clear()
    ch_mod._factories["repl"] = lambda cfg: fake_ch

    async def _factory(channel: str, chat_id: str) -> _FakeSession:
        return session

    config = MagicMock()
    config.allowed_users = ["someone_else"]
    registry = ChannelRegistry(session_factory=_factory, config=config)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(ch_mod, "_import_channels", lambda: None)
            await registry.start()
        msg = ChannelMessage(
            text="local", chat_id="default", sender="unlisted_local_user", channel="repl"
        )
        await registry._handle_message(msg)

        assert session._started
        assert session.input_queue.get_nowait() == "local"
    finally:
        await registry.stop()
        ch_mod._factories.clear()
        ch_mod._factories.update(original)
