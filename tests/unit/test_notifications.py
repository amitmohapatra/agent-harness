"""One notification interface (``notifications``): a :class:`Notification` goes to the user's
channels among those the harness has (``Harness(channels=)``, the chat's by default; which
ones, the company's ``Harness(user_channels=)``, else the chat only) through ``Harness.notify``,
at most once per channel for its idempotency key — called again (a retry, a resumed run) it is
not sent again; a channel that failed is tried again by the next call. The chat's channel
delivers through the run the notification is about: the run is in the user's thread, where the
thread stream delivers it and the conversation keeps its reply."""

from __future__ import annotations

import asyncio
import logging

import pytest

from trellis import Harness, Runtime, Settings
from trellis.contracts import ConfigurationError, RunStart
from trellis.harness.notifications import (
    ChatChannel,
    Kind,
    Notification,
    Undelivered,
)

THREAD = "chat-1"


class Fake:
    """A channel that keeps what it was given; ``fails`` times it raises first."""

    def __init__(self, name: str = "fake", *, fails: int = 0, waits: bool = False) -> None:
        self.name = name
        self.fails = fails
        self.waits = waits
        self.sent: list[Notification] = []

    async def deliver(self, notification: Notification, harness: Harness) -> None:
        if self.waits:
            await asyncio.sleep(0.01)
        if self.fails:
            self.fails -= 1
            raise Undelivered("the provider is down")
        self.sent.append(notification)


def note(key: str = "k-1", **changes: str) -> Notification:
    given: dict[str, str] = {
        "tenant": "t",
        "user": "ada",
        "text": "Stock of SKU-1 is 4.",
        "key": key,
        "thread": THREAD,
        "run_id": "run_1",
        "schedule_id": "sch_1",
    }
    return Notification(kind=Kind.RESULT, **{**given, **changes})


async def answer(input: str, agent: Runtime) -> str:
    return "Stock of SKU-1 is 4."


async def everything(user: str, tenant: str, notification: Notification) -> list[str]:
    """A company's routing that gives every user every channel it has connected."""
    return ["one", "two", "fake", "down", "up"]


async def test_a_notification_goes_to_each_of_its_channels_once_however_often_it_is_sent() -> None:
    one, two = Fake("one"), Fake("two")
    async with Harness(config=Settings(), channels=[one, two], user_channels=everything) as h:
        assert await h.notify(note()) == ["one", "two"]
        assert await h.notify(note()) == []  # a retry, a resumed run: sent already
        assert await h.notify(note("k-2")) == ["one", "two"]  # another notification
        assert await h.notify(note("k-2", tenant="t2")) == ["one", "two"]  # another tenant's
    assert [n.key for n in one.sent] == ["k-1", "k-2", "k-2"]
    assert [n.key for n in two.sent] == ["k-1", "k-2", "k-2"]


async def test_two_calls_at_once_deliver_it_once() -> None:
    slow = Fake(waits=True)
    async with Harness(config=Settings(), channels=[slow], user_channels=everything) as h:
        sent = await asyncio.gather(h.notify(note()), h.notify(note()))
    assert sorted(sent) == [[], ["fake"]]
    assert len(slow.sent) == 1


async def test_a_channel_that_fails_is_tried_again_and_the_others_are_not(
    caplog: pytest.LogCaptureFixture,
) -> None:
    down, up = Fake("down", fails=1), Fake("up")
    async with Harness(config=Settings(), channels=[down, up], user_channels=everything) as h:
        with caplog.at_level(logging.WARNING, logger="trellis.notifications"):
            assert await h.notify(note()) == ["up"]
        assert "down" in caplog.text
        assert "the provider is down" in caplog.text
        assert await h.notify(note()) == ["down"]  # the next call: only the one that failed
        assert await h.notify(note()) == []
    assert len(down.sent) == len(up.sent) == 1


class Chat(Fake):
    """The chat's channel, faked: by its name, it is what a user gets unless routed."""

    def __init__(self) -> None:
        super().__init__("chat")


async def test_with_no_routing_a_user_gets_the_chat_only() -> None:
    chat, email = Chat(), Fake("email")
    async with Harness(config=Settings(), channels=[chat, email]) as h:
        assert await h.notify(note()) == ["chat"]
    assert len(chat.sent) == 1
    assert email.sent == []


async def test_routing_that_says_nothing_for_a_user_gives_the_chat_only() -> None:
    asked: list[tuple[str, str, str]] = []

    async def nothing(user: str, tenant: str, notification: Notification) -> None:
        asked.append((user, tenant, notification.key))

    async with Harness(
        config=Settings(), channels=[Chat(), Fake("email")], user_channels=nothing
    ) as h:
        assert await h.notify(note()) == ["chat"]
    assert asked == [("ada", "t", "k-1")]


async def test_the_routing_picks_each_users_channels() -> None:
    chat, email, slack = Chat(), Fake("email"), Fake("slack")

    async def theirs(user: str, tenant: str, notification: Notification) -> list[str]:
        return {"ada": ["email"], "bob": ["slack", "chat"]}[user]

    async with Harness(config=Settings(), channels=[chat, email, slack], user_channels=theirs) as h:
        assert await h.notify(note(user="ada")) == ["email"]
        assert await h.notify(note("k-2", user="bob")) == ["chat", "slack"]  # in channels' order
    assert [n.user for n in email.sent] == ["ada"]
    assert [n.user for n in slack.sent] == [n.user for n in chat.sent] == ["bob"]


async def test_the_routing_picks_the_channels_by_what_happened() -> None:
    chat, email = Chat(), Fake("email")

    async def by_kind(user: str, tenant: str, notification: Notification) -> list[str]:
        return ["email"] if notification.kind is Kind.APPROVAL else ["chat"]

    async with Harness(config=Settings(), channels=[chat, email], user_channels=by_kind) as h:
        assert await h.notify(note()) == ["chat"]
        asked = Notification(kind=Kind.APPROVAL, tenant="t", user="ada", text="Approve?", key="k-2")
        assert await h.notify(asked) == ["email"]
    assert [n.kind for n in chat.sent] == [Kind.RESULT]
    assert [n.kind for n in email.sent] == [Kind.APPROVAL]


async def test_a_channel_the_harness_does_not_have_is_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    email = Fake("email")

    async def unknown(user: str, tenant: str, notification: Notification) -> list[str]:
        return ["teams", "email"]

    async with Harness(config=Settings(), channels=[Chat(), email], user_channels=unknown) as h:
        with caplog.at_level(logging.WARNING, logger="trellis.notifications"):
            assert await h.notify(note()) == ["email"]
    assert "teams" in caplog.text
    assert len(email.sent) == 1


async def test_routing_that_fails_gives_the_chat_only(caplog: pytest.LogCaptureFixture) -> None:
    chat, email = Chat(), Fake("email")

    async def broken(user: str, tenant: str, notification: Notification) -> list[str]:
        raise RuntimeError("the directory is down")

    async with Harness(config=Settings(), channels=[chat, email], user_channels=broken) as h:
        with caplog.at_level(logging.WARNING, logger="trellis.notifications"):
            assert await h.notify(note()) == ["chat"]
    assert "the directory is down" in caplog.text
    assert email.sent == []


async def test_at_most_once_holds_when_the_routing_changes_its_mind() -> None:
    """Per channel, tenant and key: a channel given it already is not given it again, whatever
    the routing says the next time; one routed to it newly is."""
    chat, email = Chat(), Fake("email")
    routes = [["email"], ["email", "chat"], ["chat"]]

    async def changing(user: str, tenant: str, notification: Notification) -> list[str]:
        return routes.pop(0)

    async with Harness(config=Settings(), channels=[chat, email], user_channels=changing) as h:
        assert await h.notify(note()) == ["email"]
        assert await h.notify(note()) == ["chat"]
        assert await h.notify(note()) == []
    assert len(chat.sent) == len(email.sent) == 1


async def test_the_chat_is_the_default_channel_and_none_can_be_configured() -> None:
    async with Harness(config=Settings()) as h:
        assert [c.name for c in h.channels] == ["chat"]
        assert isinstance(h.channels[0], ChatChannel)
    async with Harness(config=Settings(), channels=[]) as h:
        assert h.channels == ()
        assert await h.notify(note()) == []


def test_two_channels_of_one_name_are_refused() -> None:
    with pytest.raises(ConfigurationError, match="fake"):
        Harness(config=Settings(), channels=[Fake(), Fake()])


async def test_the_chat_delivers_through_the_users_run_in_its_thread() -> None:
    """A fired run is in the user's thread: the thread stream delivers it to an open chat, and
    the conversation keeps its reply — the chat's channel sends nothing of its own, so the
    user does not read it twice; the run is checked to be theirs, in that thread."""
    async with Harness(config=Settings()) as h:
        agent = h.wrap(answer, id="desk")
        result = await agent.run("check stock", user="ada", thread=THREAD)
        record = await h.runs.get(result.run_id)
        assert record is not None
        about = note(tenant=record.tenant_id, run_id=record.run_id)
        assert await h.notify(about) == ["chat"]
        assert await h.notify(about) == []
        # the thread may be left out: the run's own is the chat's
        assert await h.notify(
            note("k-2", tenant=record.tenant_id, run_id=record.run_id, thread="")
        ) == ["chat"]
        anywhere = Notification(
            kind=Kind.RESULT,
            tenant=record.tenant_id,
            user="ada",
            text="Stock of SKU-1 is 4.",
            key="k-3",
            run_id=record.run_id,
        )
        assert await h.notify(anywhere) == ["chat"]


@pytest.mark.parametrize(
    ("changes", "said"),
    [
        ({"run_id": "run_missing"}, "no run run_missing"),
        ({"run_id": ""}, "about no run"),
        ({"user": "bob"}, "not bob's"),
        ({"thread": "chat-2"}, "not in thread chat-2"),
    ],
)
async def test_the_chat_does_not_deliver_what_is_not_the_users_run_in_its_thread(
    changes: dict[str, str], said: str
) -> None:
    async with Harness(config=Settings()) as h:
        agent = h.wrap(answer, id="desk")
        result = await agent.run("check stock", user="ada", thread=THREAD)
        record = await h.runs.get(result.run_id)
        assert record is not None
        about = note(**{"tenant": record.tenant_id, "run_id": record.run_id, **changes})
        with pytest.raises(Undelivered, match=said):
            await ChatChannel().deliver(about, h)
        assert await h.notify(about) == []


async def test_the_chat_does_not_deliver_a_run_with_no_thread() -> None:
    """A run started in the store with no thread (not the agent's: its runs have one, their
    own at least) is in no chat."""
    async with Harness(config=Settings()) as h:
        start = RunStart(run_id="run_x", tenant_id="t", agent_id="desk", user_id="ada", input="hi")
        await h.runs.start(start)
        with pytest.raises(Undelivered, match="in no thread"):
            await ChatChannel().deliver(note(run_id="run_x", thread=""), h)


def test_the_kinds_of_notification() -> None:
    assert [k.value for k in Kind] == ["result", "failure", "paused", "approval"]
