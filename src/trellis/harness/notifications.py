"""One way to tell a user what happened while they were away — a fired run's result, a failure,
a schedule paused, a run waiting for their approval — whatever carries it: a :class:`Channel`
delivers one :class:`Notification`, and ``Harness.notify`` (:func:`notify`) sends it to the
user's channels among those the harness has (``Harness(channels=)``: those the company
connected; the chat's, :class:`ChatChannel`, by default).

* **Whose channels.** ``Harness(user_channels=)`` (:data:`ChannelsHook`), the company's code as
  ``schedule_limits=`` is, names the channels each user gets for each notification (by its
  kind, too). None given, or it says ``None`` or fails (logged): the chat only, never every
  channel. A name the harness has no channel of is logged and skipped.

* **At most once per channel.** A notification's ``key`` is stable (the same event, the same
  key): ``notify`` delivers it to a channel only when that channel was not given it already in
  this process — a second call (a retry, a resumed run, two calls at once) sends nothing
  again. A channel that raises did not deliver it: logged, and the next call tries that channel
  again; a channel that cannot tell whether it delivered returns, as delivered. The last
  :data:`KEPT` keys are kept; a channel that reaches an outside system passes ``key`` on to it
  as its idempotency key, so other processes do not send it twice either.
* **The chat.** A notification is about a run of the user's chat thread (a fired run is the
  same agent, user and thread: ``schedules``): that run is the chat's delivery — the thread
  stream (``agui.threads``) delivers its reply, its pause or its error to an open chat, a quiet
  one (``NO_REPORT``) left out, and the conversation keeps its reply — so the chat's channel
  sends nothing of its own, which would tell the user twice. It checks the run is the user's,
  in that thread; one that is not, or no run, is :class:`Undelivered`.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

from trellis.harness.identity import Identity

if TYPE_CHECKING:
    from trellis.harness.harness import Harness

log = logging.getLogger("trellis.notifications")

#: The chat's channel's name: what a user gets unless the company's routing says otherwise.
CHAT: Final = "chat"
#: The (channel, tenant, key) delivered that one process keeps, the most recent.
KEPT: Final = 10_000


class Kind(StrEnum):
    """What happened."""

    #: a fired run's result
    RESULT = "result"
    #: a run that failed
    FAILURE = "failure"
    #: a schedule paused (its runs failed in a row, or it waits to be kept)
    PAUSED = "paused"
    #: a run waiting for the user's approval
    APPROVAL = "approval"


@dataclass(frozen=True)
class Notification:
    """One thing to tell one user: what happened, for whom, what it is about, in short."""

    kind: Kind
    tenant: str
    user: str
    #: a short text for the user, the facts of what happened
    text: str
    #: stable for the same event (``trellis.contracts.stable_id`` of what it is about): what
    #: it is delivered at most once per channel by
    key: str
    thread: str | None = None
    run_id: str | None = None
    schedule_id: str | None = None


class Undelivered(Exception):
    """A channel did not deliver a notification: why."""


@runtime_checkable
class Channel(Protocol):
    """Delivers one notification: returns once it did (or cannot tell), raises when it did not.
    ``name`` is the channel's, one per harness: what at-most-once is kept by."""

    name: str

    async def deliver(self, notification: Notification, harness: Harness) -> None: ...


class ChatChannel:
    """The user's chat thread: delivered by the run the notification is about."""

    name = CHAT

    async def deliver(self, notification: Notification, harness: Harness) -> None:
        if not notification.run_id:
            raise Undelivered("the chat delivers a run of its thread: this is about no run")
        record = await harness.runs.get(notification.run_id, tenant=notification.tenant)
        if record is None:
            raise Undelivered(f"no run {notification.run_id} in tenant {notification.tenant}")
        if Identity.of(record).user != notification.user:
            raise Undelivered(f"run {record.run_id} is not {notification.user}'s")
        if record.thread_id is None:
            raise Undelivered(f"run {record.run_id} is in no thread")
        if notification.thread and notification.thread != record.thread_id:
            raise Undelivered(f"run {record.run_id} is not in thread {notification.thread}")


#: ``async (user, tenant, notification) -> names | None`` (``Harness(user_channels=)``): the
#: names of the channels the company gives ``user`` for this notification; ``None``: the chat.
ChannelsHook = Callable[[str, str, Notification], Awaitable[Sequence[str] | None]]


async def routed(harness: Harness, notification: Notification) -> list[Channel]:
    """The harness's channels the notification's user gets, in the harness's order: those its
    ``user_channels`` names — one the harness does not have skipped, logged — else (no routing,
    it says nothing, or it fails: logged) the chat's."""
    names: Sequence[str] | None = None
    if harness.user_channels is not None:
        try:
            names = await harness.user_channels(
                notification.user, notification.tenant, notification
            )
        except Exception as exc:
            log.warning("notification %s: no channels routed: %s", notification.key, exc)
    wanted = set(names) if names is not None else {CHAT}
    unknown = wanted - {c.name for c in harness.channels}
    if unknown and names is not None:
        log.warning("notification %s: no channel %s here", notification.key, sorted(unknown))
    return [c for c in harness.channels if c.name in wanted]


async def notify(harness: Harness, notification: Notification) -> list[str]:
    """``notification`` delivered to each of the user's channels (:func:`routed`) that was not
    given it yet: the names of those that took it now."""
    took: list[str] = []
    for channel in await routed(harness, notification):
        sent = (channel.name, notification.tenant, notification.key)
        if sent in harness.notified:
            continue  # given it already, or being given it now
        harness.notified.add(sent)
        try:
            await channel.deliver(notification, harness)
        except Exception as exc:
            harness.notified.discard(sent)  # not delivered: the next call tries again
            log.warning(
                "notification %s (%s) not delivered by %s: %s",
                notification.key,
                notification.kind,
                channel.name,
                exc,
            )
            continue
        took.append(channel.name)
    return took


__all__ = [
    "CHAT",
    "KEPT",
    "Channel",
    "ChannelsHook",
    "ChatChannel",
    "Kind",
    "Notification",
    "Undelivered",
    "notify",
    "routed",
]
