"""Scheduling intent: does the agent schedule when it is asked to, and only then, at the right
time, and do the runs a schedule fires report what they should, once?

Each case is one chat (one or more turns) with the retail tools (``retail.py``) and the
built-in schedule tools (``trellis.harness.schedules``), ``ReAct`` on the in-process run store,
memory off (``--adapter``: any framework the ``complex`` suite wraps). Categories:

* ``schedule`` — a request to schedule (daily, weekdays, weekly, monthly, once, relative,
  with and without a condition, Hindi/English, a skill that asks for one, the user's zone
  from the app): exactly one schedule made, firing at the expected instants (its cron read in
  its zone, so ``0 9 * * *`` in Asia/Kolkata equals ``30 3 * * *`` in UTC), ``once`` and the
  condition as asked;
* ``no_schedule`` — a question that mentions a time or a habit: no schedule tool written to,
  and the question answered (the figure, from the tools);
* ``manage`` — list, move, pause, resume, cancel one, cancel all (asked to confirm first) and
  a reference that fits two schedules (asked which);
* ``fired`` — a seeded schedule fired by the worker: a condition met is reported with the
  figure, one not met answers ``NO_REPORT``, one met twice is reported once, and a fired run
  makes no schedule;
* ``limits`` — over the per-user cap or too frequent: one clear sentence, no retry loop;
  at the user's limit or the organisation's: which limit, the number and what the user can do,
  in plain words (no cron, no error code);
* ``check`` / ``cannot`` — a condition checked once before it is scheduled; one no tool can
  check, or an ask too vague to schedule;
* ``explain`` — a schedule whose runs went before the chat (:class:`Past`: failed the way the
  harness records a failure, quiet, reported, a day ago): why it fails, how often, when it last
  ran, whether it is on and why it is paused, why it said nothing, what it found — the record's
  facts, nothing made up;
* ``ends`` — an end date set, moved and removed;
* ``hard`` — mixed languages, typos, month, year and leap-day edges, DST weeks, the n-th weekday,
  hours in a window, a time in another zone, a time gone today, a habit only, a watch no tool
  can do;
* ``facts`` — questions about schedules whose answers are facts of this harness (the device
  off, holidays, every minute, who sees them, the zone): true, or "not known", never made up.

A case the harness has no feature for yet is written to the behaviour wanted and listed in
:data:`PENDING` (with the feature), reported apart.

Every check is exact (the store's schedules after the chat, the calls made, the figures in the
answer). Run::

    python -m benchmark.cost.schedules --model claude-haiku-5-5 --max-usd 0.5 --out build/sched
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

from croniter import croniter

from benchmark.cost import prices
from benchmark.cost import retail as r
from benchmark.cost import run as runner
from benchmark.cost.complex import ADAPTERS, bound_of, target_of
from trellis import Hooks
from trellis.contracts import (
    Interrupt,
    RunStatus,
    Schedule,
    ScheduleSpec,
    ToolCall,
    ToolOutcome,
)
from trellis.harness import pipeline, snapshots
from trellis.harness import schedules as harness_schedules
from trellis.harness.frontend import Frontend
from trellis.harness.runs import THREAD, LocalRuns
from trellis.harness.schedules import (
    NO_REPORT,
    REPORT,
    REPORTS,
    SCHEDULE_TOOLS,
    TASK,
    ScheduleLimits,
    _ending,
    _every,
    _input,
)
from trellis.runs import ConflictError, ScheduleLimitError

#: The categories, in the order they are reported.
CATEGORIES: Final = (
    "schedule",
    "no_schedule",
    "manage",
    "fired",
    "limits",
    "check",
    "cannot",
    "explain",
    "ends",
    "hard",
    "facts",
)
#: How many fire instants of a recurring schedule are compared.
INSTANTS: Final = 4
#: The tools that change a schedule (every other schedule tool only reads).
READS: Final = frozenset({"list_schedules"})
SKU: Final = "SKU-00012"
OTHER_SKU: Final = "SKU-00031"


@dataclass(frozen=True)
class Seed:
    """A schedule the user already has, as ``schedule_task`` makes it."""

    task: str
    when: str
    condition: str | None = None
    #: its zone (else the case's: the app's zone when it was set)
    timezone: str | None = None
    enabled: bool = True
    #: its last day (ISO date), kept as the harness keeps an end (``end_at``)
    until: str | None = None


@dataclass(frozen=True)
class Past:
    """A run of a seed that went before the chat: fired by the worker, failing with ``error``
    (raised as the run starts, as an ``on_run_start`` hook ends a run: the harness records it
    as it records any failure — the run ``ERROR`` with its error, its turn in the chat, the
    schedule's ``trellis_failures``, paused at the third), else run by the model; ``days_ago``
    moves the run and the schedule's last fire that many days back in the run store."""

    error: str | None = None
    days_ago: int = 0
    seed: int = 0


@dataclass
class Chat:
    """What a case's chat did: each turn's answer and calls, the user's schedules before and
    after, and each fired run's output."""

    now: datetime
    #: the user's zone as their app says it
    zone: str | None = None
    answers: list[str] = field(default_factory=list)
    #: each turn's question with options the run paused on (their labels), else ``None``
    asked: list[list[str] | None] = field(default_factory=list)
    calls: list[list[tuple[str, dict[str, Any], str]]] = field(default_factory=list)
    seeded: list[Schedule] = field(default_factory=list)
    after: list[Schedule] = field(default_factory=list)
    fired: list[str] = field(default_factory=list)
    fired_calls: list[str] = field(default_factory=list)
    #: the shortest repeat the run's tools allow, in seconds (the deployment's, or the user's)
    min_interval: float | None = None
    #: how many schedules the user may have (the deployment's, or the user's)
    max_per_user: int | None = None

    def made(self) -> list[Schedule]:
        before = {s.schedule_id for s in self.seeded}
        return [s for s in self.after if s.schedule_id not in before]

    def written(self) -> list[str]:
        return [n for turn in self.calls for n, a, _ in turn if writes(n, a)]

    def named(self, *names: str) -> list[tuple[str, dict[str, Any], str]]:
        return [c for turn in self.calls for c in turn if c[0] in names]


#: The schedule tools that write.
SCHEDULE_TOOLS_W: set[str] = {n for n in SCHEDULE_TOOLS if n not in READS}
#: Every schedule tool's name.
SCHEDULE_NAMES: set[str] = set(SCHEDULE_TOOLS)

Check = Callable[[Chat], list[str]]


def writes(name: str, args: dict[str, Any]) -> bool:
    """Whether a call writes to the user's schedules."""
    return name in SCHEDULE_TOOLS_W


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    turns: tuple[str, ...]
    check: Check
    #: the user's zone as the app sends it (AG-UI ``forwardedProps.timezone``); ``None``: not
    #: known
    zone: str | None = "Asia/Kolkata"
    seeds: tuple[Seed, ...] = ()
    #: how many times the first seed is fired after the chat
    fire: int = 0
    limits: ScheduleLimits | None = None
    #: what the app tells the model besides (an AG-UI context entry)
    context: tuple[dict[str, str], ...] = ()
    #: runs of the seeds before the chat, in order
    past: tuple[Past, ...] = ()
    #: the environment's memory service on: the chat is its transcript (a failed run's turn
    #: and its record are there, as the harness writes them; with memory off a failed run's
    #: turn is not read back)
    memory: bool = False
    #: the time it is in the chat (``snapshots.now`` and the schedule tools' clock); ``None``:
    #: now
    now: datetime | None = None
    #: the run store refuses a new schedule with this (agent-runs' own limits)
    refuse: ConflictError | None = None


# --------------------------------------------------------------------------- checks


def instants(cron: str, zone: str, now: datetime, n: int = INSTANTS) -> list[datetime]:
    cursor = croniter(cron, now.astimezone(ZoneInfo(zone)))
    return [cursor.get_next(datetime).astimezone(UTC) for _ in range(n)]


def one_made(chat: Chat) -> tuple[Schedule | None, list[str]]:
    made = chat.made()
    if len(made) != 1:
        return None, [f"{len(made)} schedules made, not 1"]
    return made[0], []


def scheduled(
    cron: str | None = None,
    zone: str | None = None,
    *,
    once: bool = False,
    condition: Sequence[str] | None = None,
    at: Callable[[datetime, str], datetime] | None = None,
    within: tuple[timedelta, timedelta] | None = None,
    weekday: int | None = None,
    checked: Sequence[str] | None = None,
) -> Check:
    """One schedule made: firing at ``cron``'s instants in ``zone`` (else the user's, else
    UTC) — or first at ``at(now, zone)``, or within ``within`` of now, or weekly on
    ``weekday``, any hour —, ``once`` as said, a condition naming each of ``condition``
    (``None``: no condition); ``checked``: the turn that scheduled it checked it first with
    another tool and told the user what it found (each of ``checked``)."""

    def check(chat: Chat) -> list[str]:
        z = zone or chat.zone or "UTC"
        s, problems = one_made(chat)
        if s is None:
            return problems
        if checked is not None:
            problems += checked_first(chat, checked)
        about = s.metadata[TASK]
        got = instants(s.cadence, s.timezone, chat.now)
        said = f"'{s.cadence}' {s.timezone}"
        if cron is not None:
            n = 1 if once else INSTANTS
            if got[:n] != instants(cron, z, chat.now, n):
                problems.append(f"fires at {said}, not '{cron}' {z}")
        if at is not None and got[0] != at(chat.now, z).astimezone(UTC):
            problems.append(f"first fire {got[0]:%a %d %H:%M}Z ({said}), not {at(chat.now, z)}")
        if within is not None and not within[0] <= got[0] - chat.now <= within[1]:
            problems.append(f"first fire {got[0] - chat.now} from now ({said})")
        if weekday is not None:
            local = [g.astimezone(ZoneInfo(z)) for g in got]
            if any(t.weekday() != weekday for t in local) or got[1] - got[0] != timedelta(7):
                problems.append(f"not weekly on day {weekday}: {said}")
        if bool(about.get("once")) != once:
            problems.append(f"once={about.get('once')}")
        cond = about.get("condition")
        if condition is None and cond:
            problems.append(f"a condition nobody asked for: {cond!r}")
        if condition is not None:
            missing = [w for w in condition if not cond or w.lower() not in cond.lower()]
            if missing:
                problems.append(f"condition {cond!r} lacks {missing}")
        return problems

    return check


def checked_first(chat: Chat, says: Sequence[str]) -> list[str]:
    """The turn that called ``schedule_task`` called another tool before it, and its answer
    says each of ``says`` (what the check found)."""
    for answer, turn in zip(chat.answers, chat.calls, strict=True):
        names = [n for n, _, _ in turn]
        if "schedule_task" in names:
            before = names[: names.index("schedule_task")]
            problems = [] if set(before) - SCHEDULE_NAMES else ["scheduled without checking"]
            return problems + said_all(answer, says)
    return ["no schedule_task call"]


#: What the retail tools can watch, as an answer names it (the tools' own subjects).
WATCHABLE: Final = (
    r"\bstock",
    r"price[ _]history|our (own )?price",
    r"\bsales\b|revenue",
    r"supplier",
    r"forecast",
    r"margin",
    r"warehouse",
    r"\baudit",
)


def cannot(chat: Chat) -> list[str]:
    """Nothing scheduled; the answer says it cannot and suggests at least two things the
    tools can watch."""
    problems = [f"wrote {n}" for n in chat.written()]
    if chat.made():
        problems.append("made a schedule")
    text = chat.answers[-1].lower()
    found = [w for w in WATCHABLE if re.search(w, text)]
    if len(found) < 2:
        problems.append(f"suggests {len(found)} things the tools can watch, not 2+")
    return problems


def clarifies(chat: Chat) -> list[str]:
    """A vague ask: asked back (choices to pick, or a question) before anything is
    scheduled; one schedule once the user picked (a second turn)."""
    problems = [] if asked_in(chat, 0) else ["did not ask or propose"]
    if [n for n, a, _ in chat.calls[0] if writes(n, a)]:
        problems.append("scheduled before asking")
    want = 1 if len(chat.answers) > 1 else 0
    if len(chat.made()) != want:
        problems.append(f"{len(chat.made())} schedules, not {want}")
    return problems


def updated_twin(condition: str) -> Check:
    """A schedule like the user's: they were asked with options, the recommended one (update)
    first; picked, the one they had now has ``condition``, and there is no second."""

    def check(chat: Chat) -> list[str]:
        options = chat.asked[0] or []
        problems = [] if options else ["no options asked"]
        if options and "recommended" not in options[0].lower():
            problems.append(f"first option {options[0]!r} is not the recommended update")
        problems += ["made a second schedule"] if chat.made() else []
        kept = seeded_by(chat, 0)
        if kept is None or condition not in (kept.metadata[TASK].get("condition") or ""):
            problems.append("the schedule was not updated")
        return problems

    return check


def already_set(chat: Chat) -> list[str]:
    """The same schedule asked for again: nothing new, nothing asked, said so."""
    problems = ["made a schedule"] if chat.made() else []
    problems += ["asked with options"] if any(chat.asked) else []
    problems += said_all(chat.answers[-1], ("already",))
    return problems


def and_says(check: Check, *says: str) -> Check:
    def both(chat: Chat) -> list[str]:
        return check(chat) + said_all(chat.answers[-1], says)

    return both


def already_true(says: str, condition: str) -> Check:
    """The condition holds now: the first answer says so (the figure), and a schedule is
    there only after the user wants more alerts (the second turn)."""

    def check(chat: Chat) -> list[str]:
        problems = said_all(chat.answers[0], (says,))
        made = chat.made()
        if len(made) != 1:
            problems.append(f"{len(made)} schedules after the user asked for alerts, not 1")
        elif condition not in (made[0].metadata[TASK].get("condition") or ""):
            problems.append(f"condition {made[0].metadata[TASK].get('condition')!r}")
        return problems

    return check


def or_asks_zone(check: Check) -> Check:
    """``check``, or — the user's zone not known — nothing written and their zone asked."""

    def either(chat: Chat) -> list[str]:
        problems = check(chat)
        text = chat.answers[-1].lower()
        asked = asked_in(chat) and ("zone" in text or "utc" in text)
        return [] if problems and asked and not chat.written() else problems

    return either


def says_any(check: Check, *alternatives: str) -> Check:
    """``check``, and the last answer says one of ``alternatives`` (the time it chose, in words
    the user cannot misread)."""

    def both(chat: Chat) -> list[str]:
        text = chat.answers[-1].lower()
        said = any(a.lower() in text for a in alternatives)
        return check(chat) + ([] if said else [f"answer says none of {list(alternatives)}"])

    return both


def or_asks(check: Check, *about: str) -> Check:
    """``check``, or nothing written and the user asked (options, or a question naming one of
    ``about``): the model may settle what the words leave open itself, or ask."""

    def either(chat: Chat) -> list[str]:
        problems = check(chat)
        options = (chat.asked[-1] if chat.asked else None) or []
        # "every-other-week" names "every other": words compared however they are joined
        text = (chat.answers[-1] + " " + " ".join(options)).lower().replace("-", " ")
        asked = asked_in(chat) and any(a.lower().replace("-", " ") in text for a in about)
        return [] if problems and asked and not chat.made() else problems

    return either


def covered(cron: str) -> Check:
    """The schedules made fire at ``cron``'s instants in the user's zone together (one
    schedule, or one a day of the week)."""

    def check(chat: Chat) -> list[str]:
        made = chat.made()
        if not made:
            return ["0 schedules made"]
        zone = chat.zone or "UTC"
        got = sorted({t for s in made for t in instants(s.cadence, s.timezone, chat.now)})
        want = instants(cron, zone, chat.now)
        return [] if got[: len(want)] == want else [f"fire at {got[:4]}, not '{cron}' {zone}"]

    return check


def hours(first: int, last: int, *, once: bool = False) -> Check:
    """One schedule (``once`` as said) whose first run is between ``first`` and ``last``
    o'clock in the user's zone: a time of day the user named only in words."""

    def check(chat: Chat) -> list[str]:
        s, problems = one_made(chat)
        if s is None:
            return problems
        local = instants(s.cadence, s.timezone, chat.now)[0].astimezone(
            ZoneInfo(chat.zone or "UTC")
        )
        if not first <= local.hour <= last:
            problems.append(f"first run at {local:%H:%M}, not between {first}:00 and {last}:00")
        if bool(s.metadata[TASK].get("once")) != once:
            problems.append(f"once={s.metadata[TASK].get('once')}")
        return problems

    return check


def alternate_weeks(chat: Chat) -> list[str]:
    """Every other Monday, which a cron cannot say: weekly on Monday with a condition that
    skips the other weeks, or weekly and the answer saying it is every week instead."""
    weekly = scheduled(weekday=0)
    s, problems = one_made(chat)
    if s is None:
        return problems
    condition = (s.metadata[TASK].get("condition") or "").lower()
    if "week" in condition:  # the run checks the week itself
        return [p for p in weekly(chat) if not p.startswith("a condition nobody asked for")]
    said = ("every week", "weekly", "every monday", "can't", "cannot", "not able", "every two")
    return says_any(weekly, *said)(chat)


def ends(check: Check) -> Check:
    """``check``, and the schedule made keeps an end (``end_at``, or the task's ``until``)."""

    def both(chat: Chat) -> list[str]:
        problems = check(chat)
        made = chat.made()
        kept = made and (getattr(made[0], "end_at", None) or made[0].metadata[TASK].get("until"))
        return problems + ([] if kept else ["no end date kept"])

    return both


def unscheduled(*says: str, tools: Sequence[str] = ()) -> Check:
    """No schedule written to; the answer says each of ``says`` (a figure: digits compared
    without separators) and the turn called each of ``tools``."""

    def check(chat: Chat) -> list[str]:
        problems = [f"wrote {n}" for n in chat.written()]
        if chat.made():
            problems.append("made a schedule")
        problems += said_all(chat.answers[-1], says)
        called = {n for turn in chat.calls for n, _, _ in turn}
        problems += [f"did not call {t}" for t in tools if t not in called]
        return problems

    return check


def said_all(answer: str, says: Sequence[str]) -> list[str]:
    flat = re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", answer).lower()
    return [f"answer lacks {s!r}" for s in says if s.lower() not in flat]


def asks(text: str) -> bool:
    return "?" in text


def asked_in(chat: Chat, turn: int = -1) -> bool:
    """Whether the turn asked the user: options to pick (the run paused) or a question."""
    return bool(chat.asked and chat.asked[turn]) or asks(chat.answers[turn])


def labels(interrupt: Interrupt) -> list[str]:
    """The options as a person reads them, the recommended one marked."""
    return [
        o if isinstance(o, str) else (o.label or o.value) + (" (Recommended)" * bool(o.recommended))
        for o in interrupt.options
    ]


def said_of(text: str) -> str:
    """What a turn says when no question waits: ``@n|words`` says the words."""
    return text.split("|", 1)[1] if text.startswith("@") else text


def picks(interrupt: Interrupt, text: str) -> Any:
    """The answer a turn gives the question the run waits on: ``@n|...`` picks option ``n``,
    other words are the answer as said."""
    if text.startswith("@") and interrupt.options:
        n = int(text[1:].split("|", 1)[0])
        option = interrupt.options[min(n, len(interrupt.options) - 1)]
        return option if isinstance(option, str) else option.value
    return said_of(text)


def by_task(chat: Chat, words: str) -> Schedule | None:
    for s in chat.after:
        if words.lower() in s.metadata[TASK]["task"].lower():
            return s
    return None


def seeded_by(chat: Chat, n: int) -> Schedule | None:
    sid = chat.seeded[n].schedule_id
    return next((s for s in chat.after if s.schedule_id == sid), None)


def changed(n: int, *, cron: str | None = None, enabled: bool | None = None,
            condition: str | None = None, untouched: Sequence[int] = ()) -> Check:  # fmt: skip
    """Seed ``n`` changed as said, the ``untouched`` seeds as they were, nothing made."""

    def check(chat: Chat) -> list[str]:
        s = seeded_by(chat, n)
        if s is None:
            return [f"seed {n} is gone"]
        problems = ["made a schedule"] if chat.made() else []
        if cron is not None:
            want = instants(cron, chat.seeded[n].timezone, chat.now)
            if instants(s.cadence, s.timezone, chat.now) != want:
                problems.append(f"fires at '{s.cadence}' {s.timezone}, not '{cron}'")
        if enabled is not None and s.enabled != enabled:
            problems.append(f"enabled={s.enabled}")
        if condition is not None and condition not in (s.metadata[TASK].get("condition") or ""):
            problems.append(f"condition {s.metadata[TASK].get('condition')!r}")
        for u in untouched:
            kept = seeded_by(chat, u)
            if kept is None or kept.model_dump() != chat.seeded[u].model_dump():
                problems.append(f"seed {u} changed")
        return problems

    return check


def cancelled(*gone: int, kept: Sequence[int] = (), ask_first: bool = False) -> Check:
    def check(chat: Chat) -> list[str]:
        problems = [f"seed {n} not cancelled" for n in gone if seeded_by(chat, n) is not None]
        problems += [f"seed {n} cancelled" for n in kept if seeded_by(chat, n) is None]
        if ask_first:
            first = [n for n, a, _ in chat.calls[0] if writes(n, a)]
            if first or not asked_in(chat, 0):
                problems.append("did not confirm before cancelling all")
        return problems

    return check


def asked_which(chat: Chat) -> list[str]:
    """Asked which (nothing written before), then the one picked changed: exactly one of the
    two similar seeds."""
    problems = [] if asked_in(chat, 0) else ["did not ask which"]
    if not chat.asked[0] and [n for n, a, _ in chat.calls[0] if writes(n, a)]:
        problems.append("wrote before asking")
    gone = [n for n in (0, 1) if seeded_by(chat, n) is None]
    paused = [n for n in (0, 1) if (s := seeded_by(chat, n)) is not None and not s.enabled]
    if len(gone) + len(paused) != 1:
        problems.append(f"{len(gone)} cancelled and {len(paused)} paused, not one")
    return problems


def listed(*says: str) -> Check:
    def check(chat: Chat) -> list[str]:
        problems = [f"wrote {n}" for n in chat.written()]
        return problems + said_all(chat.answers[-1], says)

    return check


def reports(*outputs: str | None, says: Sequence[str] = ()) -> Check:
    """Each fired run's output: ``NO_REPORT`` where ``None`` is given... a report (not
    ``NO_REPORT``, saying each of ``says``) where ``"report"``; no schedule made or written to
    by a fired run."""

    def check(chat: Chat) -> list[str]:
        problems = [f"fired run called {n}" for n in chat.fired_calls if n in SCHEDULE_NAMES]
        if len(chat.after) > len(chat.seeded):
            problems.append("a fired run made a schedule")
        for i, (want, got) in enumerate(zip(outputs, chat.fired, strict=True)):
            if want is None and got != NO_REPORT:
                problems.append(f"fire {i}: reported {got[:80]!r}")
            if want == "report":
                if got == NO_REPORT:
                    problems.append(f"fire {i}: NO_REPORT")
                else:
                    problems += [f"fire {i}: {p}" for p in said_all(got, says)]
        return problems

    return check


def limited(*words: str) -> Check:
    """Nothing made, schedule_task tried at most twice, and the answer says the limit."""

    def check(chat: Chat) -> list[str]:
        problems = ["made a schedule"] if chat.made() else []
        tries = len(chat.named("schedule_task", "manage_schedules", "CronCreate"))
        if tries > 2:
            problems.append(f"schedule_task tried {tries} times")
        text = chat.answers[-1].lower()
        if not any(w in text for w in words):
            problems.append(f"answer does not say the limit ({'/'.join(words)})")
        return problems

    return check


# --------------------------------------------------------------------------- explain, facts

#: What the failed runs' record says of why: the sign-in expired.
SIGN_IN: Final = (r"sign.?in|log.?in|signed out", r"expir")
#: Causes no record of these chats holds: an answer that gives one made it up.
MADE_UP_CAUSES: Final = (
    r"rate.?limit",
    r"time(d)?.?out",
    r"network",
    r"server (error|down|issue|problem)",
    r"quota",
    r"outage",
    r"crash",
    r"\bbug\b",
    r"out of (credits|budget|money)",
    r"password (changed|reset)",
)
#: A failure claimed (for a schedule none of whose runs failed).
FAILED_CLAIMS: Final = (
    *SIGN_IN,
    r"\bfailed (\d+|once|twice|two|three)",
    r"\b\d+ (failed|failures|errors)",
    r"\b(after|because of|due to) (\w+ ){0,2}(failures|errors|failed)",
)
#: Said paused.
PAUSED: Final = r"paus|turned off|switched off|\boff\b|not (running|active|on)|disabled|stopped"
#: Said true of a schedule paused with no failure: the record says no more.
NOT_RECORDED: Final = (
    r"by hand|manually|you paused|someone paused|was paused by|turned (it )?off|"
    r"no (failures|errors|failed runs)|(hasn't|has not|didn't|did not|never) fail|"
    r"not (because of|due to) (a |any )?(failure|error)|(don't|do not) know|not (known|recorded)|"
    r"no record|doesn't say|does not say"
)
#: Said of a condition reported already: quiet until it stops holding.
REPORTED_ONCE: Final = (
    r"already (told|reported|alerted|notified|sent|let you know)|reported (it )?(already|once|"
    r"before|earlier)|once (while|until|per)|until (it|the (stock|condition|level)) (clears|"
    r"stops|goes|rises|is no longer)|stops? (holding|being (true|under|below))|"
    r"(goes|rises|is) back (up|above|over)"
)
#: A schedule runs without the user's device, and its reply waits in the chat.
RUNS_WITHOUT_YOU: Final = (
    r"still run|will run|runs (anyway|regardless|as (usual|normal|scheduled))|"
    r"(doesn't|does not|don't|do not) (need|depend|require)|not (on|from) your "
    r"(laptop|computer|device)|server|cloud|independent"
)
IN_CHAT: Final = r"chat|conversation|thread|when you (open|come back|return|log)"
NEEDS_DEVICE: Final = (
    r"(?<!n't )(?<!not )\bneeds? your (laptop|computer|device|phone)",
    r"(keep|leave) your (laptop|computer|device|phone) (on|open|running|awake)",
    r"(won't|will not|doesn't|does not|can't|cannot) run (if|when|while) your "
    r"(laptop|computer|device)",
    r"(will be|gets?|is) (skipped|missed)",
)
#: A cron has no holidays: it runs on every day it names.
RUNS_ON_HOLIDAYS: Final = (
    r"\byes\b|still run|(?<!n't )(?<!not )\bruns? (on|every|regardless|as usual)|"
    r"(doesn't|does not|won't|will not|can't|cannot) (skip|know|tell)|"
    r"no (notion|idea|knowledge|concept|calendar|list) of (public |bank )?holidays|"
    r"not aware of (public |bank )?holidays|(isn't|is not|aren't|are not) (aware|skipped)"
)
SKIPS_HOLIDAYS: Final = (
    r"(?<!n't )(?<!not )\bskips? (public |bank )?holidays",
    r"(won't|will not|doesn't|does not) run on (public |bank )?holidays",
    r"holidays are (skipped|excluded)",
    r"automatically (skip|pause)",
    r"holiday calendar (is|it) (used|follows)",
)
#: Each user sees only their own schedules (docs/runs.md, "Whose"); nothing says more.
ONLY_YOU: Final = (
    r"only you|just you|no one else|nobody else|only (be )?(visible|available) to you|"
    r"other (users|people) (can't|cannot|can not|don't|do not|won't)|yours alone|only yours|"
    r"private to you"
)
OTHERS_SEE: Final = (
    r"(?<!whether )(?<!whether your )(?<!if )(?<!if your )\b(your )?(team|teammates|colleagues|"
    r"coworkers|admins?|administrators?|managers?|everyone|anyone in your (organi[sz]ation|"
    r"company))( members)? (can|could|will|also) (also )?(see|view|access)",
    r"(shared|visible) (with|to) (your )?(team|organi[sz]ation|company|everyone|admins?)",
)
KOLKATA: Final = r"kolkata|\bist\b|india"
IN_NEW_YORK: Final = (
    r"(?<!not )(?<!n't )\b(is|runs|set|scheduled) (in|on|for|to) (the )?(new york|eastern|"
    r"america/new_york|est\b|edt\b|utc\b(?! ?\+))",
)
#: Said of a schedule with no run yet, and a run claimed for one.
NOT_RUN_YET: Final = (
    r"(hasn't|has not|not yet|never|didn't|did not) (run|ran|been run|fired)|no (runs?|record)"
    r"( of it)? yet|no (past )?runs|first run"
)
RAN_CLAIMS: Final = (r"(?<!not )(?<!never )\blast ran (on|at|today|yesterday)",)
#: Said: it runs again once resumed.
RESUME: Final = r"resume|turn (it )?(back )?on|switch (it )?(back )?on|re-?enable|unpause"
#: Results said to go anywhere but the chat (no such channel in this harness).
ELSEWHERE: Final = (
    r"(?<!not )(?<!n't )\b(by|via|in an?|to your) (e-?mail|sms|text message|push notification|"
    r"whatsapp|slack)",
)
#: Said no.
NO: Final = (
    r"\bno\b|can't|cannot|can not|not (possible|allowed|able)|isn't (possible|allowed)|"
    r"too (often|frequent)|won't"
)
#: A cron as a person would see it, and an error code.
CRON_SHOWN: Final = r"(?<![\w:.])(?:[\d*][\d*/,#L-]*\s+){4}[\d*][\d*/,#L-]*(?![\w:])"
CODE_SHOWN: Final = r"(?<![/\w])[A-Za-z]+(?:_[A-Za-z]+)+\b|\b(400|401|403|404|409|422|429|503)\b"


def _flat(text: str) -> str:
    """``text`` lowercased, digits without thousands separators (as :func:`said_all`)."""
    return re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", text).lower()


def explains(*facts: str, wrong: Sequence[str] = MADE_UP_CAUSES) -> Check:
    """Nothing written; the last answer says each of ``facts`` (patterns) and none of
    ``wrong`` (a claim no record or doc holds: made up)."""

    def check(chat: Chat) -> list[str]:
        text = _flat(chat.answers[-1])
        problems = [f"wrote {n}" for n in chat.written()]
        problems += [f"answer lacks /{f}/" for f in facts if not re.search(f, text)]
        for w in wrong:
            if found := re.search(w, text):
                problems.append(f"a made-up claim: {found.group(0)!r}")
        return problems

    return check


_WORDS: Final = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                 "ten")  # fmt: skip
_NUMBERS: Final = {w: n for n, w in enumerate(_WORDS)}
_NUM: Final = r"(\d+|" + "|".join(_NUMBERS) + ")"
#: A count of failures said: not the rule's ("paused after 3 failures") nor one to come ("one
#: more failure").
_COUNTS: Final = (
    rf"(?<!after )(?<!at )\b{_NUM}\s+(?!(?:\w+\s+){{0,1}}(?:more|another|further)\b)"
    r"(?:\w+\s+){0,2}?(?:times?|failures?|failed runs?|attempts?|errors?)\b",
    rf"\bfailed\s+{_NUM}\b",
    r"\bfailed\s+(once|twice|thrice)\b",
)


def counted(n: int) -> Check:
    """Nothing written; every count of failures the last answer says is ``n``, and it says
    one."""
    once = {"once": 1, "twice": 2, "thrice": 3}

    def check(chat: Chat) -> list[str]:
        text = _flat(chat.answers[-1])
        problems = [f"wrote {w}" for w in chat.written()]
        said = [
            int(m) if m.isdigit() else _NUMBERS.get(m, once.get(m, -1))
            for pattern in _COUNTS
            for m in re.findall(pattern, text)
        ]
        if not said:
            return [*problems, "says no count"]
        return problems + [f"counts {c}, not {n}" for c in dict.fromkeys(said) if c != n][:1]

    return check


def time_said(text: str, at: datetime) -> bool:
    """Whether ``text`` says the clock time ``at``: 24-hour or 12-hour, the minutes said unless
    on the hour (then ``9 am`` will do)."""
    h12 = at.hour % 12 or 12
    half = "am" if at.hour < 12 else "pm"
    forms = {f"{at:%H:%M}", f"{at.hour}:{at:%M}", f"{h12}:{at:%M}", f"{h12}.{at:%M}"}
    if at.minute == 0:
        forms |= {f"{h12} {half}", f"{h12}{half}", f"{h12} {half[0]}.m."}
    return any(re.search(rf"(?<![\d:.]){re.escape(f)}(?![\d])", text) for f in forms)


def last_ran(n: int) -> Check:
    """Nothing written; the last answer says when seed ``n`` last ran (its clock time in its
    zone, as the run store has it)."""

    def check(chat: Chat) -> list[str]:
        seed = chat.seeded[n]
        if seed.last_fired_at is None:
            return [f"seed {n} has no last run"]
        at = seed.last_fired_at.astimezone(ZoneInfo(seed.timezone))
        problems = [f"wrote {w}" for w in chat.written()]
        if not time_said(chat.answers[-1].lower(), at):
            problems.append(f"answer lacks the last run's time ({at:%H:%M})")
        return problems

    return check


def from_record(*says: str, tools: Sequence[str] = ("stock_level",)) -> Check:
    """Nothing written; the last answer says each of ``says`` — what a past run found, read from
    its record in the chat: none of ``tools`` called again for it."""

    def check(chat: Chat) -> list[str]:
        problems = [f"wrote {w}" for w in chat.written()]
        called = {name for turn in chat.calls for name, _, _ in turn}
        problems += [f"fetched {t} again instead of reading the run" for t in tools if t in called]
        return problems + said_all(chat.answers[-1], says)

    return check


def shortest_said() -> Check:
    """Nothing written; the last answer says no, and the shortest repeat allowed in words (the
    deployment's ``schedule_min_interval``, as the tool's description says it)."""

    def check(chat: Chat) -> list[str]:
        assert chat.min_interval is not None
        problems = [f"wrote {w}" for w in chat.written()]
        problems += said_all(chat.answers[-1], (_every(chat.min_interval),))
        if not re.search(NO, _flat(chat.answers[-1])):
            problems.append("answer does not say no")
        return problems

    return check


def most_said() -> Check:
    """Nothing written; the last answer says how many schedules the user may have (the
    deployment's ``schedule_max_per_user``, or theirs)."""

    def check(chat: Chat) -> list[str]:
        assert chat.max_per_user is not None
        problems = [f"wrote {w}" for w in chat.written()]
        return problems + said_all(chat.answers[-1], (str(chat.max_per_user),))

    return check


def every(*checks: Check) -> Check:
    """Each of ``checks``: their problems together."""
    return lambda chat: [p for c in checks for p in c(chat)]


def one_of(*checks: Check) -> Check:
    """Any of ``checks`` (where the words allow more than one reading): the first's problems
    when none passes."""

    def check(chat: Chat) -> list[str]:
        found = [c(chat) for c in checks]
        return [] if any(not f for f in found) else found[0]

    return check


# --------------------------------------------------------------------------- change, ends


def _targets(args: dict[str, Any], sid: str) -> bool:
    return args.get("id") == sid or sid in (args.get("ids") or [])


def paused_then_resumed(n: int, untouched: Sequence[int] = ()) -> Check:
    """The first turn paused seed ``n`` (``enabled`` false), the second resumed it; it is on at
    the end, nothing made, the ``untouched`` seeds as they were."""

    def check(chat: Chat) -> list[str]:
        sid = chat.seeded[n].schedule_id
        problems = ["made a schedule"] if chat.made() else []
        for turn, want, said in ((0, False, "pause"), (1, True, "resume")):
            done = [
                a
                for name, a, _ in chat.calls[turn]
                if name == "update_schedule" and _targets(a, sid) and a.get("enabled") is want
            ]
            if not done:
                problems.append(f"turn {turn + 1} did not {said} it")
        problems += changed(n, enabled=True, untouched=untouched)(chat)
        return list(dict.fromkeys(problems))

    return check


def reports_as(n: int, report: str, untouched: Sequence[int] = ()) -> Check:
    """Seed ``n`` kept (its time and condition) and now reporting as ``report``
    (:data:`REPORTS`), nothing made, the ``untouched`` seeds as they were."""

    def check(chat: Chat) -> list[str]:
        s = seeded_by(chat, n)
        if s is None:
            return [f"seed {n} is gone"]
        cond = chat.seeded[n].metadata[TASK].get("condition") or ""
        problems = changed(n, cron=chat.seeded[n].cadence, untouched=untouched)(chat)
        if (s.metadata[TASK].get("condition") or "") != cond:
            problems.append(f"condition {s.metadata[TASK].get('condition')!r}, not {cond!r}")
        got = str(s.metadata.get(REPORT) or REPORTS[0])
        return problems + ([] if got == report else [f"reports {got}, not {report}"])

    return check


def asked_which_moved(targets: dict[int, str]) -> Check:
    """Two seeds fit the user's words: asked which (nothing written before), then exactly one of
    them fires at its ``targets`` cron, the other as it was; nothing made."""

    def check(chat: Chat) -> list[str]:
        problems = [] if asked_in(chat, 0) else ["did not ask which"]
        if not chat.asked[0] and [n for n, a, _ in chat.calls[0] if writes(n, a)]:
            problems.append("wrote before asking")
        problems += ["made a schedule"] if chat.made() else []
        moved = [n for n, cron in targets.items() if not changed(n, cron=cron)(chat)]
        if len(moved) != 1:
            return [*problems, f"{len(moved)} moved, not one"]
        problems += changed(moved[0], untouched=[n for n in targets if n != moved[0]])(chat)
        return problems

    return check


def end_of(schedule: Schedule) -> datetime | None:
    """When a schedule ends: its ``end_at`` (pending in the harness), else its task's
    ``until``; a day alone ends at the end of that day in its zone."""
    zone = ZoneInfo(schedule.timezone)
    end = getattr(schedule, "end_at", None) or schedule.metadata[TASK].get("until")
    if isinstance(end, datetime):
        return end if end.tzinfo else end.replace(tzinfo=zone)
    if not isinstance(end, str) or not end:
        return None
    try:
        if len(end) == len("2026-12-31"):
            day = date.fromisoformat(end) + timedelta(days=1)
            return datetime(day.year, day.month, day.day, tzinfo=zone)
        at = datetime.fromisoformat(end)
    except ValueError:
        return None
    return at if at.tzinfo else at.replace(tzinfo=zone)


def last_fire(schedule: Schedule, now: datetime, end: datetime) -> datetime | None:
    """The schedule's last run before ``end``, in its zone."""
    cursor, last = croniter(schedule.cadence, now.astimezone(ZoneInfo(schedule.timezone))), None
    for _ in range(5000):
        at = cursor.get_next(datetime)
        if at >= end:
            return last
        last = at
    return last


def ended_by(
    days: Callable[[datetime], tuple[date, date]],
    check: Check | None = None,
    *,
    seed: int | None = None,
) -> Check:
    """``check``; and the schedule made (or seed ``seed``, its time as it was) ends so that its
    last run falls on a day between ``days(now)`` (both in)."""

    def ends_check(chat: Chat) -> list[str]:
        problems = check(chat) if check is not None else []
        if seed is None:
            s, missing = one_made(chat)
            problems += [] if check is not None else missing
        else:
            s = seeded_by(chat, seed)
            problems += changed(seed, cron=chat.seeded[seed].cadence)(chat)
        if s is None:
            return problems
        end = end_of(s)
        if end is None:
            return [*problems, "no end date kept"]
        last = last_fire(s, chat.now, end)
        first, final = days(chat.now)
        if last is None:
            return [*problems, f"no run before its end {end:%Y-%m-%d}"]
        if not first <= last.date() <= final:
            problems.append(f"last run on {last:%Y-%m-%d}, not {first} to {final}")
        return problems

    return ends_check


def no_end(n: int) -> Check:
    """Seed ``n`` kept at its time, with no end."""

    def check(chat: Chat) -> list[str]:
        problems = changed(n, cron=chat.seeded[n].cadence)(chat)
        s = seeded_by(chat, n)
        if s is not None and end_of(s) is not None:
            problems.append("an end is still kept")
        return problems

    return check


# --------------------------------------------------------------------------- limits, hard


def plain_limit(*says: str, which: str, action: str, codes: Sequence[str] = ()) -> Check:
    """Nothing made, ``schedule_task`` tried at most twice; the last answer says each of
    ``says`` (the number), names the limit (``which``) and what the user can do (``action``),
    in plain words: no cron, no error code (any, or one of ``codes``)."""

    def check(chat: Chat) -> list[str]:
        problems = ["made a schedule"] if chat.made() else []
        tries = len(chat.named("schedule_task"))
        if tries > 2:
            problems.append(f"schedule_task tried {tries} times")
        answer = chat.answers[-1]
        text = _flat(answer)
        problems += said_all(answer, says)
        if not re.search(which, text):
            problems.append(f"answer does not name the limit (/{which}/)")
        if not re.search(action, text):
            problems.append(f"answer says no way on (/{action}/)")
        if re.search(CRON_SHOWN, answer):
            problems.append("answer shows a cron")
        if re.search(CODE_SHOWN, answer) or any(c.lower() in text for c in codes):
            problems.append("answer shows an error code")
        return problems

    return check


def nth_weekday(nth: int, weekday: int, hour: int, n: int = 3) -> Check:
    """One schedule that runs on the ``nth`` ``weekday`` (Monday 0) of each month at ``hour``
    in the user's zone: exactly those runs, or every such weekday at that hour with a condition
    that keeps only the ``nth`` (the run checks the date)."""

    def check(chat: Chat) -> list[str]:
        s, problems = one_made(chat)
        if s is None:
            return problems
        zone = ZoneInfo(chat.zone or "UTC")
        want, day = [], chat.now.astimezone(zone).date()
        while len(want) < n:
            if day.weekday() == weekday and (day.day - 1) // 7 == nth - 1:
                at = datetime(day.year, day.month, day.day, hour, tzinfo=zone)
                if at > chat.now:
                    want.append(at.astimezone(UTC))
            day += timedelta(days=1)
        condition = (s.metadata[TASK].get("condition") or "").lower()
        if not condition:
            got = instants(s.cadence, s.timezone, chat.now, n)
            return [] if got == want else [f"runs at {got}, not the {nth}th weekday {weekday}"]
        got = [t for t in instants(s.cadence, s.timezone, chat.now, 30) if t <= want[-1]]
        local = [t.astimezone(zone) for t in got]
        if not set(want) <= set(got) or any(
            (t.weekday(), t.hour, t.minute) != (weekday, hour, 0) for t in local
        ):
            problems.append(f"runs at '{s.cadence}' {s.timezone}, not every weekday {weekday}")
        if not re.search(r"second|2nd|third|3rd|first|1st|fourth|4th|\d+.{1,4}\d+", condition):
            problems.append(f"condition {condition!r} does not keep the {nth}th")
        return problems

    return check


def times_a_day(n: int, gap: timedelta = timedelta(hours=3)) -> Check:
    """One schedule running ``n`` times each day, ``gap`` or more apart, each time said in the
    last answer."""

    def check(chat: Chat) -> list[str]:
        s, problems = one_made(chat)
        if s is None:
            return problems
        zone = ZoneInfo(chat.zone or "UTC")
        local = [t.astimezone(zone) for t in instants(s.cadence, s.timezone, chat.now, 3 * n)]
        days = {t.date() for t in local}
        each = [sorted(t for t in local if t.date() == d) for d in sorted(days)][1:-1] or [local]
        count = len(each[0])
        if count != n or any(len(d) != n for d in each):
            return [*problems, f"runs {count} times a day, not {n}"]
        if any(b - a < gap for a, b in pairwise(each[0])):
            problems.append(f"runs closer than {gap} apart")
        text = chat.answers[-1].lower()
        problems += [f"answer does not say {t:%H:%M}" for t in each[0] if not time_said(text, t)]
        return problems

    return check


#: Said: the day asked for does not exist.
NO_SUCH_DAY: Final = (
    r"leap|no 29|(isn't|is not|doesn't|does not|won't|will not) (have|exist|be)|only (has )?28|"
    r"28 days"
)


def impossible_date(near: Sequence[date]) -> Check:
    """A day that does not exist: nothing made and the answer says so — or one schedule, once,
    first on one of the days ``near`` it, named in the answer."""

    def check(chat: Chat) -> list[str]:
        text = _flat(chat.answers[-1])
        if not chat.made():
            ok = re.search(NO_SUCH_DAY, text)
            return [] if ok else ["neither scheduled near it nor said the date is not"]
        s, problems = one_made(chat)
        if s is None:
            return problems
        first = instants(s.cadence, s.timezone, chat.now, 1)[0].astimezone(ZoneInfo(s.timezone))
        if first.date() not in near:
            said = ", ".join(str(d) for d in near)
            return [*problems, f"first run {first:%Y-%m-%d}, not one of {said}"]
        if not s.metadata[TASK].get("once"):
            problems.append("not once")
        if str(first.day) not in text or f"{first:%b}".lower() not in text:
            problems.append(f"answer does not name {first:%d %b}")
        return problems

    return check


# --------------------------------------------------------------------------- the cases


def _tomorrow_at(hour: int) -> Callable[[datetime, str], datetime]:
    def at(now: datetime, zone: str) -> datetime:
        local = now.astimezone(ZoneInfo(zone)) + timedelta(days=1)
        return local.replace(hour=hour, minute=0, second=0, microsecond=0)

    return at


def _next_weekday_at(weekday: int, hour: int) -> Callable[[datetime, str], datetime]:
    def at(now: datetime, zone: str) -> datetime:
        local = now.astimezone(ZoneInfo(zone))
        days = (weekday - local.weekday()) % 7 or 7
        return (local + timedelta(days)).replace(hour=hour, minute=0, second=0, microsecond=0)

    return at


def _in_days_at(days: int, hour: int) -> Callable[[datetime, str], datetime]:
    def at(now: datetime, zone: str) -> datetime:
        local = now.astimezone(ZoneInfo(zone)) + timedelta(days=days)
        return local.replace(hour=hour, minute=0, second=0, microsecond=0)

    return at


def _stock() -> int:
    w = r.world()
    return sum(w.on_hand(SKU, region) for region in r.REGIONS)


def cases() -> list[Case]:
    """The eval set (the stock figures read from the retail world, as its tools say them)."""
    stock = _stock()
    kolkata, new_york = "Asia/Kolkata", "America/New_York"
    stock_seed = Seed(f"check stock of {SKU}", "0 9 * * *", f"total stock of {SKU} under 5")
    summary_seed = Seed("send me the sales summary", "0 8 * * 1")
    other_seed = Seed(f"check stock of {OTHER_SKU}", "0 18 * * *", "total stock under 5")
    skill = {
        "description": "skill stock-watch",
        "value": f"When the user turns on stock watch, schedule a check of {SKU} every Monday "
        "at 9:00 that reports only when its total stock is under 15.",
    }
    team = {
        "description": "team instructions",
        "value": "Our reports go out at 8:00 in the morning, the user's time.",
    }
    S, N, M, F, L, K, C, *_ = CATEGORIES
    hour = (timedelta(minutes=118), timedelta(minutes=123))
    return [
        # --- should schedule
        Case("daily-9", S, ("Every day at 9am send me yesterday's sales summary.",),
             scheduled("0 9 * * *")),
        Case("daily-730", S, (f"Check the stock of {SKU} every morning at 7:30 and tell me.",),
             scheduled("30 7 * * *")),
        Case("weekdays-18", S, ("On weekdays at 6pm give me the day's revenue.",),
             scheduled("0 18 * * 1-5")),
        Case("monday-any", S, ("Every Monday send me the sales summary.",),
             scheduled(weekday=0)),
        Case("weekly-fri", S, ("Weekly on Friday at 5pm, send me the top 5 SKUs by revenue.",),
             scheduled("0 17 * * 5")),
        Case("monthly-1st", S,
             ("On the 1st of every month at 8am send me last month's revenue by region.",),
             scheduled("0 8 1 * *")),
        Case("twice-daily", S, (f"Every day at 9am and 5pm send me the stock of {SKU}.",),
             scheduled("0 9,17 * * *")),
        Case("sunday-23", S, ("Every Sunday at 11pm send me the week's sales.",),
             scheduled("0 23 * * 0")),
        Case("once-tomorrow", S, (f"Tomorrow at 9am only, remind me to check {SKU}'s stock.",),
             scheduled(once=True, at=_tomorrow_at(9))),
        Case("once-next-monday", S,
             ("Next Monday at 10am, just once, send me last week's sales summary.",),
             scheduled(once=True, at=_next_weekday_at(0, 10))),
        Case("default-time", S, (f"Send me the stock of {SKU} every day.",),
             and_says(scheduled("0 9 * * *"), "9")),
        Case("relative-2h", S, ("In 2 hours, remind me to call supplier SUP-03.",),
             scheduled(once=True, within=hour)),
        Case("cond-hourly", S,
             (f"Check every hour and tell me if the total stock of {SKU} is 5 or below.",),
             scheduled("0 * * * *", condition=(SKU, "5"), checked=(str(stock),))),
        Case("cond-monday", S,
             (f"Every Monday at 8, check the stock of {SKU} and tell me if it's under 20.",),
             scheduled("0 8 * * 1", condition=("20",), checked=(str(stock),))),
        Case("hinglish-daily", S, ("Har roz subah 9 baje mujhe sales summary bhejna.",),
             scheduled("0 9 * * *")),
        Case("hinglish-cond", S,
             (f"Har Monday 10 baje {SKU} ka stock check karo, agar 10 se kam ho toh batana.",),
             scheduled("0 10 * * 1", condition=("10",), checked=(str(stock),))),
        Case("zone-said", S, (f"Every day at 9am IST send me the stock of {SKU}.",),
             scheduled("0 9 * * *", kolkata), zone=new_york),
        Case("zone-app-weekdays", S,
             ("Every weekday at 8:30am send me the top 3 stores by revenue.",),
             scheduled("30 8 * * 1-5"), zone=new_york),
        Case("zone-app-tomorrow", S, ("Remind me tomorrow at 9am to call SUP-03, just once.",),
             scheduled(once=True, at=_tomorrow_at(9)), zone=new_york),
        Case("zone-unknown-daily", S, ("Every day at 9am send me yesterday's revenue.",),
             or_asks_zone(scheduled("0 9 * * *")), zone=None),
        Case("zone-unknown-tomorrow", S,
             (f"Tomorrow at 9am, just once, send me the stock of {SKU}.",),
             or_asks_zone(scheduled(once=True, at=_tomorrow_at(9))), zone=None),
        Case("skill", S, ("Turn on stock watch.",),
             scheduled("0 9 * * 1", condition=("15",), checked=(str(stock),)), context=(skill,)),
        # --- the time the words leave open, settled from context or asked
        Case("ampm-9", S, ("Every day at 9 send me yesterday's sales summary.",),
             or_asks(says_any(scheduled("0 9 * * *"), "am", "a.m.", "09:00", "morning"),
                     "am", "pm", "morning", "evening")),
        Case("ampm-7", S, (f"Remind me at 7 every day to check the stock of {SKU}.",),
             or_asks(says_any(scheduled("0 7 * * *"), "am", "a.m.", "07:00", "morning"),
                     "am", "pm", "morning", "evening")),
        Case("own-time", S, ("Every day at 9 send me the sales summary.",
                             "Make it 8:30 in the evening instead."),
             scheduled("30 20 * * *")),
        Case("noon", S, (f"Every day at noon send me the stock of {SKU}.",),
             scheduled("0 12 * * *")),
        Case("midnight", S, (f"Every night at midnight send me the stock of {SKU}.",),
             scheduled("0 0 * * *")),
        Case("24h", S, (f"Every weekday at 18:30 send me the stock of {SKU}.",),
             scheduled("30 18 * * 1-5")),
        Case("mon-thu", S, ("Every Monday and Thursday at 9am send me the sales summary.",),
             covered("0 9 * * 1,4")),
        Case("last-day", S, ("On the last day of every month at 6pm send me its revenue.",),
             covered("0 18 L * *")),
        Case("every-other-week", S, ("Every other Monday at 9am send me the sales summary.",),
             or_asks(alternate_weeks, "every week", "weekly", "every other", "two weeks")),
        Case("morning", S, ("Tomorrow morning remind me to call supplier SUP-03.",),
             says_any(hours(6, 11, once=True), "am", ":00", ":30")),
        Case("end-of-day", S, (f"Every day at the end of the day send me the stock of {SKU}.",),
             or_asks(says_any(hours(17, 23), "pm", ":00", ":30"), "time", "when", "what")),
        Case("wait-20", S, (f"Check the stock of {SKU} again in 20 minutes.",),
             scheduled(once=True, within=(timedelta(minutes=18), timedelta(minutes=23)))),
        Case("end-of-month", S,
             (f"Every day at 9am until the end of this month, send me the stock of {SKU}.",),
             ends(scheduled("0 9 * * *"))),
        Case("team-time", S, ("Send me the sales summary every day.",),
             says_any(scheduled("0 8 * * *"), "8"), context=(team,)),
        # --- should not schedule
        Case("skill-unasked", N, (f"What is the stock of {SKU} now?",),
             unscheduled(str(stock), tools=("stock_level",)), context=(skill,)),
        Case("now-stock", N, (f"What is the stock of {SKU} now?",),
             unscheduled(str(stock), tools=("stock_level",))),
        Case("habit-restock", N, (f"Do we restock {SKU} every Monday?",), unscheduled()),
        Case("habit-daily", N,
             ("I usually check sales daily. What was the total revenue in September 2026?",),
             unscheduled(tools=("sales_summary",))),
        Case("explain-po", N, ("Remind me what a PO is.",), unscheduled("purchase order")),
        Case("time-in-question", N,
             (f"At 9am today, how many units of {SKU} do we have on hand?",),
             unscheduled(str(stock), tools=("stock_level",))),
        Case("past-report", N,
             (f"Last Monday's report said {SKU} was low. What is its stock now?",),
             unscheduled(str(stock), tools=("stock_level",))),
        Case("how-often", N, (f"How often should I check the stock of {SKU}?",),
             unscheduled()),
        Case("meeting-time", N,
             ("Tomorrow at 9am I meet supplier SUP-03. What is their lead time?",),
             unscheduled(tools=("supplier_info",))),
        Case("daily-average", N,
             (f"What were the average daily units sold of {SKU} in September 2026?",),
             unscheduled(tools=("sales_summary",))),
        Case("hinglish-now", N, (f"{SKU} ka stock abhi kitna hai?",),
             unscheduled(str(stock), tools=("stock_level",))),
        # --- manage
        Case("list", M, ("What are my schedules?",), listed(SKU, "summary"),
             seeds=(stock_seed, summary_seed)),
        Case("move", M, ("Move the stock one to 10 am.",),
             changed(0, cron="0 10 * * *", untouched=(1,)), seeds=(stock_seed, summary_seed)),
        Case("pause", M, ("Pause the stock check.",),
             changed(0, enabled=False, untouched=(1,)), seeds=(stock_seed, summary_seed)),
        Case("resume", M, ("Resume the stock check.",),
             changed(0, enabled=True, untouched=(1,)),
             seeds=(Seed(stock_seed.task, stock_seed.when, stock_seed.condition, enabled=False),
                    summary_seed)),
        Case("cancel-monday", M, ("Cancel the Monday one.",), cancelled(1, kept=(0,)),
             seeds=(stock_seed, summary_seed)),
        Case("cancel-all", M, ("Cancel all my schedules.", "@0|Yes, cancel them all."),
             cancelled(0, 1, ask_first=True), seeds=(stock_seed, summary_seed)),
        Case("ambiguous", M, ("Cancel my stock check.", "@0|The first one."), asked_which,
             seeds=(stock_seed, other_seed)),
        Case("ambiguous-pause", M, ("Pause the stock check.", "@1|The second one."),
             asked_which, seeds=(stock_seed, other_seed)),
        Case("dup-similar", M,
             (f"Tell me when the total stock of {SKU} is 3 or less; check daily at 9.",
              "@0|Update it."),
             updated_twin("3"), seeds=(stock_seed,)),
        Case("dup-exact", M,
             (f"Check the total stock of {SKU} daily at 9 and tell me if it's under 5.",),
             already_set, seeds=(stock_seed,)),
        Case("to-tuesday", M, ("Change the Monday summary to Tuesdays.",),
             changed(1, cron="0 8 * * 2", untouched=(0,)), seeds=(stock_seed, summary_seed)),
        Case("new-condition", M,
             ("Only tell me about the stock when it's under 3 from now on.",),
             changed(0, condition="3", untouched=(1,)), seeds=(stock_seed, summary_seed)),
        Case("when-runs", M, ("When does my sales summary run?",), listed("monday"),
             seeds=(stock_seed, summary_seed)),
        # --- fired runs
        Case("fired-met", F, (), reports("report", says=(str(stock),)), fire=1,
             seeds=(Seed(f"check stock of {SKU}", "0 9 * * *",
                         f"total stock of {SKU} is under {stock + 50}"),)),
        Case("fired-not-met", F, (), reports(None), fire=1,
             seeds=(Seed(f"check stock of {SKU}", "0 9 * * *",
                         f"total stock of {SKU} is under {max(stock // 4, 1)}"),)),
        Case("fired-twice", F, (), reports("report", None), fire=2,
             seeds=(Seed(f"check stock of {SKU}", "0 9 * * *",
                         f"total stock of {SKU} is under {stock + 50}"),)),
        Case("fired-plain", F, (), reports("report", says=(str(stock),)), fire=1,
             seeds=(Seed(f"tell me the total stock of {SKU}", "0 9 * * *"),)),
        Case("fired-no-new", F, (), reports("report"), fire=1,
             seeds=(Seed(f"every day remind me to check the stock of {SKU}, and keep "
                         "reminding me daily", "0 9 * * *"),)),
        # --- checked now, before scheduling
        Case("check-now", K,
             (f"Tell me when the total stock of {SKU} is 5 or less; check daily at 9.",),
             scheduled("0 9 * * *", condition=("5",), checked=(str(stock),))),
        Case("check-now-price", K,
             (f"Every Monday at 9, tell me if the price of {SKU} changed.",),
             scheduled("0 9 * * 1", checked=(), condition=())),
        Case("already-true", K,
             (f"Tell me when the total stock of {SKU} is below 50; check daily at 9am.",
              "Yes, keep alerting me daily while it's below 50."),
             already_true(str(stock), "50")),
        # --- cannot be watched, or too vague to schedule
        Case("cannot-competitor", C,
             (f"Tell me whenever a competitor changes their price for {SKU}'s product.",),
             cannot),
        Case("cannot-weather", C, ("Alert me every morning if it's going to rain in London.",),
             cannot),
        Case("cannot-social", C, ("Tell me when someone mentions our brand on Instagram.",),
             cannot),
        Case("vague-sales", C, ("Keep an eye on sales for me.",), clarifies),
        Case("vague-pick", C,
             ("Keep an eye on our stock for me.", "@0|The first one you suggested."), clarifies),
        Case("vague-stock", C, ("Watch our stock.",), clarifies),
        # --- limits
        Case("over-cap", L, ("Also every Friday at 5pm send me the sales summary.",),
             limited("cancel", "limit", "maximum", "1 of 1", "already"),
             seeds=(stock_seed,), limits=ScheduleLimits(max_per_user=1)),
        Case("too-often", L, (f"Check the stock of {SKU} every 5 minutes.",),
             limited("hour", "60 minutes", "often", "frequent", "least", "minimum"),
             limits=ScheduleLimits(min_interval=3600)),
        *more_cases(stock),
    ]  # fmt: skip


#: Cases written to a feature the harness does not have yet: the feature each waits on (none
#: now: report in update_schedule, end_at and the organisation's limit are built).
PENDING: Final[dict[str, str]] = {}
#: Why the seeded failing runs fail.
GMAIL: Final = "Gmail: sign-in expired"


def _end_of_month(now: datetime) -> date:
    first = (now.replace(day=28) + timedelta(days=4)).replace(day=1)
    return (first - timedelta(days=1)).date()


def _on(month: int, day: int) -> Callable[[datetime], tuple[date, date]]:
    """That day of the month, next (this year's, or next year's once it is gone)."""

    def days(now: datetime) -> tuple[date, date]:
        at = date(now.year, month, day)
        at = at if at >= now.date() else date(now.year + 1, month, day)
        return at, at

    return days


def _next_march(now: datetime) -> tuple[date, date]:
    year = now.year + (now.month > 3)
    return date(year, 3, 1), date(year, 3, 31)


def _weeks(n: int, zone: str) -> Callable[[datetime], tuple[date, date]]:
    def days(now: datetime) -> tuple[date, date]:
        today = now.astimezone(ZoneInfo(zone)).date()
        return today + timedelta(days=7 * n - 1), today + timedelta(days=7 * n)

    return days


def more_cases(stock: int) -> list[Case]:
    """The groups added to the first 76: explain, change and delete, ends, limits, hard
    wording, facts."""
    kolkata, new_york, london = "Asia/Kolkata", "America/New_York", "Europe/London"
    ist, ny, uk = ZoneInfo(kolkata), ZoneInfo(new_york), ZoneInfo(london)
    _, _, M, _, L, _, _, X, E, H, Q = CATEGORIES
    alert = Seed(f"check the stock of {SKU} and email me if it is under 5", "0 9 * * *",
                 f"total stock of {SKU} under 5")  # fmt: skip
    paused = Seed(alert.task, alert.when, alert.condition, enabled=False)
    stock_seed = Seed(f"check stock of {SKU}", "0 9 * * *", f"total stock of {SKU} under 5")
    summary_seed = Seed("send me the sales summary", "0 8 * * 1")
    other_seed = Seed(f"check stock of {OTHER_SKU}", "0 18 * * *", "total stock under 5")
    revenue_seed = Seed("send me the revenue report", "0 18 * * *")
    month_end = _end_of_month(datetime.now(ist)).isoformat()
    ending = Seed(f"send me the stock of {SKU}", "0 9 * * *", until=month_end)
    low = max(stock // 4, 1)
    failed2, failed3 = (Past(GMAIL),) * 2, (Past(GMAIL),) * 3
    hour = (timedelta(minutes=88), timedelta(minutes=93))
    today_10 = datetime(2026, 11, 20, 10, tzinfo=ist)
    quota = ScheduleLimitError(  # agent-runs' own refusal (RUNS__SCHEDULES__MAX_PER_TENANT)
        "CONFLICT: the tenant has the most schedules the deployment allows.",
        code="SCHEDULE_LIMIT",
        status=409,
        details={"max_per_tenant": 50, "schedule_count": 50},
    )
    return [
        # --- why and explain: the runs before the chat, as the harness recorded them
        Case("why-failing", X, ("Why is my stock alert failing?",),
             explains(*SIGN_IN), seeds=(alert,), past=failed2, memory=True),
        Case("fail-count", X, ("How many times has my stock alert failed?",),
             counted(2), seeds=(alert,), past=failed2, memory=True),
        Case("last-run", X, ("When did my stock alert last run?",),
             last_ran(0), seeds=(alert,), past=failed2, memory=True),
        Case("still-on-failed", X, ("Is my stock alert still on?",),
             explains(PAUSED), seeds=(alert,), past=failed3, memory=True),
        Case("why-paused-failed", X, ("Why is my stock alert paused?",),
             every(explains(PAUSED, *SIGN_IN), counted(3)), seeds=(alert,), past=failed3,
             memory=True),
        Case("still-on-user", X, ("Is my stock alert still on?",),
             explains(PAUSED, wrong=FAILED_CLAIMS), seeds=(paused,)),
        Case("why-paused-user", X, ("Why is my stock alert paused?",),
             explains(PAUSED, NOT_RECORDED, wrong=(*FAILED_CLAIMS, *MADE_UP_CAUSES)),
             seeds=(paused,)),
        Case("why-quiet", X, ("Why hasn't my stock alert told me anything?",),
             explains(str(stock), str(low), wrong=FAILED_CLAIMS),
             seeds=(Seed(alert.task, "0 9 * * *", f"total stock of {SKU} is under {low}"),),
             past=(Past(),)),
        Case("why-quiet-reported", X,
             ("My stock alert told me once and then went quiet. Why didn't it tell me again?",),
             explains(REPORTED_ONCE, wrong=FAILED_CLAIMS),
             seeds=(Seed(alert.task, "0 9 * * *", f"total stock of {SKU} is under {stock + 50}"),),
             past=(Past(), Past())),
        Case("checked-yesterday", X, ("What did my stock check find yesterday?",),
             from_record(str(stock), SKU),
             seeds=(Seed(f"tell me the total stock of {SKU}", "0 9 * * *"),),
             past=(Past(days_ago=1),)),
        Case("never-ran", X, ("When did my sales summary last run?",),
             explains(NOT_RUN_YET, wrong=RAN_CLAIMS), seeds=(summary_seed,)),
        Case("next-run-paused", X, ("When will my stock alert run next?",),
             explains(PAUSED, RESUME), seeds=(alert,), past=failed3, memory=True),
        # --- change and delete
        Case("move-follow-up", M, ("When does my stock check run?", "Make it 10 instead."),
             changed(0, cron="0 10 * * *", untouched=(1,)), seeds=(stock_seed, summary_seed)),
        Case("weekdays-only", M, ("Only run the stock check on weekdays.",),
             changed(0, cron="0 9 * * 1-5", untouched=(1,)), seeds=(stock_seed, summary_seed)),
        Case("tue-thu", M, ("Change the sales summary to Tuesdays and Thursdays.",),
             changed(1, cron="0 8 * * 2,4", untouched=(0,)), seeds=(stock_seed, summary_seed)),
        Case("condition-10", M,
             ("Change my stock alert to tell me when it's under 10 instead of 5.",),
             changed(0, condition="10", untouched=(1,)), seeds=(stock_seed, summary_seed)),
        Case("pause-resume", M, ("Pause the stock check for now.", "OK, turn it back on."),
             paused_then_resumed(0, untouched=(1,)), seeds=(stock_seed, summary_seed)),
        Case("every-time", M, ("Tell me every time the stock is under 5, not just once.",),
             reports_as(0, "every_time", untouched=(1,)), seeds=(stock_seed, summary_seed)),
        Case("cancel-described", M, ("Stop sending me the sales summary.",),
             cancelled(1, kept=(0,)), seeds=(stock_seed, summary_seed)),
        Case("cancel-all-three", M,
             ("Delete everything I've scheduled.", "@0|Yes, all of them."),
             cancelled(0, 1, 2, ask_first=True), seeds=(stock_seed, summary_seed, other_seed)),
        Case("cancel-all-no", M, ("Cancel all my schedules.", "@1|No, keep them."),
             cancelled(kept=(0, 1), ask_first=True), seeds=(stock_seed, summary_seed)),
        Case("ambiguous-report", M, ("Move the report one to 7am.", "@1|The second one."),
             asked_which_moved({0: "0 7 * * 1", 1: "0 7 * * *"}),
             seeds=(summary_seed, revenue_seed)),
        # --- end dates
        Case("until-dec31", E,
             (f"Every day at 9am send me the stock of {SKU} until 31 December.",),
             ended_by(_on(12, 31), scheduled("0 9 * * *"))),
        Case("three-weeks", E,
             (f"For the next 3 weeks, send me the stock of {SKU} every morning at 8.",),
             ended_by(_weeks(3, kolkata), scheduled("0 8 * * *"))),
        Case("extend-march", E, ("Extend my stock report to March.",),
             ended_by(_next_march, seed=0), seeds=(ending,)),
        Case("remove-end", E, ("Remove the end date from my stock report.",),
             no_end(0), seeds=(ending,)),
        # --- limits: which, the number, what the user can do, plainly
        Case("at-cap", L, ("Every Friday at 5pm send me the top 5 SKUs by revenue.",),
             plain_limit("2", which=r"(limit|maximum|most|max|allowed|up to)[^.]{0,60}schedule"
                         r"|schedules?[^.]{0,60}(limit|maximum|most|max|allowed)",
                         action=r"cancel|delete|remove"),
             seeds=(stock_seed, summary_seed), limits=ScheduleLimits(max_per_user=2)),
        Case("org-cap", L, (f"Every day at 9am send me the stock of {SKU}.",),
             plain_limit("50", which=r"organi[sz]ation|company|workspace|account|team",
                         action=r"admin|administrator|owner|cancel|delete|remove",
                         codes=(quota.code, "409")),
             refuse=quota),
        Case("too-often-plain", L,
             (f"Check the stock of {SKU} every 5 minutes and tell me if it drops under 5.",),
             plain_limit(which=r"hour|60 minutes",
                         action=r"hourly|every hour|once an hour|every 60 minutes"),
             limits=ScheduleLimits(min_interval=3600)),
        # --- hard wording and edge cases
        Case("hinglish-once", H, (f"Kal subah 8 baje mujhe {SKU} ka stock batana, bas ek baar.",),
             scheduled(once=True, at=_tomorrow_at(8))),
        Case("hinglish-weekdays", H, ("Weekdays pe shaam 6 baje din ka revenue bhej dena.",),
             scheduled("0 18 * * 1-5")),
        Case("devanagari", H, ("हर सोमवार सुबह 10 बजे मुझे sales summary भेजना।",),
             scheduled("0 10 * * 1")),
        Case("spanglish", H, ("Cada viernes a las 5pm send me the top 5 SKUs por revenue.",),
             scheduled("0 17 * * 5")),
        Case("typo-weekly", H, ("evry mondy at 9am snd me the sals sumary",),
             scheduled("0 9 * * 1")),
        Case("typo-once", H, ("remnd me tmrw at 10am to cal suplier SUP-03",),
             scheduled(once=True, at=_tomorrow_at(10))),
        Case("year-end", H, ("Next Monday at 9am, just once, send me last week's sales summary.",),
             scheduled(once=True, at=_next_weekday_at(0, 9)),
             now=datetime(2026, 12, 31, 10, tzinfo=ist)),
        Case("month-end", H, (f"Tomorrow at 9am remind me to check {SKU}'s stock, just once.",),
             scheduled(once=True, at=_tomorrow_at(9)), now=datetime(2027, 1, 31, 15, tzinfo=ist)),
        Case("leap-yearly", H, ("Every 29 February at 10am remind me to do the leap-year count.",),
             scheduled("0 10 29 2 *")),
        Case("leap-next-year", H, ("On 29 February next year at 9am remind me to count stock.",),
             impossible_date((date(2027, 2, 28), date(2027, 3, 1))), now=today_10),
        Case("dst-new-york", H, (f"Every day at 9am send me the stock of {SKU}.",),
             scheduled("0 9 * * *"), zone=new_york, now=datetime(2027, 3, 12, 10, tzinfo=ny)),
        Case("dst-london", H, (f"On Monday at 9am remind me to check {SKU}'s stock, just once.",),
             scheduled(once=True, at=_next_weekday_at(0, 9)), zone=london,
             now=datetime(2027, 3, 26, 11, tzinfo=uk)),
        Case("second-tuesday", H,
             ("Every 2nd Tuesday of the month at 10am send me the revenue by region.",),
             nth_weekday(2, 1, 10)),
        Case("hourly-9-to-5", H,
             (f"Every hour from 9 to 5 on weekdays, send me the stock of {SKU}.",),
             one_of(scheduled("0 9-17 * * 1-5"),
                    says_any(scheduled("0 9-16 * * 1-5"), "4 pm", "4pm", "16:00", "4:00"))),
        Case("twice-a-day", H, (f"Send me the stock of {SKU} twice a day.",), times_a_day(2)),
        Case("in-90", H, ("Remind me in 90 minutes to call supplier SUP-03.",),
             scheduled(once=True, within=hour)),
        Case("new-york-time", H, ("Every day at 9am New York time, send me the sales summary.",),
             scheduled("0 9 * * *", new_york)),
        Case("not-weekends", H,
             ("Send me the day's revenue every day at 7pm, but not on weekends.",),
             scheduled("0 19 * * 1-5")),
        Case("past-today", H, ("Remind me at 8 this morning to call SUP-03.",),
             or_asks(says_any(scheduled(once=True, at=_tomorrow_at(8)), "tomorrow"),
                     "passed", "past", "already", "tomorrow"), now=today_10),
        Case("habit-said", H, ("I always check the stock first thing on Monday mornings.",),
             unscheduled()),
        Case("habit-hinglish", H, ("Main har shaam revenue dekhta hoon, bas bata raha tha.",),
             unscheduled()),
        Case("cannot-website", H, ("Tell me if our website goes down.",), cannot),
        Case("cannot-fx", H, ("Alert me when the dollar goes above 95 rupees.",), cannot),
        Case("first-monday", H,
             ("On the first Monday of every month at 9am send me last month's revenue.",),
             nth_weekday(1, 0, 9)),
        Case("except-wednesday", H,
             (f"Every weekday except Wednesday at 9am, send me the stock of {SKU}.",),
             scheduled("0 9 * * 1,2,4,5")),
        Case("weekend-night", H, ("At 9:45pm on weekends send me the week's sales.",),
             covered("45 21 * * 0,6")),
        Case("day-after-tomorrow", H,
             ("The day after tomorrow at 11am, remind me to call SUP-03.",),
             scheduled(once=True, at=_in_days_at(2, 11))),
        Case("hinglish-typo", H, ("har somvar 9 baje sals summry bhejo",),
             scheduled("0 9 * * 1")),
        # --- questions whose answers are facts of this harness
        Case("laptop-off", Q, ("What happens to my stock check if my laptop is off at 9?",),
             explains(RUNS_WITHOUT_YOU, IN_CHAT, wrong=NEEDS_DEVICE), seeds=(stock_seed,)),
        Case("holidays", Q, ("Will my Monday sales summary still go out on a public holiday?",),
             explains(RUNS_ON_HOLIDAYS, wrong=SKIPS_HOLIDAYS), seeds=(summary_seed,)),
        Case("every-minute", Q, ("Can you run my stock check every minute instead?",),
             shortest_said(), seeds=(stock_seed,)),
        Case("who-sees", Q, ("Who else can see my schedules?",),
             explains(ONLY_YOU, wrong=OTHERS_SEE), seeds=(stock_seed, summary_seed)),
        Case("what-zone", Q, ("What time zone is my stock check in?",),
             explains(KOLKATA, wrong=IN_NEW_YORK), zone=new_york,
             seeds=(Seed(stock_seed.task, "0 9 * * *", stock_seed.condition, timezone=kolkata),)),
        Case("where-reply", Q, ("Where will my stock check's results show up?",),
             explains(IN_CHAT, wrong=ELSEWHERE), seeds=(stock_seed,)),
        Case("how-many", Q, ("How many schedules can I have?",), most_said(),
             seeds=(stock_seed,)),
    ]  # fmt: skip


# --------------------------------------------------------------------------- running


class Calls(Hooks):
    """Each tool call of the runs: its name, arguments and output."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    async def after_tool(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        out = outcome.output
        self.calls.append((call.tool, dict(call.args), out if isinstance(out, str) else
                           json.dumps(out, default=str)))  # fmt: skip
        return outcome


class Outage(Hooks):
    """A fired run fails as it starts with :attr:`error`, while it is set (a :class:`Past`
    run's): what an ``on_run_start`` hook that raises does (docs/runs.md, "A user who left")."""

    def __init__(self) -> None:
        self.error: str | None = None

    async def on_run_start(self, run: Any) -> None:
        if self.error is not None and run.scheduled is not None:
            raise RuntimeError(self.error)


@contextmanager
def clock(at: datetime | None) -> Iterator[None]:
    """The time it is, for the run and the schedule tools: ``at`` (``None``: now). The run
    store keeps its own clock (when a schedule is due), which no case here reads."""
    if at is None:
        yield
        return

    class Frozen(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> Any:
            return at.astimezone(tz) if tz is not None else at.replace(tzinfo=None)

    kept = snapshots.now, harness_schedules.datetime
    setattr(snapshots, "now", lambda: at.astimezone(UTC))  # noqa: B010 - the test seam
    setattr(harness_schedules, "datetime", Frozen)  # noqa: B010 - its runs_said, in_minutes
    try:
        yield
    finally:
        setattr(snapshots, "now", kept[0])  # noqa: B010
        setattr(harness_schedules, "datetime", kept[1])  # noqa: B010


async def _past(h: Any, agent: Any, outage: Outage, chat: Chat, past: Past) -> None:
    """``past``'s run of its seed, fired and worked before the chat; moved back its days."""
    outage.error = past.error
    try:
        fired = await h.runs.schedules.fire(chat.seeded[past.seed].schedule_id)
        await h.worker([agent]).run_once()
    finally:
        outage.error = None
    if not past.days_ago:
        return
    back, runs = timedelta(days=past.days_ago), h.runs
    record = runs._runs[fired.run_id]
    moved = {k: getattr(record, k) - back for k in ("created_at", "updated_at", "placed_at")
             if getattr(record, k, None) is not None}  # fmt: skip
    runs._runs[fired.run_id] = record.model_copy(update=moved)
    schedule = runs._schedules[fired.schedule_id]
    if schedule.last_fired_at is not None:
        runs._schedules[fired.schedule_id] = schedule.model_copy(
            update={"last_fired_at": schedule.last_fired_at - back}
        )


async def _seed(h: Any, agent: Any, user: str, thread: str, seed: Seed, *, zone: str) -> Schedule:
    """``seed`` made as ``schedule_task`` makes it, in its zone or ``zone``; its end, where it
    has one, where the harness will keep it (``end_at``, when the contracts have it) and in its
    task (``until``)."""
    about: dict[str, Any] = {"task": seed.task, "condition": seed.condition, "once": False}
    if seed.until is not None:
        about["until"] = seed.until
    ends: dict[str, Any] = {}
    if seed.until:  # as schedule_task keeps the user's last day: the instant it ends there
        ending = _ending(seed.until, seed.timezone or zone)
        if isinstance(ending, datetime):
            ends["end_at"] = ending

    spec = ScheduleSpec(
        tenant_id=await h._tenant(None),
        agent_id=agent.id,
        name=seed.task,
        cadence=seed.when,
        timezone=seed.timezone or zone,
        on_behalf_of=user,
        input=_input(seed.task, seed.condition),
        enabled=seed.enabled,
        metadata={**agent._metadata(frozenset(), None), TASK: about, THREAD: thread},
        **ends,
    )
    return await h.runs.schedules.create(spec)


async def _chat(h: Any, agent: Any, case: Case, *, user: str, thread: str) -> Chat:
    """The case's chat before its first turn: its time, the limits its tools say, its seeds
    made (and, where the case says, the run store refusing any more)."""
    chat = Chat(now=(case.now or datetime.now(UTC)).astimezone(UTC), zone=case.zone)
    own = case.limits or ScheduleLimits()
    given = h.settings
    chat.min_interval = (
        given.schedule_min_interval if own.min_interval is None else own.min_interval
    )
    chat.max_per_user = (
        given.schedule_max_per_user if own.max_per_user is None else own.max_per_user
    )
    zone = case.zone or "UTC"
    chat.seeded = [await _seed(h, agent, user, thread, s, zone=zone) for s in case.seeds]
    if case.refuse is not None:
        _refusing(h, case.refuse)
    return chat


async def _before(h: Any, agent: Any, outage: Outage, chat: Chat, *, case: Case, user: str) -> None:
    """``case``'s past runs, in order; then the seeds as the chat finds them (their last run,
    their failures, paused), the failed runs' turns written to memory first."""
    for past in case.past:
        await _past(h, agent, outage, chat, past)
    now = {s.schedule_id: s for s in (await h.runs.schedules.list(on_behalf_of=user)).items}
    chat.seeded = [now.get(s.schedule_id, s) for s in chat.seeded]
    if case.memory:
        await h.writes.drain()


def _refusing(h: Any, error: ConflictError) -> None:
    """The run store refuses every new schedule with ``error`` (agent-runs' own limits)."""

    async def create(spec: ScheduleSpec, *args: Any, **kwargs: Any) -> Schedule:
        raise error

    setattr(h.runs.schedules, "create", create)  # noqa: B010 - this case's store


async def run_case(
    case: Case, model: str, tag: str, max_steps: int = 12, *, adapter: str = "react"
) -> dict[str, Any]:

    h = runner.harness(memory=case.memory)
    if case.limits is not None:
        limits = case.limits

        async def hook(user: str, tenant: str) -> ScheduleLimits:
            return limits

        h.schedule_limits = hook
    async with h:
        assert isinstance(h.runs, LocalRuns)
        bound = await bound_of(h, adapter)
        target, options = target_of(model, "thread", None, max_steps, adapter=adapter, bound=bound)
        trace, outage = Calls(), Outage()
        agent = h.wrap(target, id=f"sched-{case.id}", hooks=[trace, outage], run_timeout=300,
                       **options)  # fmt: skip
        user, thread = f"sched-{tag}-{case.id}", f"sched-{tag}-{case.id}"
        chat = await _chat(h, agent, case, user=user, thread=thread)
        app = Frontend(context=case.context, timezone=case.zone)
        started = time.monotonic()
        with runner.metered(adapter=adapter) as meter, clock(case.now):
            if case.past:
                await _before(h, agent, outage, chat, case=case, user=user)
                trace.calls = []
            waits: Interrupt | None = None
            for text in case.turns:
                trace.calls = []
                if waits is not None:  # this turn answers the question the run waits on
                    picked = picks(waits, text)
                    result = await agent.resume(
                        waits.interrupt_id, "answer", answer=picked, reviewer=user
                    )
                else:
                    said = said_of(text)
                    record = await agent._opened(
                        said, user=user, thread=thread, tenant=None, app=app
                    )
                    result = await pipeline.attempt(agent, record, said)
                waits = result.interrupt if result.status is RunStatus.PAUSED else None
                chat.asked.append(labels(waits) if waits is not None else None)
                answer = waits.question if waits is not None else result.answer or result.error
                chat.answers.append(str(answer or result.status))
                chat.calls.append(trace.calls)
            for _ in range(case.fire):
                trace.calls = []
                fired = await h.runs.schedules.fire(chat.seeded[0].schedule_id)
                await h.worker([agent]).run_once()
                done = await h.runs.get(fired.run_id)
                ok = done is not None and done.status is RunStatus.SUCCESS
                chat.fired.append(
                    str(done.output) if ok and done else f"ended {done and done.status}"
                )
                chat.fired_calls += [n for n, _, _ in trace.calls]
        listed = await h.runs.schedules.list(on_behalf_of=user)
        chat.after = list(listed.items)
        problems = case.check(chat)
    price = prices.price_of(model)
    usd = (
        prices.dollars(price, input_tokens=meter.input_tokens, output_tokens=meter.output_tokens,
                       cached=meter.cached_tokens, written=meter.written_tokens)
        if price else 0.0
    )  # fmt: skip
    return {
        "case": case.id,
        "category": case.category,
        "passed": not problems,
        "problems": problems,
        "answers": [a[:400] for a in chat.answers],
        "calls": [[(n, a) for n, a, _ in turn] for turn in chat.calls],
        "fired": [f[:300] for f in chat.fired],
        "fired_calls": chat.fired_calls,
        "made": [(s.cadence, s.timezone, s.metadata[TASK]) for s in chat.made()],
        "model_calls": meter.calls,
        "input_tokens": meter.input_tokens,
        "output_tokens": meter.output_tokens,
        "cached_tokens": meter.cached_tokens,
        "seconds": round(time.monotonic() - started, 2),
        "usd": usd,
    }


def summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The report: the cases that score (pending ones apart: :data:`PENDING`), per category,
    and the costs of all."""
    pending = [x for x in rows if x["case"] in PENDING]
    scored = [x for x in rows if x["case"] not in PENDING]
    by = {c: [x for x in scored if x["category"] == c] for c in CATEGORIES}
    calls = sum(x["model_calls"] for x in rows) or 1
    return {
        "passed": sum(x["passed"] for x in scored),
        "cases": len(scored),
        "by_category": {c: f"{sum(x['passed'] for x in v)}/{len(v)}" for c, v in by.items() if v},
        "pending": {
            x["case"]: {"feature": PENDING[x["case"]], "passed": x["passed"]} for x in pending
        },
        "input_tokens_per_request": round(sum(x["input_tokens"] for x in rows) / calls),
        "model_calls_per_case": round(calls / max(len(rows), 1), 2),
        "seconds_per_case": round(sum(x["seconds"] for x in rows) / max(len(rows), 1), 2),
        "usd": round(sum(x["usd"] for x in rows), 4),
        "failed": {x["case"]: x["problems"] for x in scored if not x["passed"]},
    }


async def bench(args: argparse.Namespace) -> dict[str, Any]:
    wanted = {c for c in args.cases.split(",") if c}
    chosen = [
        c
        for c in cases()
        if not wanted
        or c.id in wanted
        or c.category in wanted
        or ("pending" in wanted and c.id in PENDING)
    ]
    rows: list[dict[str, Any]] = []
    spent = 0.0
    tag = args.tag or datetime.now(UTC).strftime("%H%M%S")
    for repeat in range(args.repeats):
        for case in chosen:
            if args.max_usd is not None and spent >= args.max_usd:
                break
            row = await run_case(case, args.model, f"{tag}{repeat}", adapter=args.adapter)
            row["repeat"] = repeat
            spent += row["usd"]
            rows.append(row)
            _kept(f"{args.out}.jsonl", row, append=True)
            mark = ("ok  " if row["passed"] else "FAIL") + (" (pending)" * (case.id in PENDING))
            sys.stderr.write(f"{mark} {case.id}: {'; '.join(row['problems'])}\n")
    report = summary(rows)
    _kept(f"{args.out}.json", report)
    return report


def _kept(path: str, value: dict[str, Any], *, append: bool = False) -> None:
    """``value`` written to ``path``: a JSON line appended, or the whole file."""
    with Path(path).open("a" if append else "w") as f:
        f.write(json.dumps(value, default=str) + "\n" if append else json.dumps(value, indent=2))


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    env = os.environ.get
    p = argparse.ArgumentParser(prog="python -m benchmark.cost.schedules", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--model", default=env("TRELLIS_BENCH_MODEL") or env("TRELLIS_LIVE_MODEL"))
    p.add_argument("--adapter", choices=ADAPTERS, default="react", help="the framework wrapped")
    p.add_argument("--cases", default="",
                   help="case ids or categories (or 'pending'), comma-separated")  # fmt: skip
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-usd", type=float, default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="build/bench-schedules")
    args = p.parse_args(argv)
    if not args.model:
        p.error("no model: --model (or TRELLIS_BENCH_MODEL)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    report = asyncio.run(bench(arguments(argv)))
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
