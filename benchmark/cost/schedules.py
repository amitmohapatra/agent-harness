"""Scheduling intent: does the agent schedule when it is asked to, and only then, at the right
time, and do the runs a schedule fires report what they should, once?

Each case is one chat (one or more turns) with the retail tools (``retail.py``) and the
built-in schedule tools (``trellis.harness.schedules``), ``ReAct`` on the in-process run store,
memory off. Categories:

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
* ``limits`` — over the per-user cap or too frequent: one clear sentence, no retry loop.

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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

from croniter import croniter

from benchmark.cost import prices
from benchmark.cost import retail as r
from benchmark.cost import run as runner
from trellis import Hooks
from trellis.contracts import RunStatus, Schedule, ScheduleSpec, ToolCall, ToolOutcome
from trellis.harness import pipeline
from trellis.harness.frontend import Frontend
from trellis.harness.runs import THREAD, LocalRuns
from trellis.harness.schedules import NO_REPORT, SCHEDULE_TOOLS, TASK, ScheduleLimits, _input

#: The categories, in the order they are reported.
CATEGORIES: Final = ("schedule", "no_schedule", "manage", "fired", "limits")
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


@dataclass
class Chat:
    """What a case's chat did: each turn's answer and calls, the user's schedules before and
    after, and each fired run's output."""

    now: datetime
    #: the user's zone as their app says it
    zone: str | None = None
    answers: list[str] = field(default_factory=list)
    calls: list[list[tuple[str, dict[str, Any], str]]] = field(default_factory=list)
    seeded: list[Schedule] = field(default_factory=list)
    after: list[Schedule] = field(default_factory=list)
    fired: list[str] = field(default_factory=list)
    fired_calls: list[str] = field(default_factory=list)

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
) -> Check:
    """One schedule made: firing at ``cron``'s instants in ``zone`` (else the user's, else
    UTC) — or first at ``at(now, zone)``, or within ``within`` of now, or weekly on
    ``weekday``, any hour —, ``once`` as said, a condition naming each of ``condition``
    (``None``: no condition)."""

    def check(chat: Chat) -> list[str]:
        z = zone or chat.zone or "UTC"
        s, problems = one_made(chat)
        if s is None:
            return problems
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


def or_asks_zone(check: Check) -> Check:
    """``check``, or — the user's zone not known — nothing written and their zone asked."""

    def either(chat: Chat) -> list[str]:
        problems = check(chat)
        text = chat.answers[-1].lower()
        asked = asks(text) and ("zone" in text or "utc" in text)
        return [] if problems and asked and not chat.written() else problems

    return either


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
            if first or not asks(chat.answers[0]):
                problems.append("did not confirm before cancelling all")
        return problems

    return check


def asked_which(chat: Chat) -> list[str]:
    problems = [f"wrote {n}" for n in chat.written()]
    if not asks(chat.answers[-1]):
        problems.append("did not ask which")
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
    S, N, M, F, L = CATEGORIES
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
        Case("relative-2h", S, ("In 2 hours, remind me to call supplier SUP-03.",),
             scheduled(once=True, within=hour)),
        Case("cond-hourly", S,
             (f"Check every hour and tell me if the total stock of {SKU} is 5 or below.",),
             scheduled("0 * * * *", condition=(SKU, "5"))),
        Case("cond-monday", S,
             (f"Every Monday at 8, check the stock of {SKU} and tell me if it's under 20.",),
             scheduled("0 8 * * 1", condition=("20",))),
        Case("hinglish-daily", S, ("Har roz subah 9 baje mujhe sales summary bhejna.",),
             scheduled("0 9 * * *")),
        Case("hinglish-cond", S,
             (f"Har Monday 10 baje {SKU} ka stock check karo, agar 10 se kam ho toh batana.",),
             scheduled("0 10 * * 1", condition=("10",))),
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
             scheduled("0 9 * * 1", condition=("15",)), context=(skill,)),
        # --- should not schedule
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
        Case("cancel-all", M, ("Cancel all my schedules.", "Yes, cancel them all."),
             cancelled(0, 1, ask_first=True), seeds=(stock_seed, summary_seed)),
        Case("ambiguous", M, ("Cancel my stock check.",), asked_which,
             seeds=(stock_seed, other_seed)),
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
        # --- limits
        Case("over-cap", L, ("Also every Friday at 5pm send me the sales summary.",),
             limited("cancel", "limit", "maximum", "1 of 1", "already"),
             seeds=(stock_seed,), limits=ScheduleLimits(max_per_user=1)),
        Case("too-often", L, (f"Check the stock of {SKU} every 5 minutes.",),
             limited("hour", "60 minutes", "often", "frequent", "least", "minimum"),
             limits=ScheduleLimits(min_interval=3600)),
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


async def _seed(h: Any, agent: Any, user: str, thread: str, seed: Seed, *, zone: str) -> Schedule:
    """``seed`` made as ``schedule_task`` makes it, in its zone or ``zone``."""
    about = {"task": seed.task, "condition": seed.condition, "once": False}

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
    )
    return await h.runs.schedules.create(spec)


async def run_case(case: Case, model: str, tag: str, max_steps: int = 12) -> dict[str, Any]:
    from benchmark.cost.complex import target_of

    h = runner.harness()
    if case.limits is not None:
        limits = case.limits

        async def hook(user: str, tenant: str) -> ScheduleLimits:
            return limits

        h.schedule_limits = hook
    async with h:
        assert isinstance(h.runs, LocalRuns)
        target, options = target_of(model, "thread", None, max_steps, adapter="react")
        trace = Calls()
        agent = h.wrap(target, id=f"sched-{case.id}", hooks=[trace], run_timeout=300, **options)
        user, thread = f"sched-{tag}-{case.id}", f"sched-{tag}-{case.id}"
        chat = Chat(now=datetime.now(UTC), zone=case.zone)
        zone = case.zone or "UTC"
        chat.seeded = [await _seed(h, agent, user, thread, s, zone=zone) for s in case.seeds]
        app = Frontend(context=case.context, timezone=case.zone)
        started = time.monotonic()
        with runner.metered() as meter:
            for text in case.turns:
                trace.calls = []
                record = await agent._opened(text, user=user, thread=thread, tenant=None, app=app)
                result = await pipeline.attempt(agent, record, text)
                chat.answers.append(str(result.answer or result.error or result.status))
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
    by = {c: [x for x in rows if x["category"] == c] for c in CATEGORIES}
    calls = sum(x["model_calls"] for x in rows) or 1
    return {
        "passed": sum(x["passed"] for x in rows),
        "cases": len(rows),
        "by_category": {c: f"{sum(x['passed'] for x in v)}/{len(v)}" for c, v in by.items() if v},
        "input_tokens_per_request": round(sum(x["input_tokens"] for x in rows) / calls),
        "model_calls_per_case": round(calls / max(len(rows), 1), 2),
        "seconds_per_case": round(sum(x["seconds"] for x in rows) / max(len(rows), 1), 2),
        "usd": round(sum(x["usd"] for x in rows), 4),
        "failed": {x["case"]: x["problems"] for x in rows if not x["passed"]},
    }


async def bench(args: argparse.Namespace) -> dict[str, Any]:
    wanted = {c for c in args.cases.split(",") if c}
    chosen = [c for c in cases() if not wanted or c.id in wanted or c.category in wanted]
    rows: list[dict[str, Any]] = []
    spent = 0.0
    tag = args.tag or datetime.now(UTC).strftime("%H%M%S")
    for repeat in range(args.repeats):
        for case in chosen:
            if args.max_usd is not None and spent >= args.max_usd:
                break
            row = await run_case(case, args.model, f"{tag}{repeat}")
            row["repeat"] = repeat
            spent += row["usd"]
            rows.append(row)
            _kept(f"{args.out}.jsonl", row, append=True)
            mark = "ok  " if row["passed"] else "FAIL"
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
    p.add_argument("--cases", default="", help="case ids or categories, comma-separated")
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
