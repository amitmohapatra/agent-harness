"""When and how the agent asks its user (``agent.ASKING``, ``ask_user``): does it ask when it
should, only then, in the right shape, and does it say the remembered values it used?

Each case is one chat with the retail tools (``retail.py``), four write tools of its own
(alerts, reorders, reports), ``ask_user`` and the case's memory context as the memory service
renders it (``## Profile``, dated ``standing_rules``, ``preferences`` and ``about_user``),
answered by an in-process memory service (``tests.support.memory``), ``ReAct`` (``--adapter``: any
framework the ``complex`` suite wraps). Categories:

* ``act`` — a clear request, a safe default, a decided value, "just do it": no question, the
  write made;
* ``ask`` — a conflict, which one, a trade-off, missing information with no safe default: a
  question, nothing written;
* ``shape`` — an open question without options, exclusive choices without several picks,
  combinable ones with them, a date or a number as its kind, a recommendation only for a reason;
* ``frontend`` — the app's own tool when it fits the question, else ``ask_user``;
* ``remembered`` — a remembered rule used, and said with its date;
* ``fill`` — values filled from an earlier one: said with their date when cheap or asked "as
  before", asked first when costly or old, the user's new values winning.

Every check is exact (the calls, the question's options, what the answer says). ``--note off``
leaves the note out (the arm before it), ``--no-ask-tool`` offers no ``ask_user``. Run::

    python -m benchmark.cost.asking --model claude-haiku-5-5 --max-usd 0.5 --out build/ask
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
from datetime import UTC, datetime
from typing import Any, Final

from tests.support.memory import FakeMemoryService

from benchmark.cost import prices
from benchmark.cost import run as runner
from benchmark.cost.complex import ADAPTERS, target_of
from benchmark.cost.schedules import Calls, _kept
from trellis import Harness, Settings, ask_user, tool
from trellis.contracts import Interrupt, RunStatus
from trellis.harness import agent as agent_module
from trellis.harness import pipeline
from trellis.harness.frontend import FRONTEND_TOOL, Frontend

CATEGORIES: Final = ("act", "ask", "shape", "frontend", "remembered", "fill")
WRITES: Final = ("create_alert", "reorder", "send_report", "delete_alerts")


async def create_alert(sku: str, threshold: int, region: str, channel: str) -> str:
    """Create a low-stock alert: tell the user by ``channel`` (email or slack) when a SKU's
    units in ``region`` fall under ``threshold``."""
    return f"alert created: {sku} under {threshold} in {region} by {channel}"


async def reorder(
    sku: str, quantity: int, supplier: str, store: str, deliver_on: str | None = None
) -> str:
    """Place a purchase order with a supplier for a store (costs money)."""
    return f"ordered {quantity} x {sku} from {supplier} for {store}" + (
        f" for {deliver_on}" if deliver_on else ""
    )


async def send_report(report: str, to: str, format: str) -> str:
    """Email a report (e.g. 'north sales summary') to an address, as csv or pdf."""
    return f"sent {report} to {to} as {format}"


async def delete_alerts(which: str) -> str:
    """Delete low-stock alerts: 'all', a SKU, or an alert's description."""
    return f"deleted {which}"


TOOLS: Final = [tool(f, side_effects="write") for f in (create_alert, reorder, send_report,
                                                        delete_alerts)]  # fmt: skip
PICK_DATE: Final = {
    "name": "pick_date",
    "description": "Show the person a date picker in the app; returns the date they pick.",
    "parameters": {"type": "object", "properties": {"label": {"type": "string"}}},
}
PICK_STORE: Final = {
    "name": "pick_store",
    "description": "Show the person their stores in the app to pick one; returns its id.",
    "parameters": {"type": "object", "properties": {"label": {"type": "string"}}},
}


def memory(rules: Sequence[str] = (), prefs: Sequence[str] = (), about: Sequence[str] = ()) -> str:
    """The memory context as the service renders its profile sections."""
    blocks = [(n, ls) for n, ls in (("standing_rules", rules), ("preferences", prefs),
                                    ("about_user", about)) if ls]  # fmt: skip
    if not blocks:
        return ""
    return "## Profile\n" + "\n".join(f"### {n}\n" + "\n".join(f"- {x}" for x in ls)
                                      for n, ls in blocks)  # fmt: skip


@dataclass
class Chat:
    answers: list[str] = field(default_factory=list)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    #: the first question the run paused on (ask_user or a front-end tool), if any
    asked: Interrupt | None = None

    def wrote(self, name: str | None = None) -> list[dict[str, Any]]:
        return [a for n, a in self.calls if n in WRITES and (name is None or n == name)]

    def via(self) -> str | None:
        """How it asked: ``ask_user``, a front-end tool's name, ``text``, or ``None``."""
        if self.asked is not None:
            front = (self.asked.payload or {}).get(FRONTEND_TOOL)
            return front or self.asked.component or "ask"
        if not self.wrote() and "?" in (self.answers[-1] if self.answers else ""):
            return "text"
        return None


Check = Callable[[Chat], list[str]]


def _said(text: str, words: Sequence[str]) -> list[str]:
    low = text.lower()
    return [f"answer lacks {w!r}" for w in words if not re.search(w, low)]


def day(iso: str) -> str:
    """A pattern for a date as an answer may say it (2026-11-02, 2 Nov, November 2, 11/02)."""
    y, m, d = iso.split("-")
    mon = datetime(int(y), int(m), int(d)).strftime("%b").lower()
    n = str(int(d))
    return rf"{iso}|\b{n}(st|nd|rd|th)? {mon}|{mon}\w* {n}\b|{m}/{d}|{d}/{m}|{n}/{int(m)}"


def acts(name: str, *, args: dict[str, str] | None = None, says: Sequence[str] = ()) -> Check:
    """No question; ``name`` called (with each of ``args`` in its argument, any case); the
    answer says each of ``says`` (patterns)."""

    def check(chat: Chat) -> list[str]:
        problems = [f"asked ({chat.via()})"] if chat.via() else []
        made = chat.wrote(name) if name in WRITES else [a for n, a in chat.calls if n == name]
        if not made:
            return [*problems, f"{name} not called"]
        for key, want in (args or {}).items():
            if not any(want.lower() in str(a.get(key, "")).lower() for a in made):
                problems.append(f"{name} {key}={[a.get(key) for a in made]}, not {want!r}")
        return problems + _said(chat.answers[-1], says)

    return check


def asks(
    *,
    tool_only: bool = False,
    options: bool | None = None,
    multiple: bool | None = None,
    kind: str | None = None,
    recommended: str | bool | None = None,
    via: str | None = None,
) -> Check:
    """A question before anything is written: through ``ask_user`` (``tool_only``) or any way;
    asked through ``ask_user``, its shape as given — ``options`` some or none, ``multiple``,
    ``kind``, ``recommended`` (an option's words, ``True`` any, ``False`` none); ``via`` a
    front-end tool's name."""

    def check(chat: Chat) -> list[str]:
        how = chat.via()
        if how is None:
            return ["did not ask"]
        problems = [f"wrote {n}" for n in (c[0] for c in chat.calls) if n in WRITES]
        if via is not None:
            return problems + ([] if how == via else [f"asked via {how}, not {via}"])
        if tool_only and how != "ask_user":
            return [*problems, f"asked via {how}, not ask_user"]
        q = chat.asked
        if q is None or q.component != "ask_user":
            return problems
        props = q.props or {}
        if options is not None and bool(q.options) != options:
            problems.append(f"options {q.options} on {'a choice' if options else 'an open'} ask")
        if multiple is not None and q.multiple != multiple:
            problems.append(f"multiple={q.multiple}")
        if kind is not None and props.get("kind") != kind:
            problems.append(f"kind={props.get('kind')}")
        best = [o for o in q.options if not isinstance(o, str) and o.recommended]
        if recommended is False and best:
            problems.append(f"recommended {best[0].value!r} with no reason")
        if recommended is True and not best:
            problems.append("no option recommended")
        if isinstance(recommended, str) and not (
            best and recommended.lower() in best[0].value.lower()
        ):
            problems.append(f"recommended {[b.value for b in best]}, not {recommended!r}")
        return problems

    return check


def both(*checks: Check) -> Check:
    return lambda chat: [p for c in checks for p in c(chat)]


def either(*checks: Check) -> Check:
    """Any one of ``checks`` passes; else the first one's problems."""

    def check(chat: Chat) -> list[str]:
        found = [c(chat) for c in checks]
        return [] if any(not p for p in found) else found[0]

    return check


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    turns: tuple[str, ...]
    check: Check
    memory: str = ""
    frontend: tuple[dict[str, Any], ...] = ()


LAST_ALERT: Final = memory(about=["Your last low-stock alert: SKU-00012, threshold 20, region "
                                  "north, channel email. (said 2026-11-02)"])  # fmt: skip
LAST_ORDER: Final = "Your last reorder: SKU-00012, 200 units from SUP-04 for store S12."
TWO_PRIYAS: Final = memory(about=[
    "Two people named Priya on my team: Priya Shah (buying, priya.shah@acme.com) and Priya "
    "Nair (finance, priya.nair@acme.com). (said 2026-10-10)"])  # fmt: skip
ALERTS: Final = memory(about=[
    "My low-stock alerts: SKU-00012 north at 20 by email (made 2026-03-02), SKU-00031 south "
    "at 10 by slack (made 2026-09-14), SKU-00044 east at 5 by email (made 2026-10-30). "
    "(said 2026-10-30)"])  # fmt: skip
SUMMARY: Final = "Send the north sales summary to ops@acme.com"


def cases() -> list[Case]:
    a, b = "act", "ask"
    return [
        # ------------------------------------------------------------------ act
        Case("act-read", a, ("How many units of SKU-00012 are in stock in the north region?",),
             acts("stock_level")),
        Case("act-given", a, ("Set a low-stock alert for SKU-00012 at 10 units in the north, "
                              "by email.",), acts("create_alert", args={"threshold": "10"})),
        Case("act-safe-default", a, ("Set a low-stock alert for SKU-00012 at 15 units in the "
                                     "north.",), acts("create_alert", args={"threshold": "15"})),
        Case("act-decided-pref", a, ("Create a low-stock alert for SKU-00031 at 10 units in the "
                                     "south.",),
             acts("create_alert", args={"channel": "slack"}, says=[day("2026-10-20")]),
             memory=memory(prefs=["Alerts go to Slack #ops. (said 2026-10-20)"])),
        Case("act-just-do-it", a, ("Just do it, don't ask me anything: reorder 40 units of "
                                   "SKU-00012 for store S12 from my usual supplier.",),
             acts("reorder", args={"supplier": "SUP-04"}),
             memory=memory(about=["My usual supplier for SKU-00012 is SUP-04. "
                                  "(said 2026-10-28)"])),
        Case("act-findable", a, ("Which supplier of SKU-00012 has the shortest lead time, and "
                                 "what is it?",), acts("sku_details")),
        Case("act-decided-chat", a, (f"{SUMMARY} as a PDF.",
                                     "Now send the same to finance@acme.com."),
             acts("send_report", args={"to": "finance", "format": "pdf"})),
        Case("act-rule", a, (f"{SUMMARY}.",),
             acts("send_report", args={"format": "csv"}, says=[day("2026-10-15")]),
             memory=memory(rules=["Always send reports as CSV. (said 2026-10-15)"])),
        Case("act-all-given", a, ("Reorder 300 units of SKU-00031 for store S03 from SUP-01.",),
             acts("reorder", args={"quantity": "300"})),
        Case("act-named-delete", a, ("Delete my low-stock alert for SKU-00044.",),
             acts("delete_alerts", args={"which": "00044"}), memory=ALERTS),
        # ------------------------------------------------------------------ ask
        Case("ask-which-one", b, ("Send the weekly sales summary to Priya.",), asks(),
             memory=TWO_PRIYAS),
        Case("ask-rule-conflict", b, ("Reorder 100 units of SKU-00012 for store S12 from "
                                      "SUP-07.",), asks(),
             memory=memory(rules=["Never reorder from SUP-07. (said 2026-10-01)"])),
        Case("ask-missing-qty", b, ("Reorder SKU-00031 for store S07.",), asks()),
        Case("ask-missing-to", b, ("Send the Q3 margin report.",), asks(options=False)),
        Case("ask-trade-off", b, ("Reorder 200 units of SKU-00012 for store S12. SUP-07 is "
                                  "cheaper, SUP-04 delivers a week sooner.",), asks()),
        Case("ask-scope", b, ("Delete my old alerts.",), asks(), memory=ALERTS),
        Case("ask-superseded", b, ("Send this week's sales report to ops@acme.com.",),
             asks(recommended="pdf"),
             memory=memory(rules=["Send weekly reports as PDF. (said 2026-11-02; earlier: Send "
                                  "weekly reports as CSV., said 2026-10-01)"])),
        # a second alert is cheap to undo and does not contradict the first (20 warns early,
        # 5 late): asking, or acting and naming the one they have, both follow the note
        Case("ask-existing", b, ("Set a low-stock alert for SKU-00012 at 5 units in the north "
                                 "by email.",),
             either(asks(), acts("create_alert", args={"threshold": "5"}, says=[r"\b20\b"])),
             memory=memory(about=["You already have a low-stock alert for SKU-00012 at 20 "
                                  "units in the north by email. (said 2026-10-25)"])),
        # ------------------------------------------------------------------ shape
        Case("shape-open", "shape", ("Send the north sales summary to our new regional manager: "
                                     "ask me their email address.",),
             asks(tool_only=True, options=False)),
        Case("shape-exclusive", "shape", ("Reorder 50 units of SKU-00012 for store S12. Ask me "
                                          "which supplier: SUP-07 or SUP-04.",),
             asks(tool_only=True, options=True, multiple=False, recommended=False)),
        Case("shape-combine", "shape", ("Create low-stock alerts for SKU-00031 at 10 units by "
                                        "email in the regions I choose: ask me which regions.",),
             asks(tool_only=True, options=True, multiple=True)),
        Case("shape-neutral", "shape", (f"{SUMMARY}. Ask me which format, CSV or PDF; I have no "
                                        "preference yet.",),
             asks(tool_only=True, options=True, multiple=False, recommended=False)),
        Case("shape-reason", "shape", (f"{SUMMARY}. Ask me which format, CSV or PDF.",),
             asks(tool_only=True, options=True, recommended="pdf"),
             memory=memory(prefs=["I usually pick PDF for reports. (said 2026-10-12)"])),
        Case("shape-date", "shape", ("Reorder 30 units of SKU-00012 for S12 from SUP-04. Ask "
                                     "me the delivery date.",),
             asks(tool_only=True, options=False, kind="date")),
        Case("shape-number", "shape", ("Create a low-stock alert for SKU-00012 in the north by "
                                       "email. Ask me the threshold.",),
             asks(tool_only=True, options=False, kind="number")),
        Case("shape-safer", "shape", ("Ask me first: should I delete all three low-stock alerts "
                                      "or only the oldest one?",),
             asks(tool_only=True, options=True, recommended="old"), memory=ALERTS),
        # ------------------------------------------------------------------ frontend
        Case("front-date", "frontend", ("Reorder 30 units of SKU-00012 for S12 from SUP-04. "
                                        "Ask me the delivery date.",),
             asks(via="pick_date"), frontend=(PICK_DATE,)),
        Case("front-store", "frontend", ("Reorder 30 units of SKU-00012 from SUP-04 for one of "
                                         "my stores: ask me which store.",),
             asks(via="pick_store"), frontend=(PICK_STORE,)),
        Case("front-not-date", "frontend", (f"{SUMMARY}. Ask me the format, CSV or PDF.",),
             asks(via="ask_user"), frontend=(PICK_DATE,)),
        Case("front-not-store", "frontend", ("Reorder SKU-00012 from SUP-04 for store S12. Ask "
                                             "me how many units.",),
             asks(via="ask_user"), frontend=(PICK_STORE,)),
        # ------------------------------------------------------------------ remembered
        Case("mem-rule-dated", "remembered", ("Send the weekly sales summary.",),
             acts("send_report", args={"to": "ops@acme.com"}, says=[day("2026-10-15")]),
             memory=memory(rules=["Always send reports to ops@acme.com as CSV. "
                                  "(said 2026-10-15)"])),
        Case("mem-pref-dated", "remembered", ("Set a low-stock alert for SKU-00012 at 10 units "
                                              "in the east.",),
             acts("create_alert", args={"channel": "email"}, says=[day("2026-10-18")]),
             memory=memory(prefs=["Alerts come to me by email. (said 2026-10-18)"])),
        Case("mem-rule-applies", "remembered", ("How many units of SKU-00012 are in stock?",),
             acts("stock_level", says=["north", day("2026-11-02")]),
             memory=memory(prefs=["Only show me the north region. (said 2026-11-02)"])),
        # ------------------------------------------------------------------ fill
        Case("fill-stated", "fill", ("Create a low-stock alert for SKU-00031 too.",),
             acts("create_alert", args={"threshold": "20", "region": "north",
                                        "channel": "email"},
                  says=["20", "north", "email", day("2026-11-02")]), memory=LAST_ALERT),
        Case("fill-as-before", "fill", ("Reorder SKU-00012 again, same as last time.",),
             acts("reorder", args={"quantity": "200", "supplier": "SUP-04", "store": "S12"},
                  says=["200", day("2026-11-02")]),
             memory=memory(about=[f"{LAST_ORDER} (said 2026-11-02)"])),
        Case("fill-costly", "fill", ("Reorder SKU-00031 too.",), asks(),
             memory=memory(about=[f"{LAST_ORDER} (said 2026-11-02)"])),
        Case("fill-new-wins", "fill", ("Create a low-stock alert for SKU-00031 at 5 units.",),
             acts("create_alert", args={"threshold": "5", "region": "north"},
                  says=["north", day("2026-11-02")]), memory=LAST_ALERT),
        Case("fill-old", "fill", ("Reorder SKU-00012 again.",), asks(),
             memory=memory(about=[f"{LAST_ORDER} (said 2025-12-01)"])),
        Case("fill-as-before-new-to", "fill", ("Send the report again like before, to "
                                               "finance@acme.com.",),
             acts("send_report", args={"to": "finance", "format": "pdf", "report": "north"}),
             memory=memory(about=["Your last report sent: north sales summary, PDF, to "
                                  "ops@acme.com. (said 2026-11-02)"])),
    ]  # fmt: skip


async def run_case(
    case: Case,
    model: str,
    tag: str,
    *,
    ask_tool: bool = True,
    max_steps: int = 12,
    adapter: str = "react",
) -> dict[str, Any]:
    from benchmark.cost import retail as r

    service = FakeMemoryService(context_text=case.memory, shown=[])
    config = Settings.from_env().model_copy(update={"runs_url": None, "otlp_endpoint": None})
    tools = [*r.TOOLS, *TOOLS, *([ask_user()] if ask_tool else [])]
    async with Harness(config=config, memory=service.client(), gateway=False) as h:
        target, options = target_of(model, "thread", None, max_steps, adapter=adapter, tools=tools)
        trace = Calls()
        agent = h.wrap(target, id=f"ask-{case.id}", hooks=[trace], run_timeout=300,
                       without=["tool_search"], **options)  # fmt: skip
        user = thread = f"ask-{tag}-{case.id}"
        app = Frontend(tools=case.frontend)
        chat = Chat()
        started = time.monotonic()
        with runner.metered(adapter=adapter) as meter:
            for text in case.turns:
                trace.calls = []
                record = await agent._opened(text, user=user, thread=thread, tenant=None, app=app)
                result = await pipeline.attempt(agent, record, text)
                chat.calls += [(n, a) for n, a, _ in trace.calls]
                if result.status is RunStatus.PAUSED:
                    chat.asked = result.interrupt
                    chat.answers.append(result.interrupt.question if result.interrupt else "")
                    break
                chat.answers.append(str(result.answer or result.error or result.status))
        problems = case.check(chat)
    price = prices.price_of(model)
    usd = (
        prices.dollars(price, input_tokens=meter.input_tokens, output_tokens=meter.output_tokens,
                       cached=meter.cached_tokens, written=meter.written_tokens)
        if price else 0.0
    )  # fmt: skip
    q = chat.asked
    return {
        "case": case.id,
        "category": case.category,
        "passed": not problems,
        "problems": problems,
        "answers": [x[:500] for x in chat.answers],
        "calls": chat.calls,
        "asked": None if q is None else q.model_dump(mode="json", include={
            "question", "options", "multiple", "props", "component", "payload"}),
        "model_calls": meter.calls,
        "input_tokens": meter.input_tokens,
        "seconds": round(time.monotonic() - started, 2),
        "usd": usd,
    }  # fmt: skip


def summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by = {c: [x for x in rows if x["category"] == c] for c in CATEGORIES}
    calls = sum(x["model_calls"] for x in rows) or 1
    return {
        "passed": sum(x["passed"] for x in rows),
        "cases": len(rows),
        "by_category": {c: f"{sum(x['passed'] for x in v)}/{len(v)}" for c, v in by.items() if v},
        "input_tokens_per_request": round(sum(x["input_tokens"] for x in rows) / calls),
        "seconds_per_case": round(sum(x["seconds"] for x in rows) / max(len(rows), 1), 2),
        "usd": round(sum(x["usd"] for x in rows), 4),
        "failed": {x["case"]: x["problems"] for x in rows if not x["passed"]},
    }


async def bench(args: argparse.Namespace) -> dict[str, Any]:
    if args.note == "off":
        agent_module.ASKING = ""  # type: ignore[misc]  # the arm before the note
    wanted = {c for c in args.cases.split(",") if c}
    chosen = [c for c in cases() if not wanted or c.id in wanted or c.category in wanted]
    rows: list[dict[str, Any]] = []
    tag = args.tag or datetime.now(UTC).strftime("%H%M%S")
    for case in chosen:
        if args.max_usd is not None and sum(x["usd"] for x in rows) >= args.max_usd:
            break
        row = await run_case(
            case, args.model, tag, ask_tool=not args.no_ask_tool, adapter=args.adapter
        )
        rows.append(row)
        _kept(f"{args.out}.jsonl", row, append=True)
        mark = "ok  " if row["passed"] else "FAIL"
        sys.stderr.write(f"{mark} {case.id}: {'; '.join(row['problems'])}\n")
    report = summary(rows)
    _kept(f"{args.out}.json", report)
    return report


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    env = os.environ.get
    p = argparse.ArgumentParser(prog="python -m benchmark.cost.asking", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--model", default=env("TRELLIS_BENCH_MODEL") or env("TRELLIS_LIVE_MODEL"))
    p.add_argument("--adapter", choices=ADAPTERS, default="react", help="the framework wrapped")
    p.add_argument("--cases", default="", help="case ids or categories, comma-separated")
    p.add_argument("--note", choices=("on", "off"), default="on")
    p.add_argument("--no-ask-tool", action="store_true", help="offer no ask_user")
    p.add_argument("--max-usd", type=float, default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="build/bench-asking")
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
