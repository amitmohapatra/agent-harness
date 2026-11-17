"""Whole workflows on the live model, on every adapter (``tests/live/engines.py``): a skill a
team uploaded (a ``SKILL.md`` folder, ``skills.directory``) leads the model through an ordered
procedure with a branch; what a user said in one run decides another's result, and never
another user's; a file one run kept answers a later run's question; a skill's write waits for
a person, runs once, and its outcome is kept as a file and remembered; a skill hands a lookup
to a sub-agent; a skill's "check again later" becomes a one-time schedule.

What is asserted is what a model cannot fake: the tools' own ledger (which ran, in which
order, with which arguments, how often), the skill's load among the run's calls, the file as
the store keeps it, the child run's record, the schedule as the run store keeps it, and figures
computed from data seeded per test. Never the answer's wording.

Writes here run announced, unasked (``writes_unasked``): what a workflow does, not whether a
plain write asks. The write that waits for a person is ``irreversible``, which always asks."""

from __future__ import annotations

import random
import uuid
from pathlib import Path
from typing import Any, Final

import pytest

import trellis
from tests.live.conftest import live_harness, needs_memory
from tests.live.engines import ENGINES, Kit, engines, made, says
from tests.live.proof import TEST_SECONDS
from tests.live.support import eventually, memory_scope
from trellis import Runtime, skills, tool
from trellis.contracts import RunStatus
from trellis.harness.schedules import TASK
from trellis.harness.skills import SkillSet

pytestmark = [pytest.mark.live, pytest.mark.usefixtures("writes_unasked")]

MONTH: Final = "2026-10"
#: An order over this is flagged at the month's close.
THRESHOLD: Final = 1000
#: The engines offered ``read_file`` for a file kept for the user: where the harness gives the
#: run its tools per model call or per attempt. A plain ``create_agent`` or Deep Agents graph
#: runs its own loop, and no reader is added (docs/frameworks/langgraph.md, "Large results and
#: compute"); a function's code is offered none (``tools/stored.py``, ``looks``). Kimi Code is
#: left out: its own shell runs on this machine in the live suite's deployment
#: (``TRELLIS_HOST_EXECUTION=allow``), where it finds the local file store's files itself.
READERS: Final = ("react", "middleware", "openai_agents", "claude_agent_sdk")


def uploaded(folder: Path, name: str, description: str, body: str) -> SkillSet:
    """A skill as its team keeps it: ``<name>/SKILL.md`` in a folder, read at each run."""
    home = folder / "skills" / name
    home.mkdir(parents=True)
    (home / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n")
    return skills.directory(folder / "skills")


@pytest.mark.timeout(TEST_SECONDS)
@pytest.mark.parametrize("engine", ENGINES, indirect=True)
async def test_a_skill_leads_a_months_close_through_its_steps_and_its_branch(
    engine: str, tmp_path: Path
) -> None:
    suffix = uuid.uuid4().hex[:8]
    seeded = random.Random(suffix)
    amounts = [*seeded.sample(range(100, THRESHOLD), 2), *seeded.sample(range(1001, 5000), 2)]
    seeded.shuffle(amounts)
    orders = {f"ORD-{suffix}-{n}": amount for n, amount in enumerate(amounts, 1)}
    over = sorted(order for order, amount in orders.items() if amount > THRESHOLD)
    ledger: list[tuple[str, Any]] = []

    @tool(name=f"open_orders_{suffix}", side_effects="read")
    def open_orders(month: str) -> list[dict[str, Any]]:
        """The open orders of a month (YYYY-MM): each one's id and amount."""
        ledger.append(("open", month))
        return [{"order": order, "amount": amount} for order, amount in orders.items()]

    @tool(name=f"flag_order_{suffix}", side_effects="write")
    def flag_order(order: str) -> str:
        """Flag an order for review, by its id."""
        ledger.append(("flag", order))
        return f"flagged {order}"

    @tool(name=f"save_summary_{suffix}", side_effects="write")
    async def save_summary(file_name: str, csv: str) -> str:
        """Keep a CSV summary as a file for the user."""
        runtime = trellis.current()
        assert runtime is not None
        kept = await runtime.save_file(
            file_name, csv.encode(), source=f"save_summary_{suffix}", content_type="text/csv"
        )
        ledger.append(("save", (runtime.files_folder(), kept.name)))
        return f"saved {kept.name}"

    skill = uploaded(
        tmp_path,
        "close-month",
        "How this team closes a month's books. Read it before closing any month.",
        "# Closing a month\n"
        "Do these steps in this order, each once.\n"
        f"1. Call `open_orders_{suffix}` with the month to list its open orders.\n"
        f"2. For each order whose amount is over {THRESHOLD}, and for no other, call "
        f"`flag_order_{suffix}` with its id.\n"
        f"3. Call `save_summary_{suffix}` with `file_name` `close-{MONTH}.csv` and `csv` a "
        "header line `order,amount` then one line for each flagged order.\n"
        "4. Answer with how many orders you flagged and the sum of their amounts.",
    )
    kit = Kit([open_orders, flag_order, save_summary], skills=skill)
    async with live_harness() as h:
        close = await made(engine, h, kit, tmp_path, "close")
        done = await close.run(
            f"Close the month {MONTH}, following the close-month skill.", user=f"live-{suffix}"
        )
        assert done.status is RunStatus.SUCCESS, done
        assert close.seen.loaded("close-month"), [call.tool for call in close.seen.asked]
        steps = [step for step, _ in ledger]
        assert steps[0] == "open" and MONTH in str(ledger[0][1]), ledger
        flagged = [what for step, what in ledger if step == "flag"]
        assert sorted(flagged) == over, ledger  # the branch: those over the threshold, once each
        [(folder, name)] = [what for step, what in ledger if step == "save"]
        assert steps.index("save") > len(steps) - 1 - steps[::-1].index("flag"), steps
        summary = (await h.files.read(folder, name) or b"").decode()
    assert all(order in summary for order in over), summary
    assert not any(order in summary for order in orders if order not in over), summary
    assert says(done.answer, sum(orders[order] for order in over)), done.answer


@needs_memory
@pytest.mark.timeout(TEST_SECONDS)
@pytest.mark.parametrize("engine", ENGINES, indirect=True)
async def test_what_a_user_said_in_one_run_decides_anothers_result_and_no_other_users(
    engine: str, tmp_path: Path
) -> None:
    suffix = uuid.uuid4().hex[:8]
    user, stranger = f"live-ann-{suffix}", f"live-bob-{suffix}"
    customer, tier = f"Harbor-{suffix}", f"T{suffix[:5].upper()}"
    currency = random.Random(suffix).choice(["ISK", "HUF", "CLP", "NOK", "PLN"])
    quoted: list[dict[str, str]] = []

    @tool(name=f"quote_{suffix}", side_effects="read")
    def quote(customer: str, tier: str, currency: str) -> str:
        """A quote for a customer: it needs the customer's tier and the currency to quote in
        (an ISO code), exactly as they are known."""
        quoted.append({"customer": customer, "tier": tier, "currency": currency})
        return f"quote Q-{len(quoted)} for {customer}"

    async with live_harness() as h:
        sales = await made(engine, h, Kit([quote]), tmp_path, "sales")
        told = await sales.run(
            f"Remember these for later: my reporting currency is {currency}, and the customer "
            f"{customer} is in tier {tier}. That is all for now.",
            user=user,
            thread=f"told-{suffix}",
        )
        assert told.status is RunStatus.SUCCESS and quoted == [], told
        await h.writes.drain()
        scope = await memory_scope(h, user=user, agent_id=sales.agent.id)

        async def remembered() -> bool:
            said = " ".join(item.text for item in await scope.search(f"{customer} tier currency"))
            return tier in said and currency in said

        assert await eventually(remembered), "the run kept neither fact for its user"
        ask = f"Get a quote for the customer {customer} in my reporting currency."
        done = await sales.run(ask, user=user, thread=f"asked-{suffix}")
        assert done.status is RunStatus.SUCCESS, done
        assert quoted and quoted[-1]["tier"].strip().upper() == tier, (quoted, done.answer)
        assert quoted[-1]["currency"].strip().upper() == currency, quoted
        before = len(quoted)
        other = await sales.run(ask, user=stranger, thread=f"asked-{suffix}")
    assert other.status in (RunStatus.SUCCESS, RunStatus.PAUSED), other
    guessed = [call["tier"].strip().upper() for call in quoted[before:]]
    assert tier not in guessed and tier not in str(other.answer), (guessed, other.answer)


@pytest.mark.timeout(TEST_SECONDS)
@pytest.mark.parametrize("engine", engines(only=READERS), indirect=True)
async def test_a_file_one_run_kept_answers_a_later_runs_question(
    engine: str, tmp_path: Path
) -> None:
    suffix = uuid.uuid4().hex[:8]
    user, thread = f"live-ann-{suffix}", f"exports-{suffix}"
    exported: list[tuple[str, str]] = []

    @tool(name=f"export_orders_{suffix}", side_effects="write")
    async def export_orders() -> str:
        """Export this week's orders to a CSV file kept for the user."""
        runtime = trellis.current()
        assert runtime is not None
        batch = f"BATCH-{uuid.uuid4().hex[:10].upper()}"  # each export's own
        rows = f"batch,order,units\n{batch},ORD-1,4\n{batch},ORD-2,9\n"
        kept = await runtime.save_file(
            f"orders-{suffix}.csv", rows.encode(), source=f"export_orders_{suffix}"
        )
        exported.append((kept.name, batch))
        return f"exported 2 orders to {kept.name}"

    async with live_harness() as h:
        clerk = await made(engine, h, Kit([export_orders]), tmp_path, "clerk")
        first = await clerk.run("Export this week's orders.", user=user, thread=thread)
        assert first.status is RunStatus.SUCCESS and len(exported) == 1, first
        [(name, batch)] = exported
        ask = f"Read the file {name} you exported. Which batch code do its rows carry?"
        second = await clerk.run(ask, user=user, thread=thread)
        # the code is in the file alone: the run's answer of its export never said it
        assert second.status is RunStatus.SUCCESS and batch in str(second.answer), second
        assert len(exported) == 1, "the question was answered from the file, not a new export"
        other = await clerk.run(ask, user=f"live-bob-{suffix}", thread=thread)
    assert batch not in str(other.answer), "another user's file is never read"


@needs_memory
@pytest.mark.timeout(TEST_SECONDS)
@pytest.mark.parametrize("engine", ENGINES, indirect=True)
async def test_a_skills_write_waits_for_a_person_runs_once_and_its_outcome_is_kept(
    engine: str, tmp_path: Path
) -> None:
    suffix = uuid.uuid4().hex[:8]
    user, customer = f"live-ann-{suffix}", f"Quay-{suffix}"
    seeded = random.Random(suffix)
    days = seeded.sample(range(10, 400), 3)
    invoices = {
        f"INV-{uuid.uuid4().hex[:6].upper()}": (age, seeded.randrange(200, 9000)) for age in days
    }
    oldest = max(invoices, key=lambda invoice: invoices[invoice][0])
    written: list[dict[str, Any]] = []
    receipts: list[tuple[str, str]] = []

    @tool(name=f"overdue_{suffix}", side_effects="read")
    def overdue(customer: str) -> list[dict[str, Any]]:
        """A customer's overdue invoices: each one's id, days overdue and amount."""
        gone = {call["invoice"] for call in written}
        return [
            {"invoice": invoice, "days_overdue": age, "amount": amount}
            for invoice, (age, amount) in invoices.items()
            if invoice not in gone
        ]

    @tool(name=f"write_off_{suffix}", side_effects="irreversible")
    def write_off(invoice: str, amount: int) -> str:
        """Write an invoice off, by its id and its amount. It cannot be undone."""
        written.append({"invoice": invoice, "amount": amount})
        return f"written off {invoice}"

    @tool(name=f"save_receipt_{suffix}", side_effects="write")
    async def save_receipt(file_name: str, text: str) -> str:
        """Keep a receipt as a text file for the user."""
        runtime = trellis.current()
        assert runtime is not None
        kept = await runtime.save_file(file_name, text.encode(), source=f"save_receipt_{suffix}")
        receipts.append((runtime.files_folder(), kept.name))
        return f"saved {kept.name}"

    skill = uploaded(
        tmp_path,
        "write-off",
        "How this team writes off a customer's overdue invoice. Read it before any write-off.",
        "# Writing off an overdue invoice\n"
        "Do these steps in this order, each once.\n"
        f"1. Call `overdue_{suffix}` with the customer to list its overdue invoices.\n"
        "2. Take the one invoice that is overdue the longest, and no other. Call "
        f"`write_off_{suffix}` with its id and its amount. A person approves this call.\n"
        f"3. Call `save_receipt_{suffix}` with `file_name` `write-off.txt` and `text` naming "
        "the invoice's id and its amount.\n"
        "4. Call `memory_remember` to keep, for the user, which invoice of which customer was "
        "written off.\n"
        "5. Answer with the invoice's id and its amount.",
    )
    kit = Kit([overdue, write_off, save_receipt], skills=skill)
    async with live_harness() as h:
        books = await made(engine, h, kit, tmp_path, "books")
        paused = await books.run(
            f"Write off the oldest overdue invoice of {customer}, following the write-off skill.",
            user=user,
            thread=f"write-off-{suffix}",
        )
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused
        asked = paused.interrupt.tool_call
        assert asked is not None and asked.tool == f"write_off_{suffix}", paused.interrupt
        assert asked.args["invoice"] == oldest and written == [], (asked, written)
        assert books.seen.loaded("write-off"), [call.tool for call in books.seen.asked]
        done = await books.resume(paused, "approve", user=user)
        assert done.status is RunStatus.SUCCESS, done
        assert [call["invoice"] for call in written] == [oldest], written  # once, approved
        assert len(receipts) == 1, receipts
        receipt = (await h.files.read(*receipts[0]) or b"").decode()
        assert oldest in receipt, receipt
        await h.writes.drain()
        scope = await memory_scope(h, user=user, agent_id=books.agent.id)

        async def remembered() -> bool:
            found = await scope.search(f"invoice written off for {customer}")
            return any(oldest in item.text for item in found)

        assert await eventually(remembered), "the outcome was not remembered"
        # the invoice is no longer listed as overdue: only memory names it
        recalled = await books.run(
            f"Which invoice of {customer} did we write off? Give its id.",
            user=user,
            thread=f"recall-{suffix}",
        )
    assert recalled.status is RunStatus.SUCCESS and oldest in str(recalled.answer), recalled
    assert len(written) == 1, written


@pytest.mark.timeout(TEST_SECONDS)
@pytest.mark.parametrize("engine", ENGINES, indirect=True)
async def test_a_skill_hands_a_lookup_to_a_sub_agent_and_computes_with_its_answer(
    engine: str, tmp_path: Path
) -> None:
    suffix = uuid.uuid4().hex[:8]
    seeded = random.Random(suffix)
    sku, units, price = (
        f"SKU-{suffix[:4].upper()}",
        seeded.randrange(11, 50),
        seeded.randrange(3, 10),
    )
    looked: list[str] = []

    async def researcher(input: str, agent: Runtime) -> str:
        """Looks up how many units of a SKU were sold, and its unit price."""
        looked.append(input)
        return f"{sku}: {units} units sold at {price} each"

    async with live_harness() as h:
        child = h.wrap(researcher, id=f"researcher-{suffix}")
        skill = uploaded(
            tmp_path,
            "sku-revenue",
            "How this team works out a SKU's revenue. Read it before answering about revenue.",
            "# A SKU's revenue\n"
            "1. Do not look the figures up yourself. Hand the lookup to the sub-agent: call "
            f"`{child.id}` with a `message` asking for the SKU's units sold and unit price.\n"
            "2. Multiply the units by the unit price.\n"
            "3. Answer with that revenue as a number.",
        )
        lead = await made(engine, h, Kit([child], skills=skill), tmp_path, "lead")
        done = await lead.run(
            f"What is the revenue of {sku}? Follow the sku-revenue skill.", user=f"live-{suffix}"
        )
        assert done.status is RunStatus.SUCCESS, done
        assert lead.seen.loaded("sku-revenue"), [call.tool for call in lead.seen.asked]
        kids = [kid async for kid in h.runs.iterate(parent_run_id=done.run_id)]
        assert [kid.agent_id for kid in kids] == [child.id], kids
        record = await h.runs.get(kids[0].run_id)
    assert record is not None and record.status is RunStatus.SUCCESS, record
    assert len(looked) == 1 and says(done.answer, units * price), (looked, done.answer)


@pytest.mark.timeout(TEST_SECONDS)
@pytest.mark.parametrize("engine", ENGINES, indirect=True)
async def test_a_skills_check_again_later_becomes_a_one_time_schedule(
    engine: str, tmp_path: Path
) -> None:
    """What the harness offers for "later" today: ``schedule_task`` (``in_minutes``: once, at
    that minute), which every adapter's run is offered while the run store can schedule. The
    step left over is a schedule's task: a fresh run of the same agent for the same user, in
    the same chat, when it fires — the run that set it does not wait, and nothing of its
    state is carried but the task's words. So what is true today is asserted: one one-time
    schedule of the run's user and agent, whose task names the export to check."""
    suffix = uuid.uuid4().hex[:8]
    user = f"live-ann-{suffix}"
    export = f"EXP-{uuid.uuid4().hex[:8].upper()}"
    started: list[str] = []

    @tool(name=f"start_export_{suffix}", side_effects="write")
    def start_export(dataset: str) -> str:
        """Start an export of a dataset. It takes about half an hour; its id is returned."""
        started.append(dataset)
        return f"export {export} started: running"

    @tool(name=f"export_status_{suffix}", side_effects="read")
    def export_status(export_id: str) -> str:
        """An export's status, by its id."""
        return f"{export_id}: running"

    skill = uploaded(
        tmp_path,
        "start-export",
        "How this team starts a data export. Read it before starting any export.",
        "# Starting an export\n"
        f"1. Call `start_export_{suffix}` with the dataset. It returns the export's id.\n"
        "2. An export takes about half an hour: do not wait for it, and do not check it now. "
        "Call `schedule_task` with `in_minutes` 30 and a `task` that says to check the status "
        f"of that export, naming its id, with `export_status_{suffix}`.\n"
        "3. Answer with the export's id.",
    )
    kit = Kit([start_export, export_status], skills=skill)
    async with live_harness() as h:
        exports = await made(engine, h, kit, tmp_path, "exports")
        tenant = await h._tenant()
        done = await exports.run(
            "Start an export of the orders dataset, following the start-export skill.",
            user=user,
            thread=f"exports-{suffix}",
        )
        page = await h.runs.schedules.list(
            agent_id=exports.agent.id, on_behalf_of=user, tenant=tenant
        )
        for schedule in page.items:  # nothing is left to fire once the test has read them
            await h.runs.schedules.delete(schedule.schedule_id, tenant=tenant)
    assert done.status is RunStatus.SUCCESS and len(started) == 1, (done, started)
    assert exports.seen.loaded("start-export"), [call.tool for call in exports.seen.asked]
    [schedule] = page.items
    asked = schedule.metadata[TASK]
    assert asked["once"] is True and len(schedule.cadence.split()) == 5, schedule
    assert export in asked["task"] and export in str(schedule.input), schedule
