"""A tool the harness runs that says it ``write``\\s asks a person before it runs, as Claude Code
asks: allow once, allow for this conversation, or deny — and, with a memory service, which keeps
it, always allow. "This conversation" is every later run of the same thread for the same user —
a resumed run, another worker and a child run included — never another thread, another user or
another tool (an ``irreversible`` tool asks every time: ``test_approvals``); "always" is every
conversation of that user until the permission is revoked.
A schedule set in chat allows the writes its task names when it is confirmed, and its fired runs
honour that; an uncovered write in a fired run pauses in the chat's thread. The frameworks' own
tools keep their own way (``test_govern_framework_tools``, ``test_claude``, ``test_kimi``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from tests.support.planned import Call
from trellis import Ask, Harness, Hooks, Runtime, Settings, tool
from trellis.contracts import ConfigurationError, InterruptReason, RunStatus, ToolCall, ToolSpec
from trellis.harness import conversation
from trellis.harness.governance.decision import ALWAYS, CONVERSATION, DENY, ONCE
from trellis.harness.runs import LocalRuns
from trellis.harness.schedules import ALLOW, ALLOWED, EACH_TIME, NEITHER
from trellis.harness.tools import bridge
from trellis.harness.tools.convert import text_of

done: list[str] = []
THREAD = "chat-1"
#: what plain chat reads after the question: the three choices, and the four with a memory
#: service, which keeps what a person allows always
THREE = "Allow once, allow for this conversation, or deny."
FOUR = "Allow once, allow for this conversation, always allow, or deny."


@tool(side_effects="write")
def pay(amount: int) -> str:
    """Pay an amount."""
    done.append(f"pay {amount}")
    return f"paid {amount}"


@tool(side_effects="write")
def refund(amount: int) -> str:
    """Refund an amount."""
    done.append(f"refund {amount}")
    return f"refunded {amount}"


@tool(side_effects="irreversible")
def wire(amount: int) -> str:
    """Wire an amount."""
    done.append(f"wire {amount}")
    return f"wired {amount}"


class OverHundred(Hooks):
    """A team's rule in code: a payment over 100 asks its own question."""

    async def before_tool(self, call: ToolCall) -> Ask | None:
        return Ask("Over 100: pay it?") if call.args.get("amount", 0) > 100 else None


@pytest.fixture(autouse=True)
def _reset() -> None:
    done.clear()


def calling(*calls: tuple[str, int]) -> Any:
    """A function target that makes ``calls`` in order and answers what they returned."""

    async def fn(input: str, agent: Runtime) -> list[str]:
        return [text_of(await agent.tools.call(name, amount=n)) for name, n in calls]

    return fn


async def approved(agent: Any, result: Any, remember: str = "once") -> Any:
    assert result.status is RunStatus.PAUSED and result.interrupt is not None, result
    return await agent.resume(
        result.interrupt.interrupt_id, "approve", reviewer="ada", remember=remember
    )


# --------------------------------------------------------------------------- a write asks


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_a_write_asks_with_three_choices_and_allow_once_covers_that_call_on_every_adapter(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    plan: list[Call] = [("pay", {"amount": 5}), ("pay", {"amount": 6})]
    target, tools = await BUILDERS[framework](harness, [pay], tmp_path, plan)
    agent = arm(harness.wrap(target, id=f"payer-{framework}"), tools)
    first = await agent.run("pay twice", user="ada")
    asked = first.interrupt
    assert asked is not None and asked.reason is InterruptReason.APPROVAL, first
    assert done == []  # asked before it ran
    assert asked.question == f"Approve pay? pay is write. {THREE}"
    assert list(asked.options) == [ONCE, CONVERSATION, DENY]
    second = await approved(agent, first)
    assert done == ["pay 5"] and second.status is RunStatus.PAUSED, second  # once: one call
    finished = await approved(agent, second)
    assert finished.status is RunStatus.SUCCESS and done == ["pay 5", "pay 6"], finished


async def test_deny_never_runs_and_a_picked_option_is_the_decision_it_names(
    harness: Harness,
) -> None:
    agent = arm(harness.wrap(calling(("pay", 1), ("pay", 2)), id="picks"), [pay])
    first = await agent.run("go", user="ada", thread=THREAD)
    assert first.interrupt is not None
    denied = await agent.resume(
        first.interrupt.interrupt_id, "answer", answer=DENY.value, reviewer="ada"
    )
    assert denied.interrupt is not None and done == []  # denied: never ran; the next asks
    finished = await agent.resume(
        denied.interrupt.interrupt_id, "answer", answer=CONVERSATION.value, reviewer="ada"
    )
    assert finished.status is RunStatus.SUCCESS and done == ["pay 2"], finished
    assert finished.answer[0] == "pay was not run: the approver rejected it"
    again = await agent.run("go", user="ada", thread=THREAD)  # the pick reached the conversation
    assert again.status is RunStatus.SUCCESS and done == ["pay 2", "pay 1", "pay 2"], again


# --------------------------------------------------------------------------- this conversation


async def test_allow_for_this_conversation_covers_that_tool_in_later_runs_of_that_thread_only(
    harness: Harness,
) -> None:
    payer = arm(harness.wrap(calling(("pay", 1)), id="payer"), [pay, refund])
    allowed = await approved(payer, await payer.run("go", user="ada", thread=THREAD), "run")
    assert allowed.status is RunStatus.SUCCESS and done == ["pay 1"]
    later = await payer.run("go", user="ada", thread=THREAD)
    assert later.status is RunStatus.SUCCESS and done == ["pay 1", "pay 1"], later
    assert (await payer.run("go", user="ada", thread="chat-2")).status is RunStatus.PAUSED
    assert (await payer.run("go", user="bob", thread=THREAD)).status is RunStatus.PAUSED
    refunder = arm(harness.wrap(calling(("refund", 1)), id="refunder"), [pay, refund])
    assert (await refunder.run("go", user="ada", thread=THREAD)).status is RunStatus.PAUSED
    assert done == ["pay 1", "pay 1"]  # another tool, thread or user: asked, not run


async def test_a_resumed_run_another_worker_and_a_child_run_see_the_same_approval() -> None:
    store = LocalRuns()
    async with Harness(config=Settings(), runs=store) as one:
        payer = arm(one.wrap(calling(("pay", 1)), id="payer"), [pay, refund])
        await approved(payer, await payer.run("go", user="ada", thread=THREAD), "run")
        both = arm(one.wrap(calling(("refund", 2), ("pay", 3)), id="both"), [pay, refund])
        paused = await both.run("go", user="ada", thread=THREAD)  # refund asks; pay then not
        resumed = await approved(both, paused)
        assert resumed.status is RunStatus.SUCCESS, resumed
        assert done == ["pay 1", "refund 2", "pay 3"]
    async with Harness(config=Settings(), runs=store) as other:  # another worker, same store
        payer = arm(other.wrap(calling(("pay", 4)), id="payer"), [pay])
        assert (await payer.run("go", user="ada", thread=THREAD)).status is RunStatus.SUCCESS
        child = arm(other.wrap(calling(("pay", 5)), id="child"), [pay])

        async def parent(input: str, agent: Runtime) -> Any:
            return (await child.run("pay", user="ada")).answer

        parenting = other.wrap(parent, id="parent")
        result = await parenting.run("go", user="ada", thread=THREAD)
        assert result.status is RunStatus.SUCCESS, result
        assert done[-2:] == ["pay 4", "pay 5"]


async def test_what_a_conversation_allowed_is_read_page_by_page(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conversation, "RESOLUTIONS_PAGE", 1)
    both = arm(harness.wrap(calling(("pay", 1), ("refund", 2)), id="both"), [pay, refund])
    first = await both.run("go", user="ada", thread=THREAD)
    await approved(both, await approved(both, first, "run"), "run")  # two answers, two pages
    again = await both.run("go", user="ada", thread=THREAD)
    assert again.status is RunStatus.SUCCESS and len(done) == 4, again


# --------------------------------------------------------------------------- always


def kept(memory: FakeMemoryService) -> list[tuple[str, str, str | None]]:
    """The permissions the memory service keeps: user, tool, assignee."""
    return [(p["user"], p["tool"], p["assignee"]) for p in memory.permissions.values()]


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_always_allow_is_offered_with_a_memory_service_and_covers_the_users_conversations(
    memory_harness: Harness, memory_service: FakeMemoryService, framework: str, tmp_path: Path
) -> None:
    plan: list[Call] = [("pay", {"amount": 5})]
    target, tools = await BUILDERS[framework](memory_harness, [pay], tmp_path, plan)
    agent = arm(memory_harness.wrap(target, id=f"payer-{framework}"), tools)
    first = await agent.run("pay", user="ada", thread=THREAD)
    asked = first.interrupt
    assert asked is not None and asked.question == f"Approve pay? pay is write. {FOUR}", first
    assert list(asked.options) == [ONCE, CONVERSATION, ALWAYS, DENY]
    allowed = await agent.resume(asked.interrupt_id, "answer", answer=ALWAYS.value, reviewer="ada")
    assert allowed.status is RunStatus.SUCCESS and done == ["pay 5"], allowed
    await memory_harness.writes.drain()
    assert kept(memory_service) == [("ada", "pay", None)]
    elsewhere = await agent.run("pay", user="ada", thread="chat-2")  # another conversation
    assert elsewhere.status is RunStatus.SUCCESS and done == ["pay 5", "pay 5"], elsewhere
    bobs = await agent.run("pay", user="bob", thread="chat-3")  # another user still asks
    assert bobs.status is RunStatus.PAUSED and done == ["pay 5", "pay 5"], bobs


async def test_always_is_kept_once_and_read_once_a_run_however_often_the_run_resumes(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    agent = arm(
        memory_harness.wrap(calling(("pay", 1), ("refund", 2), ("pay", 3)), id="both"),
        [pay, refund],
    )
    first = await agent.run("go", user="ada", thread=THREAD)
    assert first.interrupt is not None
    second = await agent.resume(
        first.interrupt.interrupt_id, "answer", answer=ALWAYS.value, reviewer="ada"
    )  # refund asks next: the run resumes again, its pay approval replayed
    assert second.interrupt is not None and second.interrupt.tool_call is not None
    assert second.interrupt.tool_call.tool == "refund"
    finished = await approved(agent, second)
    assert finished.status is RunStatus.SUCCESS and done == ["pay 1", "refund 2", "pay 3"]
    await memory_harness.writes.drain()
    assert len(memory_service.named("allow_tool")) == 1 and kept(memory_service) == [
        ("ada", "pay", None)
    ]
    assert memory_service.named("allow_tool")[0].body["allowed_by"] == "ada"
    reads = len(memory_service.named("permissions"))
    many = arm(
        memory_harness.wrap(calling(("pay", 4), ("pay", 5), ("refund", 6)), id="many"),
        [pay, refund],
    )
    paused = await many.run("go", user="ada", thread="chat-2")  # pays unasked; refund asks
    assert paused.interrupt is not None and paused.interrupt.tool_call is not None
    assert paused.interrupt.tool_call.tool == "refund"
    assert len(memory_service.named("permissions")) == reads + 1  # one read for the attempt


async def test_a_revoked_permission_asks_again_from_the_next_run(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    agent = arm(memory_harness.wrap(calling(("pay", 1)), id="payer"), [pay])
    first = await agent.run("go", user="ada", thread=THREAD)
    assert first.interrupt is not None
    await agent.resume(first.interrupt.interrupt_id, "answer", answer=ALWAYS.value, reviewer="ada")
    await memory_harness.writes.drain()
    assert (await agent.run("go", user="ada", thread="chat-2")).status is RunStatus.SUCCESS
    memory_service.permissions.clear()  # revoked (DELETE /v1/tool-permissions/{id})
    again = await agent.run("go", user="ada", thread="chat-3")
    assert again.status is RunStatus.PAUSED and done == ["pay 1", "pay 1"], again


async def test_an_irreversible_tool_never_offers_or_keeps_always(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    agent = arm(memory_harness.wrap(calling(("wire", 1)), id="wirer"), [wire])
    first = await agent.run("go", user="ada", thread=THREAD)
    asked = first.interrupt
    assert asked is not None and list(asked.options) == [ONCE, DENY], first
    with pytest.raises(ConfigurationError):  # not one of its options
        await agent.resume(asked.interrupt_id, "answer", answer=ALWAYS.value, reviewer="ada")
    ran = await approved(agent, first, "always")  # said anyway: approved this call only
    assert ran.status is RunStatus.SUCCESS and done == ["wire 1"], ran
    await memory_harness.writes.drain()
    assert memory_service.named("allow_tool") == [] and kept(memory_service) == []
    memory_service.permissions["tpm_x"] = {
        "permission_id": "tpm_x",
        "tenant": "acme",
        "user": "ada",
        "tool": "wire",
        "assignee": None,
        "allowed_by": "ada",
        "created_at": "2026-10-01T00:00:00Z",
    }  # kept by some other way: still never honoured for an irreversible tool
    assert (await agent.run("go", user="ada", thread="chat-2")).status is RunStatus.PAUSED


async def test_a_rule_decides_before_always(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """An ``approve_when`` (an administrator's, or an accepted suggestion) governs the tool:
    Always allow is not offered, and a permission kept before does not skip the rule."""
    memory_service.catalog["pay"] = {"side_effects": "write", "approve_when": "amount > 10"}
    memory_service.permissions["tpm_x"] = {
        "permission_id": "tpm_x",
        "tenant": "acme",
        "user": "ada",
        "tool": "pay",
        "assignee": None,
        "allowed_by": "ada",
        "created_at": "2026-10-01T00:00:00Z",
    }
    agent = arm(memory_harness.wrap(calling(("pay", 50)), id="payer"), [pay])
    first = await agent.run("go", user="ada", thread=THREAD)
    asked = first.interrupt
    assert asked is not None and ALWAYS not in asked.options, first
    assert done == []


async def test_a_hooks_ask_decides_before_always(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """A ``before_tool`` hook's ``Ask`` is your rule in code: it asks whatever the user allowed
    always, and never offers Always allow."""
    agent = arm(memory_harness.wrap(calling(("pay", 1)), id="payer"), [pay])
    first = await agent.run("go", user="ada", thread=THREAD)
    assert first.interrupt is not None
    await agent.resume(first.interrupt.interrupt_id, "answer", answer=ALWAYS.value, reviewer="ada")
    await memory_harness.writes.drain()
    big = arm(memory_harness.wrap(calling(("pay", 500)), id="big", hooks=[OverHundred()]), [pay])
    asked = await big.run("go", user="ada", thread="chat-2")
    assert asked.interrupt is not None and "Over 100: pay it?" in asked.interrupt.question
    assert ALWAYS not in asked.interrupt.options and done == ["pay 1"], asked


async def test_a_hooks_ask_beats_what_was_allowed_for_the_conversation(harness: Harness) -> None:
    """As a PreToolUse "ask" in Claude Code: the hook's question is asked though the tool was
    allowed for this conversation — in the same run and in a later one."""
    agent = arm(
        harness.wrap(calling(("pay", 1), ("pay", 500)), id="payer", hooks=[OverHundred()]), [pay]
    )
    first = await agent.run("go", user="ada", thread=THREAD)
    hooked = await approved(agent, first, "run")  # pay allowed for the conversation
    assert hooked.interrupt is not None and "Over 100" in hooked.interrupt.question, hooked
    assert done == ["pay 1"]
    finished = await approved(agent, hooked)
    assert finished.status is RunStatus.SUCCESS and done == ["pay 1", "pay 500"], finished
    later = await agent.run("go", user="ada", thread=THREAD)  # pay 1 covered; pay 500 asks
    assert later.interrupt is not None and "Over 100" in later.interrupt.question, later
    assert done == ["pay 1", "pay 500", "pay 1"]


async def test_a_hooks_ask_beats_what_a_schedule_allowed() -> None:
    async with Harness(config=Settings()) as h:
        chat = Chat(("pay", 1), ("pay", 500))
        agent = arm(h.wrap(chat, id="desk", hooks=[OverHundred()]), [pay])
        asked = await agent.run("pay my rent monthly", user="ada", thread=THREAD)
        assert asked.interrupt is not None, asked
        await agent.resume(asked.interrupt.interrupt_id, "answer", answer=ALLOW, reviewer="ada")
        run = await fired(h, agent)
        assert run.status is RunStatus.PAUSED and done == ["pay 1"], run  # allowed, then hooked
        assert "Over 100" in run.awaiting.question


async def test_always_is_not_offered_when_the_permissions_cannot_be_read_or_kept(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """A grant is never offered where it would be lost: the permissions could not be read in
    this attempt (the service away, a refusal), or the key cannot grant (an issuer's token)."""
    agent = arm(memory_harness.wrap(calling(("pay", 1)), id="payer"), [pay])
    memory_service.fail.add("permissions")
    unread = await agent.run("go", user="ada", thread=THREAD)
    assert unread.interrupt is not None and list(unread.interrupt.options) == [
        ONCE,
        CONVERSATION,
        DENY,
    ], unread
    assert THREE in unread.interrupt.question
    memory_service.fail.clear()
    memory_service.failures["permissions"] = 403
    refused = await agent.run("go", user="ada", thread="chat-2")
    assert refused.interrupt is not None and ALWAYS not in refused.interrupt.options


async def test_a_key_that_cannot_grant_is_offered_three_choices_and_still_honours_permissions(
    memory_service: FakeMemoryService,
) -> None:
    memory_service.role = "jwt"  # an issuer's token: the service refuses its grants
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        agent = arm(h.wrap(calling(("pay", 1)), id="payer"), [pay])
        asked = await agent.run("go", user="ada", thread=THREAD)
        assert asked.interrupt is not None and ALWAYS not in asked.interrupt.options, asked
        memory_service.permissions["tpm_x"] = {
            "permission_id": "tpm_x",
            "tenant": "acme",
            "user": "ada",
            "tool": "pay",
            "assignee": None,
            "allowed_by": "ada",
            "created_at": "2026-10-01T00:00:00Z",
        }  # granted before, by the service key
        kept = await agent.run("go", user="ada", thread="chat-2")
        assert kept.status is RunStatus.SUCCESS and done == ["pay 1"], kept


async def test_a_grant_whose_write_fails_is_kept_in_the_spool(
    memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """The permissions were read (Always allow offered), the write fails: spooled, not lost."""
    memory_service.fail.add("allow_tool")
    settings = Settings(spool_dir=str(tmp_path))
    async with Harness(config=settings, memory=memory_service.client()) as h:
        agent = arm(h.wrap(calling(("pay", 1)), id="payer"), [pay])
        first = await agent.run("go", user="ada", thread=THREAD)
        assert first.interrupt is not None and ALWAYS in first.interrupt.options
        await agent.resume(
            first.interrupt.interrupt_id, "answer", answer=ALWAYS.value, reviewer="ada"
        )
        await h.writes.drain()
        assert h.writes.spooled == 1 and h.writes.failed == 0
    spooled = (tmp_path / "trellis-writes.jsonl").read_text()
    assert '"op": "allow_always"' in spooled and '"tool": "pay"' in spooled


async def test_a_permission_store_that_cannot_be_read_asks_and_warns_once(
    memory_harness: Harness,
    memory_service: FakeMemoryService,
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent = arm(memory_harness.wrap(calling(("pay", 1)), id="payer"), [pay])
    first = await agent.run("go", user="ada", thread=THREAD)
    assert first.interrupt is not None
    await agent.resume(first.interrupt.interrupt_id, "answer", answer=ALWAYS.value, reviewer="ada")
    await memory_harness.writes.drain()
    memory_service.fail.add("permissions")
    for thread in ("chat-2", "chat-3"):
        asked = await agent.run("go", user="ada", thread=thread)  # fails closed: it asks
        assert asked.status is RunStatus.PAUSED and done == ["pay 1"], asked
    unread = [r for r in caplog.records if "tool permissions" in r.getMessage()]
    assert len(unread) == 1 and unread[0].levelname == "WARNING", unread


# --------------------------------------------------------------------------- what does not ask


async def test_a_frameworks_own_write_is_announced_as_before_not_asked(harness: Harness) -> None:
    """Claude Code's and Kimi's built-ins go through ``bridge.permitted`` under their own
    permission flow: a ``write`` there is announced, as it was."""
    seen: list[Any] = []

    async def fn(input: str, agent: Runtime) -> Any:
        seen.append(await bridge.permitted(ToolSpec(name="Write", side_effects="write"), {}))
        return "ok"

    result = await harness.wrap(fn, id="claude-like").run("go", user="ada")
    assert result.status is RunStatus.SUCCESS and seen[0].tool == "Write", result


# --------------------------------------------------------------------------- schedules


class Chat:
    """In chat: schedules its task naming the write it will use; a fired run calls ``calls``."""

    def __init__(self, *calls: tuple[str, int]) -> None:
        self.calls = calls

    async def __call__(self, input: str, agent: Runtime) -> Any:
        if input.startswith("Scheduled task:"):
            return [text_of(await agent.tools.call(n, amount=a)) for n, a in self.calls]
        args = {"task": "pay the rent", "when": "0 8 1 * *", "tools": ["pay", "wire", "nope"]}
        return text_of(await agent.tools.call("schedule_task", **args))


async def fired(h: Harness, agent: Any) -> Any:
    [schedule] = h.runs._schedules.values()  # type: ignore[union-attr]
    run_id = (await h.runs.schedules.fire(schedule.schedule_id)).run_id
    assert await h.worker([agent]).run_once() is True
    return await h.runs.get(run_id)


@pytest.mark.parametrize("picked", [ALLOW, EACH_TIME])
async def test_a_schedule_allows_the_writes_its_task_names_when_it_is_confirmed(
    picked: str,
) -> None:
    async with Harness(config=Settings()) as h:
        agent = arm(h.wrap(Chat(("pay", 1), ("refund", 2)), id="desk"), [pay, refund, wire])
        asked = await agent.run("pay my rent monthly", user="ada", thread=THREAD)
        interrupt = asked.interrupt
        assert interrupt is not None, asked
        # only the write it names that asks: not an irreversible one, not an unknown name
        assert "pay" in interrupt.question and "wire" not in interrupt.question
        assert interrupt.option_values == [ALLOW, EACH_TIME, NEITHER]
        made = await agent.resume(interrupt.interrupt_id, "answer", answer=picked, reviewer="ada")
        assert made.status is RunStatus.SUCCESS, made
        [schedule] = h.runs._schedules.values()  # type: ignore[attr-defined]
        assert schedule.metadata.get(ALLOWED, []) == (["pay"] if picked == ALLOW else [])
        run = await fired(h, agent)
        # an uncovered write pauses, in the chat's thread, for its user: never run silently
        assert run.status is RunStatus.PAUSED and run.thread_id == THREAD, run
        assert run.awaiting.tool_call.tool == ("refund" if picked == ALLOW else "pay")
        assert done == (["pay 1"] if picked == ALLOW else [])


async def test_cancelling_the_confirmation_makes_no_schedule() -> None:
    async with Harness(config=Settings()) as h:
        agent = arm(h.wrap(Chat(), id="desk"), [pay])
        asked = await agent.run("pay my rent monthly", user="ada", thread=THREAD)
        assert asked.interrupt is not None
        await agent.resume(asked.interrupt.interrupt_id, "answer", answer=NEITHER, reviewer="ada")
        assert not h.runs._schedules  # type: ignore[attr-defined]
