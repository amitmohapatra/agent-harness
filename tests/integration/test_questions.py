"""W5 questions on every adapter, ``ReAct`` and Way 2: labelled options and several picks, a
form from a pydantic model with widget hints, the asker's own screen; the answer checked by the
one check agent-runs makes; whoever waits is told (notifiers); a sub-agent's question carried
to its parent whole."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import trellis
from tests.support.adapters import BUILDERS
from tests.support.planned import Call
from trellis import Harness, Runtime, Settings, tool
from trellis.contracts import (
    ConfigurationError,
    Interrupt,
    InterruptReason,
    Option,
    RunEventType,
    RunStart,
    RunStatus,
)
from trellis.harness import pipeline
from trellis.harness.asking import Question
from trellis.harness.events import NOTIFIED, WARNING
from trellis.harness.runs import LocalRuns
from trellis.testing import Reviewer

PLANS = [Option(value="a", label="Plan A", description="the small one"), "b", "c"]


@tool(side_effects="read")
async def pick_plans(customer: str) -> str:
    """Ask the customer's account manager which plans to offer."""
    runtime = trellis.current()
    assert runtime is not None
    picked = await runtime.ask(
        f"Which plans for {customer}?",
        options=PLANS,
        multiple=True,
        component="plan-picker",
        props={"customer": customer},
        assignee="role:sales",
    )
    return ",".join(picked)


class Address(BaseModel):
    street: str
    zip: int


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_a_choice_of_labelled_options_with_several_picks_on_every_adapter(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    plan: list[Call] = [("pick_plans", {"customer": "acme"})]
    target, tools = await BUILDERS[framework](harness, [pick_plans], tmp_path, plan)
    agent = harness.wrap(target, id=f"plans-{framework}", tools=tools)
    paused = await agent.run("offer plans to acme", user="ada")
    asked = paused.interrupt
    assert asked is not None and asked.reason is InterruptReason.CHOICE, paused
    assert asked.options == PLANS and asked.multiple and asked.ui == "choice"
    assert (asked.component, asked.props, asked.assignee) == (
        "plan-picker",
        {"customer": "acme"},
        "role:sales",
    )
    with pytest.raises(ConfigurationError, match="not a list of the options"):
        await agent.resume(asked.interrupt_id, "answer", answer="a", reviewer="lee")
    with pytest.raises(ConfigurationError, match="not among the options"):
        await agent.resume(asked.interrupt_id, "answer", answer=["a", "z"], reviewer="lee")
    done = await Reviewer({"plan-picker": ["a", "c"]}).settle(agent, paused)
    assert done.status is RunStatus.SUCCESS and done.answer == "Done. a,c", done


async def test_a_form_from_a_model_with_widget_hints_reads_the_answer_back(
    harness: Harness,
) -> None:
    async def fn(input: str, agent: Runtime) -> Any:
        address = await agent.ask(
            "Where to?", form=Address, ui_schema={"street": {"ui:widget": "textarea"}}
        )
        assert isinstance(address, Address)
        return f"{address.street} {address.zip}"

    agent = harness.wrap(fn, id="shipping")
    paused = await agent.run("ship it", user="ada")
    asked = paused.interrupt
    assert asked is not None and asked.ui == "form"
    assert asked.expects == Address.model_json_schema()
    assert asked.ui_schema == {"street": {"ui:widget": "textarea"}}
    with pytest.raises(ConfigurationError, match="does not fit"):
        await agent.resume(asked.interrupt_id, "answer", answer={"street": "x"}, reviewer="a")
    done = await agent.resume(
        asked.interrupt_id, "answer", answer={"street": "Main 1", "zip": 10115}, reviewer="a"
    )
    assert done.answer == "Main 1 10115"


async def test_a_question_that_cannot_be_asked_is_refused_where_it_is_asked(
    harness: Harness,
) -> None:
    async def fn(input: str, agent: Runtime) -> Any:
        kinds: dict[str, dict[str, Any]] = {
            "both": {"form": Address, "expects": {"type": "string"}},
            "props": {"props": {"a": 1}},
            "twice": {"options": ["a", Option(value="a")]},
            "multiple": {"multiple": True},
            "schema": {"expects": {"type": "nothing"}},
        }
        return await agent.ask("Which?", **kinds[input])

    agent = harness.wrap(fn, id="refused")
    for kind, why in (
        ("both", "give form= .* or expects=, not both"),
        ("props", "props are a component's"),
        ("twice", "distinct values"),
        ("multiple", "multiple needs options"),
        ("schema", "not a valid JSON Schema"),
    ):
        failed = await agent.run(kind, user="u")
        assert failed.error is not None and failed.status is RunStatus.ERROR
        assert failed.error.code == ConfigurationError.code
        assert re.search(why, failed.error.message), (kind, failed.error.message)


async def test_way_2_asks_the_same_question_through_its_own_run_store() -> None:
    runs = LocalRuns()
    run = await runs.start(RunStart(tenant_id="acme", agent_id="mine", input="x"))
    question = Question("Where to?", form=Address)
    interrupt = question.interrupt(tenant="acme", run_id=run.run_id)
    await runs.pause(interrupt, checkpoint={"step": 1})
    reviewer = Reviewer({"Where to?": {"street": "Main 1", "zip": 1}})
    resumed = await runs.resume(reviewer.resolution(interrupt), tenant="acme")
    assert resumed.last_resolution is not None
    assert question.answer(resumed.last_resolution) == Address(street="Main 1", zip=1)
    with pytest.raises(ConfigurationError, match="is not a Address"):
        question.answer(reviewer.resolution(interrupt).model_copy(update={"answer": {"zip": 1}}))


# --------------------------------------------------------------------------- notifiers


class Recording:
    name = "recording"

    def __init__(self) -> None:
        self.told: list[tuple[Interrupt, str | None]] = []

    async def notify(self, interrupt: Interrupt, link: str | None) -> None:
        self.told.append((interrupt, link))


class Broken:
    async def notify(self, interrupt: Interrupt, link: str | None) -> None:
        raise ConnectionError("the pager is down")


async def test_whoever_waits_is_told_redacted_with_a_link_and_a_broken_notifier_is_a_warning() -> (
    None
):
    told = Recording()
    settings = Settings(inbox_url="https://ops.example/inbox")
    async with Harness(config=settings, notifiers=[told, Broken()]) as h:

        async def fn(input: str, agent: Runtime) -> Any:
            return await agent.tools.call(
                "wire", amount=5, password="hunter2", to="bob@example.com"
            )

        @tool(side_effects="irreversible")
        def wire(amount: int, password: str, to: str) -> str:
            """Wire."""
            return "wired"

        agent = h.wrap(fn, id="paging", tools=[wire])
        events = [e async for e in agent.stream("pay bob@example.com", user="ada")]
        await h.writes.drain()
        [(interrupt, link)] = told.told
        assert interrupt.tool_call is not None
        assert interrupt.tool_call.args["password"] == "[redacted]"
        assert interrupt.tool_call.args["to"] == "[email]"
        assert link == f"https://ops.example/inbox#{interrupt.interrupt_id}"
        assert events[-1].outcome is not None and events[-1].outcome.value == "interrupt"
    customs = [e.data for e in events if e.type is RunEventType.CUSTOM]
    # the notifiers report on the run's events after its RUN_FINISHED: none were streamed
    assert not [c for c in customs if c["name"] in (NOTIFIED, WARNING)]


async def test_a_notifier_failure_is_a_warning_on_the_run_and_a_success_an_event() -> None:
    told = Recording()
    async with Harness(config=Settings(), notifiers=[told, Broken()]) as h:

        async def fn(input: str, agent: Runtime) -> Any:
            return await agent.ask("Go?")

        agent = h.wrap(fn, id="told")
        seen: list[Any] = []
        record = await agent._opened("go", user="ada", thread=None, tenant=None)
        result = await pipeline.attempt(agent, record, "go", listener=seen.append)
        await h.writes.drain()
    assert result.status is RunStatus.PAUSED
    names = [e.data.get("name") for e in seen if e.type is RunEventType.CUSTOM]
    assert NOTIFIED in names and WARNING in names  # after the pause, on its events
    warning = next(e.data for e in seen if e.data.get("code") == "notify_failed")
    assert "Broken was not told" in warning["message"]
    assert "the pager is down" in warning["message"]


async def test_a_sub_agents_question_is_carried_whole_and_told_once() -> None:
    told = Recording()
    async with Harness(config=Settings(), notifiers=[told]) as h:

        async def child(input: str, agent: Runtime) -> Any:
            return await agent.ask(
                "Which plan?",
                options=PLANS,
                multiple=True,
                component="plan-picker",
                props={"for": "child"},
                ui_schema={"x": 1},
            )

        helper = h.wrap(child, id="helper")

        async def parent(input: str, agent: Runtime) -> Any:
            return await agent.tools.call("helper", message="pick")

        boss = h.wrap(parent, id="boss", tools=[helper.as_tool()])
        paused = await boss.run("go", user="ada")
        await h.writes.drain()
        asked = paused.interrupt
        assert asked is not None
        assert (asked.multiple, asked.component, asked.props, asked.ui_schema) == (
            True,
            "plan-picker",
            {"for": "child"},
            {"x": 1},
        )
        assert [i.run_id for i, _ in told.told] == [paused.run_id]  # the parent's, once
        done = await boss.resume(asked.interrupt_id, "answer", answer=["b"], reviewer="lee")
        assert done.answer == ["b"]
