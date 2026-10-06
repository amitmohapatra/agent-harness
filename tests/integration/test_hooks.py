"""Hooks around runs, model calls and tool calls: the tool hooks in the bridge on every adapter
(deny, rewrite, ask — journaled — and the outcome changed), the run hooks in the pipeline, the
model hooks through each framework's own mechanism (``ReAct``'s loop, LangChain middleware, the
OpenAI Agents SDK's ``RunHooks``), and the same hooks in Way 2 (``governed``, the middleware
and ``RunHooks`` outside a harness run)."""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import Any

import pytest
from agents import Agent as OpenAIAgent
from agents import Runner
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage

from tests.support.adapters import BUILDERS
from tests.support.planned import Call, PlannedChat, PlannedChatModel, PlannedModel
from trellis import Ask, Deny, Harness, Hooks, ModelCall, ReAct, Rewrite, Runtime, Settings, tool
from trellis.contracts import ModelError, RunEventType, RunStatus, ToolCall, ToolOutcome
from trellis.harness.governance import Decision, Denied, Governance, governed
from trellis.harness.hooks.openai_agents import ModelHooks as RunHooks
from trellis.harness.middleware import ModelHooks as Middleware
from trellis.harness.result import Result

paid: list[int] = []
quoted: list[str] = []


@tool(side_effects="read")
def quote(sku: str) -> str:
    """The price of a SKU."""
    quoted.append(sku)
    return f"{sku} costs 7"


@tool(side_effects="write")
def pay(amount: int) -> str:
    """Pay an amount."""
    paid.append(amount)
    return f"paid {amount}"


@pytest.fixture(autouse=True)
def _reset() -> None:
    paid.clear()
    quoted.clear()


class Guard(Hooks):
    """No secrets quoted, nothing paid over 100, every outcome checked."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def before_tool(self, call: ToolCall) -> Deny | Ask | Rewrite | None:
        self.asked.append(call.tool)
        if call.args.get("sku") == "secret":
            return Deny("secret SKUs are not quoted")
        if call.tool == "pay" and call.args["amount"] > 100:
            return Rewrite({"amount": 100})
        return None

    async def after_tool(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        return outcome.model_copy(update={"output": f"{outcome.output} (checked)"})


def results(events: list[Any]) -> dict[str, tuple[str, Any]]:
    return {
        e.data["tool"]: (e.data["status"], e.data["output"])
        for e in events
        if e.type is RunEventType.TOOL_CALL_RESULT
    }


# --------------------------------------------------------------------------- tool hooks


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_a_hook_denies_rewrites_and_checks_calls_on_every_adapter(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    plan: list[Call] = [("quote", {"sku": "secret"}), ("pay", {"amount": 500})]
    target, tools = await BUILDERS[framework](harness, [quote, pay], tmp_path, plan)
    agent = harness.wrap(target, id=f"guarded-{framework}", tools=tools, hooks=[Guard()])
    events = [e async for e in agent.stream("pay for the secret", user="u")]
    finished = events[-1]
    assert finished.outcome is not None and finished.outcome.value == "success", finished
    assert finished.data["result"] == "Done. paid 100 (checked)"  # what the model read
    assert quoted == [] and paid == [100]
    seen = results(events)
    assert seen["quote"] == ("rejected", "quote was not run: secret SKUs are not quoted")
    assert seen["pay"] == ("ok", "paid 100 (checked)")


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_a_hook_asks_a_person_and_its_decision_is_journaled_on_every_adapter(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    class Asking(Hooks):
        def __init__(self) -> None:
            self.asked = 0

        async def before_tool(self, call: ToolCall) -> Ask | None:
            self.asked += 1
            return Ask("Quote A for this customer?", assignee="role:sales")

    hooks = Asking()
    target, tools = await BUILDERS[framework](harness, [quote], tmp_path, [("quote", {"sku": "A"})])
    agent = harness.wrap(target, id=f"asking-{framework}", tools=tools, hooks=[hooks])
    paused = await agent.run("quote A", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused
    assert paused.interrupt.question == "Approve quote? Quote A for this customer?"
    assert paused.interrupt.assignee == "role:sales" and quoted == []
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="sales")
    assert done.status is RunStatus.SUCCESS and done.answer == "Done. A costs 7"
    assert quoted == ["A"] and hooks.asked == 1  # the resumed run read the journal


async def test_hooks_run_in_order_the_harness_then_the_agent() -> None:
    order: list[str] = []

    class Named(Hooks):
        def __init__(self, name: str) -> None:
            self.name = name

        async def before_tool(self, call: ToolCall) -> Rewrite:
            order.append(self.name)
            return Rewrite({"sku": f"{call.args['sku']}-{self.name}"})

    async def fn(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("quote", sku="A")

    async with Harness(config=Settings(), hooks=[Named("harness")]) as h:
        result = await h.wrap(fn, id="ordered", tools=[quote], hooks=[Named("agent")]).run(
            "q", user="u"
        )
    assert order == ["harness", "agent"] and result.answer == "A-harness-agent costs 7"


# --------------------------------------------------------------------------- run hooks


async def test_run_hooks_see_each_attempt_and_every_failure(harness: Harness) -> None:
    seen: list[Any] = []

    class Watching(Hooks):
        async def on_run_start(self, run: Runtime) -> None:
            seen.append(("start", run.attempt, run.task))

        async def on_run_end(self, run: Runtime, result: Result) -> None:
            seen.append(("end", result.status.value))
            raise RuntimeError("a broken hook")  # logged: the run's outcome stands

        async def on_error(self, stage: str, error: Exception) -> None:
            seen.append(("error", stage, str(error)))
            raise RuntimeError("another broken hook")

    @tool(side_effects="read")
    def broken(sku: str) -> str:
        """Always fails."""
        raise ValueError("no such SKU")

    async def fn(input: str, agent: Runtime) -> Any:
        if input == "fail":
            raise RuntimeError("the agent broke")
        failed = await agent.tools.call("broken", sku="A")
        return f"{failed} / {await agent.ask('Go on?')}"

    agent = harness.wrap(fn, id="watched", tools=[broken], hooks=[Watching()])
    paused = await agent.run("go", user="u")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="u")
    assert done.answer == "broken failed: no such SKU / yes"
    failed = await agent.run("fail", user="u")

    async def slow(input: str, agent: Runtime) -> str:
        await asyncio.sleep(1)
        return "late"

    timed_out = await harness.wrap(slow, id="slow", hooks=[Watching()], timeout=0.01).run(
        "slow", user="u"
    )
    assert failed.status is RunStatus.ERROR and timed_out.status is RunStatus.TIMEOUT
    assert seen == [
        ("start", 1, "go"),
        ("error", "tool", "no such SKU"),
        ("end", "PAUSED"),
        ("start", 2, "go"),
        ("error", "tool", "no such SKU"),  # a failed call runs again in the next attempt
        ("end", "SUCCESS"),
        ("start", 1, "fail"),
        ("error", "run", "the agent broke"),
        ("end", "ERROR"),
        ("start", 1, "slow"),
        ("error", "run", "the run worked past its time limit of 0.01s"),
        ("end", "TIMEOUT"),
    ]


async def test_a_hook_that_raises_before_a_run_starts_fails_it(harness: Harness) -> None:
    class Refusing(Hooks):
        async def on_run_start(self, run: Runtime) -> None:
            raise PermissionError("not today")

    async def fn(input: str, agent: Runtime) -> str:
        return "ran"

    result = await harness.wrap(fn, id="refused", hooks=[Refusing()]).run("q", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "not today" in result.error.message


# --------------------------------------------------------------------------- model hooks


class Redacting(Hooks):
    """Every model call: the user's card number out of what is sent; the replies kept."""

    def __init__(self) -> None:
        self.calls: list[ModelCall] = []
        self.replies: list[Any] = []
        self.errors: list[tuple[str, str]] = []

    async def before_model(self, call: ModelCall) -> ModelCall:
        self.calls.append(call)
        return dataclasses.replace(call, messages=[_redacted(m) for m in call.messages])

    async def after_model(self, call: ModelCall, reply: Any) -> None:
        self.replies.append(reply)

    async def on_error(self, stage: str, error: Exception) -> None:
        self.errors.append((stage, type(error).__name__))


def _redacted(message: Any) -> Any:
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return {**message, "content": message["content"].replace("4111", "****")}
    if isinstance(message, BaseMessage) and isinstance(message.content, str):
        return message.model_copy(update={"content": message.content.replace("4111", "****")})
    return message


async def test_react_model_calls_go_through_the_hooks(harness: Harness) -> None:
    hooks = Redacting()
    model = PlannedChat([("quote", {"sku": "A"})])
    agent = harness.wrap(ReAct(system="s", model=model), id="r", tools=[quote], hooks=[hooks])
    result = await agent.run("card 4111: quote A", user="u")
    assert result.answer == "Done. A costs 7"
    assert [c.framework for c in hooks.calls] == ["react", "react"]
    assert hooks.calls[0].model == "PlannedChat" and len(hooks.replies) == 2
    sent = model.said()
    assert "****" in sent and "4111" not in sent  # the model never read the card

    class Failing:
        async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
            raise ConnectionError("the model is down")

    failing = harness.wrap(ReAct(system="s", model=Failing()), id="down", hooks=[hooks])
    assert (await failing.run("q", user="u")).status is RunStatus.ERROR
    assert hooks.errors[-2:] == [("model", "ConnectionError"), ("run", "ConnectionError")]


async def test_a_react_model_call_out_of_time_is_a_model_error_the_hooks_see(
    harness: Harness,
) -> None:
    class Slow:
        async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
            await asyncio.sleep(5)
            return {}

    hooks = Redacting()
    target = ReAct(system="s", model=Slow(), model_timeout=0.01)
    result = await harness.wrap(target, id="slow", hooks=[hooks]).run("q", user="u")
    assert result.status is RunStatus.ERROR
    assert hooks.errors[0] == ("model", ModelError.__name__)


@pytest.mark.parametrize("deep", [False, True], ids=["create_agent", "deepagents"])
async def test_langchain_model_calls_go_through_the_middleware(
    harness: Harness, deep: bool
) -> None:
    hooks = Redacting()
    model = PlannedChatModel(plan=[("quote", {"sku": "A"})])
    tools = await harness.tools(quote, framework="deepagents" if deep else "langgraph")
    graph = (
        create_deep_agent(model=model, tools=tools, middleware=[Middleware()])
        if deep
        else create_agent(model, tools=tools, middleware=[Middleware()])
    )
    agent = harness.wrap(graph, id="graph", hooks=[hooks])
    result = await agent.run("card 4111: quote A", user="u")
    assert result.answer == "Done. A costs 7"
    assert {c.framework for c in hooks.calls} == {"langgraph"} and len(hooks.replies) == 2
    sent = model.said()
    assert "****" in sent and "4111" not in sent


async def test_a_failed_langchain_model_call_reaches_on_error(harness: Harness) -> None:
    class Down(PlannedChatModel):
        def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
            raise ConnectionError("the model is down")

    hooks = Redacting()
    graph = create_agent(Down(plan=[]), tools=[], middleware=[Middleware()])
    result = await harness.wrap(graph, id="down", hooks=[hooks]).run("q", user="u")
    assert result.status is RunStatus.ERROR
    assert hooks.errors == [("model", "ConnectionError"), ("run", "ConnectionError")]


async def test_without_hooks_the_middleware_and_the_sdk_run_as_they_are(
    harness: Harness,
) -> None:
    model = PlannedChatModel(plan=[])
    graph = create_agent(model, tools=[], middleware=[Middleware()])
    assert (await harness.wrap(graph, id="plain").run("q", user="u")).answer == "Done. "


async def test_openai_agents_model_calls_are_reported_to_the_hooks(harness: Harness) -> None:
    hooks = Redacting()
    target = OpenAIAgent(name="quoter", instructions="You quote.", model=PlannedModel([]))
    result = await harness.wrap(target, id="oai", hooks=[hooks]).run("quote A", user="u")
    assert result.answer == "Done. "
    [call] = hooks.calls
    assert call.framework == "openai_agents" and call.system == "You quote."
    assert len(hooks.replies) == 1


# --------------------------------------------------------------------------- Way 2


async def test_governed_code_gets_the_same_tool_hooks() -> None:
    asked: list[Decision] = []
    errors: list[str] = []

    class Checking(Guard):
        async def on_error(self, stage: str, error: Exception) -> None:
            errors.append(f"{stage}: {error}")

    async def charge(amount: int) -> str:
        if amount == 13:
            raise ValueError("unlucky")
        return f"charged {amount}"

    gov = Governance()
    calls = governed(charge, gov, side_effects="read", on_ask=asked.append, hooks=[Checking()])
    assert await calls(amount=5) == "charged 5 (checked)"
    with pytest.raises(ValueError, match="unlucky"):
        await calls(amount=13)
    assert errors == ["tool: unlucky"]
    secret = governed(quote.fn, gov, side_effects="read", on_ask=asked.append, hooks=[Guard()])
    with pytest.raises(Denied, match="quote was not run: secret SKUs are not quoted"):
        await secret(sku="secret")
    paying = governed(pay.fn, gov, on_ask=lambda d: True, hooks=[Guard()])
    assert await paying(amount=500) == "paid 100 (checked)" and paid == [100]

    class Asking(Hooks):
        async def before_tool(self, call: ToolCall) -> Ask:
            return Ask("Charge it?")

    def approve(decision: Decision) -> bool:
        asked.append(decision)
        return True

    checked = governed(charge, gov, side_effects="read", on_ask=approve, hooks=[Asking()])
    assert await checked(amount=7) == "charged 7"
    assert [d.question for d in asked] == ["Approve charge? Charge it?"]


async def test_the_model_hooks_outside_a_harness_run() -> None:
    """Code that runs LangChain or the OpenAI Agents SDK itself passes its hooks to the
    framework's own mechanism."""
    hooks = Redacting()
    model = PlannedChatModel(plan=[])
    graph = create_agent(model, tools=[], middleware=[Middleware(hooks)])
    await graph.ainvoke({"messages": [HumanMessage("card 4111")]})
    assert "4111" not in model.said() and len(hooks.calls) == 1
    unchanged = PlannedChatModel(plan=[])  # hooks that leave the call as it is
    await create_agent(unchanged, tools=[], middleware=[Middleware(Hooks())]).ainvoke(
        {"messages": [HumanMessage("card 4111")]}
    )
    assert "4111" in unchanged.said()
    agent = OpenAIAgent(name="a", instructions="i", model=PlannedModel([]))
    await Runner.run(agent, "hello", hooks=RunHooks(hooks))
    assert [c.framework for c in hooks.calls] == ["langgraph", "openai_agents"]
