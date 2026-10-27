"""Hooks around runs and tool calls: the tool hooks in the bridge on every adapter (deny,
rewrite, ask — journaled — and the outcome changed), the run hooks in the pipeline, a failed
model call reaching ``on_error``, model calls left to each framework's own callbacks (LangChain
middleware, the OpenAI Agents SDK's ``AgentHooks``) — a hook class with the removed model hooks
refused — and the same hooks in Way 2 (``governed``; the middleware and ``RunHooks`` outside a
harness run)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from agents import Agent as OpenAIAgent
from agents import AgentHooks, Runner
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage, HumanMessage

from tests.support.adapters import BUILDERS
from tests.support.arm import arm, native_tools
from tests.support.chat_model import ScriptedChatModel
from tests.support.planned import Call, PlannedChat, PlannedChatModel, PlannedModel
from trellis import Ask, Deny, Harness, Hooks, ReAct, Rewrite, Runtime, Settings, tool
from trellis.contracts import ConfigurationError, RunEventType, RunStatus, ToolCall, ToolOutcome
from trellis.harness.governance import Decision, Denied, Governance, governed
from trellis.harness.hooks.openai_agents import ModelHooks as RunHooks
from trellis.harness.middleware import ModelHooks as Middleware
from trellis.harness.result import Result
from trellis.harness.runtime import ModelTimeout

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
    agent = arm(harness.wrap(target, id=f"guarded-{framework}", hooks=[Guard()]), tools)
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
    agent = arm(harness.wrap(target, id=f"asking-{framework}", hooks=[hooks]), tools)
    paused = await agent.run("quote A", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused
    assert paused.interrupt.question == "Approve quote? Quote A for this customer?"
    assert paused.interrupt.assignee == "role:sales" and quoted == []
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="sales")
    assert done.status is RunStatus.SUCCESS and done.answer == "Done. A costs 7"
    assert quoted == ["A"] and hooks.asked == 1  # the resumed run read the journal


async def test_hooks_run_in_the_order_given() -> None:
    order: list[str] = []

    class Named(Hooks):
        def __init__(self, name: str) -> None:
            self.name = name

        async def before_tool(self, call: ToolCall) -> Rewrite:
            order.append(self.name)
            return Rewrite({"sku": f"{call.args['sku']}-{self.name}"})

    async def fn(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("quote", sku="A")

    async with Harness(config=Settings()) as h:
        hooks = [Named("first"), Named("second")]
        result = await arm(h.wrap(fn, id="ordered", hooks=hooks), [quote]).run("q", user="u")
    assert order == ["first", "second"] and result.answer == "A-first-second costs 7"


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

    agent = arm(harness.wrap(fn, id="watched", hooks=[Watching()]), [broken])
    paused = await agent.run("go", user="u")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="u")
    assert done.answer == "broken failed: no such SKU / yes"
    failed = await agent.run("fail", user="u")

    async def slow(input: str, agent: Runtime) -> str:
        await asyncio.sleep(1)
        return "late"

    timed_out = await harness.wrap(slow, id="slow", hooks=[Watching()], run_timeout=0.01).run(
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


# --------------------------------------------------------------------------- model calls


class ModelErrors(Hooks):
    """The errors a run meets, by stage."""

    def __init__(self) -> None:
        self.errors: list[tuple[str, str]] = []

    async def on_error(self, stage: str, error: Exception) -> None:
        self.errors.append((stage, type(error).__name__))


class Redact(AgentMiddleware):
    """LangChain's own slot for a model call: the user's card number out of what is sent."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        self.calls += 1
        return await handler(request.override(messages=[_redacted(m) for m in request.messages]))


def _redacted(message: BaseMessage) -> BaseMessage:
    if isinstance(message.content, str):
        return message.model_copy(update={"content": message.content.replace("4111", "****")})
    return message


async def test_a_react_model_call_is_its_own_middleware_s_and_a_failed_one_reaches_on_error(
    harness: Harness,
) -> None:
    redact = Redact()
    model = PlannedChat([("quote", {"sku": "A"})])
    target = ReAct(system_prompt="s", model=model, middleware=[redact])
    agent = arm(harness.wrap(target, id="r"), [quote])
    result = await agent.run("card 4111: quote A", user="u")
    assert result.answer == "Done. A costs 7" and redact.calls == 2
    sent = model.said()
    assert "****" in sent and "4111" not in sent  # the model never read the card

    class Failing(ScriptedChatModel):
        async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
            raise ConnectionError("the model is down")

    hooks = ModelErrors()
    failing = harness.wrap(
        ReAct(system_prompt="s", model=Failing(turns=[])), id="down", hooks=[hooks]
    )
    assert (await failing.run("q", user="u")).status is RunStatus.ERROR
    assert hooks.errors == [("model", "ConnectionError"), ("run", "ConnectionError")]


async def test_a_react_model_call_out_of_time_is_a_model_error_the_hooks_see(
    harness: Harness,
) -> None:
    hooks = ModelErrors()
    target = ReAct(system_prompt="s", model=ScriptedChatModel(turns=[], stall=5), call_timeout=0.01)
    result = await harness.wrap(target, id="slow", hooks=[hooks]).run("q", user="u")
    assert result.status is RunStatus.ERROR
    assert hooks.errors[0] == ("model", ModelTimeout.__name__)  # a ModelError


@pytest.mark.parametrize("deep", [False, True], ids=["create_agent", "deepagents"])
async def test_a_graph_s_model_calls_are_its_own_middleware_s_beside_the_harness_s(
    harness: Harness, deep: bool
) -> None:
    redact = Redact()
    model = PlannedChatModel(plan=[("quote", {"sku": "A"})])
    tools = await native_tools(harness, quote, framework="deepagents" if deep else "langgraph")
    graph = (
        create_deep_agent(model=model, tools=tools, middleware=[redact, Middleware()])
        if deep
        else create_agent(model, tools=tools, middleware=[redact, Middleware()])
    )
    result = await harness.wrap(graph, id="graph").run("card 4111: quote A", user="u")
    assert result.answer == "Done. A costs 7" and redact.calls == 2
    sent = model.said()
    assert "****" in sent and "4111" not in sent


async def test_a_failed_langchain_model_call_reaches_on_error(harness: Harness) -> None:
    class Down(PlannedChatModel):
        def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
            raise ConnectionError("the model is down")

    hooks = ModelErrors()
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


async def test_openai_agents_model_calls_are_its_own_agent_hooks_s(harness: Harness) -> None:
    seen: list[str | None] = []

    class Watching(AgentHooks[Any]):
        async def on_llm_start(
            self, context: Any, agent: Any, system_prompt: Any, input_items: Any
        ) -> None:
            seen.append(system_prompt)

        async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
            seen.append("replied")

    model = PlannedModel([])
    target = OpenAIAgent(name="quoter", instructions="You quote.", model=model, hooks=Watching())
    result = await harness.wrap(target, id="oai").run("quote A", user="u")
    assert result.answer == "Done. " and seen == ["You quote.", "replied"]


class Legacy(Hooks):
    async def before_model(self, call: Any) -> None:
        return None


class LegacyReply(Hooks):
    async def after_model(self, call: Any, reply: Any) -> None:
        return None


@pytest.mark.parametrize(("hook", "name"), [(Legacy, "before_model"), (LegacyReply, "after_model")])
async def test_a_hook_with_a_removed_model_hook_is_refused_where_it_is_given(
    harness: Harness, hook: type[Hooks], name: str
) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return input

    with pytest.raises(ConfigurationError, match=rf"{hook.__name__}\.{name} was removed: .*own"):
        harness.wrap(fn, id="legacy", hooks=[hook()])
    with pytest.raises(ConfigurationError, match=rf"{name} was removed"):
        governed(fn, Governance(), side_effects="read", on_ask=print, hooks=[hook()])


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


async def test_the_middleware_and_the_run_hooks_step_aside_outside_a_harness_run() -> None:
    """Code that runs LangChain or the OpenAI Agents SDK itself may keep the harness's model-call
    pieces: outside a harness run they let every call through as it is."""
    model = PlannedChatModel(plan=[])
    graph = create_agent(model, tools=[], middleware=[Middleware()])
    await graph.ainvoke({"messages": [HumanMessage("card 4111")]})
    assert "4111" in model.said()
    agent = OpenAIAgent(name="a", instructions="i", model=PlannedModel([]))
    assert (await Runner.run(agent, "hello", hooks=RunHooks())).final_output == "Done. "
