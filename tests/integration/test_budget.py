"""A run's budget whatever the framework (``budget.py``): tool calls counted as they start —
the calls of one step decided together, a sub-agent's toward its parent's, a resumed run's
from what it spent — and a read tool's retries: after an error that may
pass, waiting what its ``Retry-After`` says when there is time left; a write never."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.support.arm import arm
from trellis import Budget, Harness, Runtime, tool
from trellis.contracts import ConfigurationError, RunEventType, RunStatus, ToolError
from trellis.harness import budget
from trellis.harness.tools import base


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "RETRY_BACKOFF_SECONDS", 0.001)


class Shop:
    """What the tools did."""

    def __init__(self, fails: int = 0, after: float | None = None) -> None:
        self.reads: list[str] = []
        self.writes: list[str] = []
        self.fails = fails
        self.after = after

    def tools(self) -> list[Any]:
        @tool(side_effects="read")
        async def stock(sku: str) -> int:
            """Units of a SKU in stock."""
            self.reads.append(sku)
            await asyncio.sleep(0)
            if len(self.reads) <= self.fails:
                error = ToolError("the stock service is busy", retryable=True)
                error.retry_after = self.after  # type: ignore[attr-defined]
                raise error
            return 7

        @tool(side_effects="write")
        async def order(sku: str) -> str:
            """Order a SKU."""
            self.writes.append(sku)
            raise ToolError("the shop is busy", retryable=True)

        return [stock, order]


# --------------------------------------------------------------------------- budget
async def test_the_calls_of_one_step_past_the_limit_are_not_started(harness: Harness) -> None:
    shop = Shop()

    async def counter(input: str, agent: Runtime) -> list[Any]:
        calls = [agent.tools.call("stock", sku=s) for s in ("a", "b", "c")]
        return list(await asyncio.gather(*calls))

    agent = arm(harness.wrap(counter, id="count", budget=Budget(max_tool_calls=2)), shop.tools())
    result = await agent.run("count", user="u")
    assert sorted(shop.reads) == ["a", "b"]  # the two under way finished; the third never ran
    assert result.answer[:2] == [7, 7]
    assert str(result.answer[2]).startswith("stock was not run: " + budget.SPENT)
    assert result.status is RunStatus.PARTIAL and result.error is not None
    assert result.error.code == budget.BUDGET_EXCEEDED


async def test_a_runs_own_budget_goes_over_its_agents_and_is_kept_with_its_record(
    harness: Harness,
) -> None:
    """``run``/``start(budget=)``: this run's cap, over the agent's — kept with the queued run,
    so the worker's attempt holds to it."""
    shop = Shop()

    async def counter(input: str, agent: Runtime) -> list[Any]:
        return [await agent.tools.call("stock", sku=s) for s in ("a", "b", "c")]

    agent = arm(harness.wrap(counter, id="count", budget=Budget(max_tool_calls=3)), shop.tools())
    handle = await agent.start("count", user="u", budget=Budget(max_tool_calls=1))
    record = await handle.status()
    assert record.metadata["budget"] == {"max_tool_calls": 1}
    assert Budget.of(record.metadata["budget"]) == Budget(max_tool_calls=1)
    assert await harness.worker([agent]).run_once()
    done = await handle.result(wait=5)
    assert shop.reads == ["a"] and done.status is RunStatus.PARTIAL
    assert Budget.of(None) is None


async def test_a_sub_agents_calls_count_toward_its_parents_budget(harness: Harness) -> None:
    shop = Shop()

    async def counter(input: str, agent: Runtime) -> list[Any]:
        return [await agent.tools.call("stock", sku=s) for s in ("a", "b", "c")]

    child = arm(harness.wrap(counter, id="child"), shop.tools())

    async def lead(input: str, agent: Runtime) -> Any:
        return await agent.tools.call(child.id, message="count")

    parent = arm(harness.wrap(lead, id="lead", budget=Budget(max_tool_calls=2)), [child])
    result = await parent.run("count", user="u")
    # the parent's call to the child is one, the child's first read the second: the limit
    assert shop.reads == ["a"], result
    assert result.status is RunStatus.PARTIAL


async def test_a_resumed_run_goes_on_from_what_it_spent(harness: Harness) -> None:
    shop = Shop()

    async def counter(input: str, agent: Runtime) -> list[Any]:
        first = await agent.tools.call("stock", sku="a")
        await agent.ask("Count b too?")
        return [first, await agent.tools.call("stock", sku="b")]

    agent = arm(harness.wrap(counter, id="count", budget=Budget(max_tool_calls=1)), shop.tools())
    paused = await agent.run("count", user="u")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="r")
    assert shop.reads == ["a"]  # a replayed from the journal, b refused: one call was the budget
    assert done.status is RunStatus.PARTIAL
    assert str(done.answer[1]).startswith("stock was not run")


async def test_without_a_budget_nothing_is_refused(harness: Harness) -> None:
    shop = Shop()

    async def counter(input: str, agent: Runtime) -> list[Any]:
        return [await agent.tools.call("stock", sku=s) for s in ("a", "b")]

    agent = arm(harness.wrap(counter, id="count"), shop.tools())
    result = await agent.run("count", user="u")
    assert result.answer == [7, 7] and result.status is RunStatus.SUCCESS


def test_a_budget_is_numbers_from_zero() -> None:
    assert Budget() == Budget(max_tokens=None, max_cost=None, max_tool_calls=None)  # no cap
    with pytest.raises(ConfigurationError, match="max_cost is a number from 0"):
        Budget(max_cost=-1)


async def test_a_budget_in_dollars_counts_what_the_gateway_said(harness: Harness) -> None:
    async def spender(input: str, agent: Runtime) -> Any:
        budget.account(agent, model="m", input_tokens=10, output_tokens=0, cost=0.6)
        assert budget.reached(agent) is None
        budget.account(agent, model="m", input_tokens=10, output_tokens=0, cost=0.6)
        return budget.reached(agent)

    agent = harness.wrap(spender, id="spend", budget=Budget(max_cost=1.0))
    said = (await agent.run("spend", user="u")).answer
    assert said == "max_cost 1 reached (1.2 spent)"


# --------------------------------------------------------------------------- tool retries
async def test_a_read_is_retried_after_an_error_that_may_pass(harness: Harness) -> None:
    shop = Shop(fails=2)

    async def reader(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("stock", sku="a")

    result = await arm(harness.wrap(reader, id="r"), shop.tools()).run("read", user="u")
    assert result.answer == 7 and shop.reads == ["a", "a", "a"]


async def test_a_read_waits_what_retry_after_says_when_there_is_time(harness: Harness) -> None:
    shop = Shop(fails=1, after=0.05)

    async def reader(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("stock", sku="a")

    began = asyncio.get_running_loop().time()
    result = await arm(harness.wrap(reader, id="r"), shop.tools()).run("read", user="u")
    assert result.answer == 7 and asyncio.get_running_loop().time() - began >= 0.05


async def test_a_retry_after_past_the_time_left_is_not_waited(harness: Harness) -> None:
    shop = Shop(fails=1, after=30.0)

    async def reader(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("stock", sku="a")

    agent = arm(harness.wrap(reader, id="r", run_timeout=2), shop.tools())
    result = await agent.run("read", user="u")
    assert result.answer == "stock failed: the stock service is busy" and shop.reads == ["a"]


async def test_a_write_is_never_retried(harness: Harness) -> None:
    shop = Shop()

    async def writer(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("order", sku="a")

    result = await arm(harness.wrap(writer, id="w"), shop.tools()).run("order", user="u")
    assert result.answer == "order failed: the shop is busy" and shop.writes == ["a"]


# --------------------------------------------------------------------------- the pieces
def test_when_an_error_says_to_come_back() -> None:
    import httpx

    assert base.retry_after(ValueError("no")) is None
    caused = ToolError("busy", retryable=True)
    caused.__cause__ = httpx.HTTPStatusError(
        "busy",
        request=httpx.Request("GET", "http://x"),
        response=httpx.Response(503, headers={"Retry-After": "3"}),
    )
    assert base.retry_after(caused) == 3.0
    dated = httpx.HTTPStatusError(
        "busy",
        request=httpx.Request("GET", "http://x"),
        response=httpx.Response(503, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
    )
    assert base.retry_after(dated) is None  # a date is not read


def test_a_reply_with_no_model_message_spends_nothing() -> None:
    from types import SimpleNamespace

    runtime: Any = SimpleNamespace()  # never read: there is nothing to account
    budget.account_reply(runtime, SimpleNamespace(result=["not a message"]))


def test_the_calls_of_an_invocation_with_no_user_message_count_from_its_start() -> None:
    from types import SimpleNamespace

    from langchain_core.messages import AIMessage

    from trellis.harness import automatic

    request: Any = SimpleNamespace(state={"messages": [AIMessage("a"), AIMessage("b")]})
    assert automatic.calls_made(request) == 2


# --------------------------------------------------------------------------- what a run spent
async def test_a_finished_run_says_what_it_spent_on_its_end(harness: Harness) -> None:
    """Its tokens, model calls, dollars and tool calls — counted in its journal while it works
    — are on its ``RUN_FINISHED`` event, whatever its ending (ADV-11)."""
    shop = Shop()

    async def spender(input: str, agent: Runtime) -> Any:
        budget.account(agent, model="m", input_tokens=100, output_tokens=20, cached=40, cost=0.25)
        budget.account(agent, model="m", input_tokens=50, output_tokens=5, cost=0.5)
        await agent.tools.call("stock", sku="a")
        if input == "fail":
            raise ValueError("broken")
        return "done"

    agent = arm(harness.wrap(spender, id="spender"), shop.tools())
    spent = {"tokens": 175, "cached_tokens": 40, "cost": 0.75, "model_calls": 2, "tool_calls": 1}
    for asked in ("ok", "fail"):
        events = [e async for e in agent.stream(asked, user="u")]
        finished = events[-1]
        assert finished.type is RunEventType.RUN_FINISHED, finished
        assert finished.data["spent"] == spent, (asked, finished.data)


async def test_a_run_that_spent_nothing_says_so(harness: Harness) -> None:
    async def idle(input: str, agent: Runtime) -> str:
        return "ok"

    agent = harness.wrap(idle, id="idle")
    events = [e async for e in agent.stream("x", user="u")]
    assert events[-1].data["spent"] == dict.fromkeys(budget.SPENT_KEYS, 0)
