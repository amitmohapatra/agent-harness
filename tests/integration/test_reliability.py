"""How a harness tool call is run, whichever framework makes it — the function target, ``ReAct``,
LangChain's ``create_agent``, Deep Agents, the OpenAI Agents SDK and the Claude Agent SDK all
call the bridge: a call takes at most its ``timeout``; one that does more than read and runs out
of time has an unknown effect, which the model is told; one that only reads is tried again
after an error that may pass; the tool reads its idempotency key from ``trellis.current()``.
And the same rules for code that is not wrapped (``governed``)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Final

import httpx
import pytest
import respx

from tests.support.adapters import BUILDERS
from tests.support.planned import Call
from tests.unit.test_sources import DOCUMENT
from trellis import Harness, ReAct, Runtime, current, openapi, tool
from trellis.contracts import ConfigurationError, RunEvent, RunEventType, ToolError
from trellis.harness.governance import Governance, governed
from trellis.harness.journal import content_key
from trellis.harness.tools import base
from trellis.harness.tools.base import ToolTimeout

UNKNOWN_TEXT: Final = (
    "transfer timed out after 0.05s; it may or may not have taken effect: check before "
    "calling it again"
)


class Ledger:
    """What the tools did: each call that started, the keys they were handed, and how often
    the flaky read was tried."""

    def __init__(self) -> None:
        self.transfers: list[int] = []
        self.keys: list[str | None] = []
        self.quotes = 0

    def tools(self) -> list[Any]:
        @tool(side_effects="write", timeout=0.05)
        async def transfer(amount: int) -> str:
            """Transfer money."""
            runtime = current()
            self.keys.append(runtime.idempotency_key if runtime else None)
            self.transfers.append(amount)
            await asyncio.sleep(5)  # the bank never answers in time
            return "sent"

        @tool(side_effects="read")
        async def quote(sku: str) -> str:
            """The price of a SKU."""
            self.quotes += 1
            if self.quotes < 3:
                raise ToolError("the price service is busy", retryable=True)
            return f"{sku} costs 7"

        return [transfer, quote]


PLAN: Final[list[Call]] = [("quote", {"sku": "A-1"}), ("transfer", {"amount": 5})]


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "RETRY_BACKOFF_SECONDS", 0.001)


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_a_write_out_of_time_is_unknown_and_a_flaky_read_is_retried(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    ledger = Ledger()
    target, tools = await BUILDERS[framework](harness, ledger.tools(), tmp_path, PLAN)
    agent = harness.wrap(target, id=f"payer-{framework}", tools=tools)
    events = [e async for e in agent.stream("pay A-1", user="u")]
    finished = events[-1]
    assert finished.type is RunEventType.RUN_FINISHED and finished.outcome is not None
    assert finished.outcome.value == "success", finished
    assert str(finished.data.get("result")).endswith(UNKNOWN_TEXT)  # what the model read
    results = _results(events)
    assert results["quote"] == {"status": "ok", "output": "A-1 costs 7"}
    assert results["transfer"]["status"] == "timeout"
    assert ledger.quotes == 3  # two errors that may pass, then the price
    assert ledger.transfers == [5]  # once: a write is never tried again
    [key] = ledger.keys
    assert key is not None and key.startswith(f"{finished.run_id}:")


def _results(events: list[RunEvent]) -> dict[str, dict[str, Any]]:
    return {
        e.data["tool"]: {"status": e.data["status"], "output": e.data["output"]}
        for e in events
        if e.type is RunEventType.TOOL_CALL_RESULT
    }


# --------------------------------------------------------------------------- the journal


async def test_a_timed_out_write_is_replayed_as_unknown_never_run_again(harness: Harness) -> None:
    ledger = Ledger()
    transfer, _ = ledger.tools()

    async def pay(input: str, agent: Runtime) -> str:
        said = await agent.tools.call("transfer", amount=5)
        answer = await agent.ask("Anything else?")
        return f"{said} / {answer}"

    agent = harness.wrap(pay, id="pay", tools=[transfer])
    paused = await agent.run("pay", user="u")
    assert paused.interrupt is not None and ledger.transfers == [5]
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="no", reviewer="u")
    assert done.answer == f"{UNKNOWN_TEXT} / no"
    assert ledger.transfers == [5]  # the resumed run read the journal, not the bank


async def test_a_read_out_of_time_says_so_and_a_timeout_of_its_own_counts(
    harness: Harness,
) -> None:
    @tool(side_effects="read", timeout=0.05)
    def slow(sku: str) -> str:
        """A sync read that blocks: it runs in a worker thread, so its timeout still holds."""
        time.sleep(0.3)
        return "late"

    @tool(side_effects="read")
    async def stuck(sku: str) -> str:
        """A read whose own client gives up."""
        raise TimeoutError("the client gave up")

    async def read(input: str, agent: Runtime) -> list[Any]:
        return [await agent.tools.call(name, sku="A") for name in ("slow", "stuck")]

    agent = harness.wrap(read, id="read", tools=[slow, stuck])
    began = time.monotonic()
    result = await agent.run("read", user="u")
    assert time.monotonic() - began < 0.25  # the loop was free while the thread slept
    assert result.answer == ["slow timed out after 0.05s", "stuck timed out after 0s"]


async def test_an_error_that_will_not_pass_is_not_retried(harness: Harness) -> None:
    tried: list[str] = []

    @tool(side_effects="read")
    def missing(sku: str) -> str:
        """A read that fails for good."""
        tried.append(sku)
        raise ValueError(f"no SKU {sku}")

    async def read(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("missing", sku="Z")

    result = await harness.wrap(read, id="read", tools=[missing]).run("read", user="u")
    assert result.answer == "missing failed: no SKU Z" and tried == ["Z"]


def test_a_timeout_is_a_number_of_seconds_over_zero() -> None:
    with pytest.raises(ConfigurationError, match="never: a timeout is a number of seconds over 0"):
        tool(lambda: None, name="never", timeout=0)
    with pytest.raises(ConfigurationError, match="model_timeout is a number of seconds over 0"):
        ReAct(system="s", model="m", model_timeout=-1)


# --------------------------------------------------------------------------- Way 2


async def test_governed_code_gets_the_same_timeouts_and_retries() -> None:
    governance = Governance()
    tries: list[int] = []

    async def lookup(sku: str) -> str:
        tries.append(1)
        if len(tries) < 2:
            raise ToolError("busy", retryable=True)
        return f"{sku}: 4"

    async def ship(order: str) -> str:
        await asyncio.sleep(5)
        return "shipped"

    reads = governed(lookup, governance, side_effects="read", on_ask=lambda d: True)
    assert await reads(sku="A") == "A: 4" and len(tries) == 2
    writes = governed(ship, governance, timeout=0.05, on_ask=lambda d: True)
    with pytest.raises(ToolTimeout, match="may or may not have taken effect") as raised:
        await writes(order="o1")
    assert raised.value.unknown

    def fails(order: str) -> str:
        raise ValueError("no such order")

    with pytest.raises(ValueError, match="no such order"):
        await governed(fails, governance, on_ask=lambda d: True)(order="o2")


# --------------------------------------------------------------------------- OpenAPI, A2A, MCP


@respx.mock
async def test_openapi_reads_are_retried_and_writes_carry_the_idempotency_key(
    harness: Harness,
) -> None:
    order = respx.get("http://erp.test/orders/o1").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, json={"id": "o1"}),
        ]
    )
    missing = respx.get("http://erp.test/orders/o9").mock(return_value=httpx.Response(404))
    created = respx.post("http://erp.test/orders").mock(
        return_value=httpx.Response(201, json={"id": "o2"})
    )

    async def ordering(input: str, agent: Runtime) -> list[Any]:
        return [
            await agent.tools.call("get_order", order_id="o1"),
            await agent.tools.call("get_order", order_id="o9"),
            await agent.tools.call("create_order", body={"sku": "a"}),
        ]

    source = openapi(DOCUMENT, timeout=5)
    assert {t.timeout for t in await source.resolve()} == {5}
    agent = harness.wrap(ordering, id="ordering", tools=[source])
    result = await agent.run("x", user="u")
    found, gone, made = result.answer
    assert found == {"id": "o1"} and order.call_count == 3
    assert gone.startswith("get_order failed: Client error '404 Not Found'")
    assert missing.call_count == 1  # a 404 will not pass: not tried again
    assert made == {"id": "o2"}
    [request] = created.calls
    key = request.request.headers["idempotency-key"]
    assert key.startswith(f"{result.run_id}:")
    assert "idempotency-key" not in order.calls[0].request.headers  # a read sends none


async def test_a_write_of_unknown_effect_is_recorded_so(
    memory_harness: Harness, memory_service: Any
) -> None:
    transfer, _ = Ledger().tools()

    async def pay(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("transfer", amount=5)

    agent = memory_harness.wrap(pay, id="pay", tools=[transfer])
    assert (await agent.run("pay", user="u")).answer == UNKNOWN_TEXT
    await memory_harness.writes.drain()
    [recorded] = memory_service.named("record_tool")
    assert recorded.body["status"] == "timeout"
    assert recorded.body["error_class"] == "OutcomeUnknown"


async def test_the_key_a_tool_hands_on_is_the_runs_and_the_calls_own_names_the_call(
    harness: Harness,
) -> None:
    keys: list[str | None] = []

    @tool(side_effects="irreversible")
    def refund(order: str) -> str:
        """Refund an order."""
        runtime = current()
        keys.append(runtime.idempotency_key if runtime else None)
        return "refunded"

    async def refunding(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("refund", order=input)

    agent = harness.wrap(refunding, id="refunds", tools=[refund])
    first, second = [await agent.run("o1", user="u") for _ in range(2)]
    for paused in (first, second):
        assert paused.interrupt is not None and paused.interrupt.tool_call is not None
        # the call itself, the same in every run: what approvals are learned from
        assert paused.interrupt.tool_call.idempotency_key == content_key(
            "call", "refund", {"order": "o1"}
        )
        await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="cfo")
    assert keys[0] != keys[1]  # each run's own: a second refund is not deduplicated away
    assert all(k and k.startswith(r.run_id) for k, r in zip(keys, (first, second), strict=True))
