"""One worker process of the load benchmark: a harness (memory, the run store and tracing off,
as the cost benchmark's) running ``ReAct`` agents, ``concurrency`` runs at once, until it has
run ``runs`` of them or ``seconds`` have passed — each run's latency and outcome, and the
process's CPU seconds and memory, sampled as it goes.

The agent is a two-step task: a tool call, then the answer. Its model is a gateway model name
(``model``: the latency is the model's and the harness's), or ``Paced`` — a scripted model that
answers after ``pace`` seconds (no model: the latency is the harness's own, under load)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from benchmark import procstat

#: The task each run does, and the tool it needs.
TASK = "How many units of SKU a-1 are in stock?"


@dataclass(frozen=True)
class Plan:
    """What one worker process runs."""

    concurrency: int
    runs: int | None = None  # None: until ``seconds``
    seconds: float | None = None
    model: str | None = None  # None: Paced
    pace: float = 0.2
    window: int = 8192
    sample_seconds: float = 5.0
    run_seconds: float = 600.0


@dataclass
class Measured:
    latencies: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    cpu_seconds: float | None = None
    peak_rss_mb: float | None = None
    #: (seconds since the start, resident MB, runs done)
    samples: list[tuple[float, float | None, int]] = field(default_factory=list)


class Paced(BaseChatModel):
    """A model that answers after ``pace`` seconds: a tool call to ``stock``, then the answer
    once the tool's result is in."""

    pace: float = 0.2

    @property
    def _llm_type(self) -> str:
        return "paced"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, Any]:  # type: ignore[override]
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("Paced answers asynchronously")  # pragma: no cover

    async def _agenerate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        await asyncio.sleep(self.pace)
        if isinstance(messages[-1], ToolMessage):
            message = AIMessage(content=f"{messages[-1].content} units.")
        else:
            message = AIMessage(
                content="",
                tool_calls=[{"name": "stock", "args": {"sku": "a-1"}, "id": f"c{time.time_ns()}"}],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def stock(sku: str) -> int:
    """Units of a SKU in stock.

    Args:
        sku: The SKU, such as a-1.
    """
    return 7


def agent(h: Any, plan: Plan, without: frozenset[str] = frozenset()) -> Any:
    """The agent under load, wrapped by ``h`` (``without``: the parts it runs without)."""
    from trellis import ReAct, tool

    model: Any = plan.model or Paced(pace=plan.pace)
    target = ReAct(
        "You answer stock questions with the stock tool.", model, context_window=plan.window
    )
    return h.wrap(
        target,
        id="load",
        tools=[tool(side_effects="read")(stock)],
        timeout=plan.run_seconds,
        without=set(without),
    )


async def work(plan: Plan) -> Measured:
    """Runs as ``plan`` says, ``concurrency`` at once, measured."""
    from benchmark.cost.run import harness
    from trellis.contracts import RunStatus

    measured = Measured()
    began = time.monotonic()
    cpu = procstat.cpu_seconds()
    started = 0
    deadline = None if plan.seconds is None else began + plan.seconds

    def more() -> bool:
        if plan.runs is not None and started >= plan.runs:
            return False
        return deadline is None or time.monotonic() < deadline

    async with harness() as h:
        wrapped = agent(h, plan)

        async def one(n: int) -> None:
            t = time.monotonic()
            try:
                result = await wrapped.run(TASK, user=f"load-{n}", thread=f"load-{n}")
            except Exception as exc:  # a failed run is a measured error
                measured.errors.append(f"{type(exc).__name__}: {exc}"[:200])
                return
            if result.status is not RunStatus.SUCCESS:
                said = result.error.message if result.error else result.status.value
                measured.errors.append(str(said)[:200])
                return
            measured.latencies.append(time.monotonic() - t)

        async def lane() -> None:
            nonlocal started
            while more():
                started += 1
                await one(started)

        async def sampler() -> None:
            while True:
                done = len(measured.latencies) + len(measured.errors)
                measured.samples.append((time.monotonic() - began, procstat.rss_mb(), done))
                await asyncio.sleep(plan.sample_seconds)

        sampling = asyncio.create_task(sampler())
        await asyncio.gather(*(lane() for _ in range(plan.concurrency)))
        sampling.cancel()
    done = len(measured.latencies) + len(measured.errors)
    measured.samples.append((time.monotonic() - began, procstat.rss_mb(), done))
    used = procstat.cpu_seconds()
    measured.cpu_seconds = None if used is None or cpu is None else used - cpu
    measured.peak_rss_mb = procstat.peak_rss_mb()
    return measured


def process(plan: Plan) -> Measured:
    """A worker process's entry: its own event loop."""
    return asyncio.run(work(plan))
