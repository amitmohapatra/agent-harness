"""Durable runs are the same for every framework: an agent of each — ``ReAct``, a function,
LangGraph, Deep Agents, the OpenAI Agents SDK, the Claude Agent SDK — is queued with
``agent.start`` (a ``RunHandle``: its id, its status, its result), and one worker executes
them all, by priority; a schedule fires a run for its person; a queued run is cancelled before
a worker takes it.

With ``RUNS_URL`` (and ``MEMORY_URL``, ``TRELLIS_API_KEY``) the queue, the schedule and the
records are agent-runs', and any worker process — ``python -m trellis.harness.worker
module:h`` — takes the runs; offline they are kept in this process, which takes the same calls.

    python -m examples.06_scenarios.runs_every_framework
"""

from __future__ import annotations

import asyncio

from agents import Agent, set_tracing_disabled
from claude_agent_sdk import ClaudeAgentOptions
from deepagents import create_deep_agent
from examples._support.offline import (
    claude_cli,
    langchain_model,
    offline_blocks,
    openai_agents_model,
    react_model,
)
from langchain.agents import create_agent

from trellis import Harness, ReAct, Runtime
from trellis.contracts import RunStatus

set_tracing_disabled(True)  # the SDK's own tracing goes to OpenAI; the harness traces via OTel
QUESTION = "Summarise yesterday's orders."
SUMMARY = "12 orders, 2 refunds."


async def clerk(question: str, agent: Runtime) -> str:
    """A function target: your own code decides (with what memory knows, when it is on)."""
    known = agent.context
    return f"{SUMMARY} (for {agent.user}{', with their context' if known else ''})"


async def main() -> None:
    async with Harness(**offline_blocks()) as h:
        targets = {
            "react": ReAct(system_prompt="You summarise.", model=react_model([SUMMARY] * 2)),
            "function": clerk,
            "langgraph": create_agent(langchain_model([SUMMARY])),
            "deepagents": create_deep_agent(model=langchain_model([SUMMARY])),
            "openai": Agent(name="clerk", model=openai_agents_model([SUMMARY])),
            "claude": ClaudeAgentOptions(**claude_cli([{"text": SUMMARY}], "trellis")),
        }
        agents = [h.wrap(target, id=name, run_timeout=120) for name, target in targets.items()]

        # queued: each a RunHandle; a higher priority is claimed first
        handles = [
            await agent.start(QUESTION, user="ada", priority=n) for n, agent in enumerate(agents)
        ]
        # cancelled while it waits: no worker ever takes it
        extra = await agents[0].start(QUESTION, user="ada")
        cancelled = await agents[0].cancel(extra.id, reason="asked twice")
        print(f"{'cancelled':>10}: {cancelled.status.value} ({extra.id})")

        # a schedule: each fire queues a run for its person (agent-runs' ticker, at its time)
        schedule = await agents[0].schedule("weekdays", user="ada", input=QUESTION)
        fired = await h.runs.schedules.fire(schedule.schedule_id)  # now, not at its time

        worker = h.worker(agents)  # in production: python -m trellis.harness.worker module:h
        while await worker.run_once():  # one queued run at a time, to its end
            pass

        for agent, handle in zip(agents, handles, strict=True):
            result = await handle.result(wait=30)
            print(f"{agent.id:>10}: {result.status.value} {result.answer!r}")
            assert result.status is RunStatus.SUCCESS, result.error
        scheduled = await h.runs.get(fired.run_id)
        assert scheduled is not None and scheduled.status is RunStatus.SUCCESS
        print(f"{'scheduled':>10}: {scheduled.status.value} {scheduled.output!r}")


if __name__ == "__main__":
    asyncio.run(main())
