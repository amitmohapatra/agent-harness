"""Live check of the one approval rule with Haiku: an MCP tool with no annotations asks, the
person picks "Always allow", and the next conversation runs it unasked — on ReAct (the
harness's mcp(...)) and on the Claude Agent SDK (Claude's own stdio MCP server, the real CLI).
Usage, with the harness and its siblings importable (the suite's ``PYTHONPATH``):
``LLM_API_KEY=... python live_check.py react OUT.json`` or ``ANTHROPIC_API_KEY=... python
live_check.py claude OUT.json``."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions
from tests.support.memory import FakeMemoryService

from trellis import Harness, Hooks, ReAct, Settings, mcp, stdio
from trellis.contracts import RunEventType
from trellis.harness import budget


class Spend(Hooks):
    """Each run's spend at its end (every attempt and resume of it included)."""

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, float]] = {}

    async def on_run_end(self, run: Any, result: Any) -> None:
        self.runs[run.run_id] = budget.spent(run)


def usd(spent: dict[str, float]) -> float:
    """The CLI's own dollars where it says them; else every token at Haiku's output price (a
    bound above the true cost: input is a fifth of that)."""
    return spent.get("cost") or spent.get("tokens", 0) * PRICE_OUT / 1e6


SERVER = str(Path(__file__).parents[3] / "tests" / "support" / "mcp_risks_server.py")
MODEL = "claude-haiku-5-5"
PRICE_OUT = 0.50  # Haiku's output price, $ per million tokens (benchmark/cost/prices.py)
ASK = "Tidy entry {n} of the ledger with the ledger's tidy tool, then say done."
SYSTEM = "You work the ledger. Use the ledger's tools when asked."


def target(kind: str, ran: Path, work: Path) -> tuple[Any, Any]:
    if kind == "react":
        source = mcp(stdio(sys.executable, SERVER, env={"RAN": str(ran)}), name="ledger")
        return ReAct(SYSTEM, MODEL, tools=[source]), "ledger-tidy"
    server = {
        "type": "stdio",
        "command": sys.executable,
        "args": [SERVER],
        "env": {"RAN": str(ran)},
    }
    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt=SYSTEM,
        max_turns=6,
        setting_sources=[],
        cwd=str(work),
        mcp_servers={"ledger": server},
        allowed_tools=["mcp__ledger__tidy"],
        tools=[],
    )
    return options, "mcp__ledger__tidy"


async def turn(agent: Any, n: int, thread: str) -> dict[str, Any]:
    """One conversation's turn, streamed for its spend; what was asked."""
    asked: list[dict[str, Any]] = []
    spent: dict[str, Any] = {}
    result = None
    errors: list[Any] = []
    async for e in agent.stream(ASK.format(n=n), user="ada", thread=thread):
        if e.type is RunEventType.INTERRUPT:
            asked.append(e.data)
        if e.type is RunEventType.RUN_ERROR:
            errors.append(e.data)
        if e.type is RunEventType.RUN_FINISHED:
            spent = e.data.get("spent") or {}
            result = e.data.get("result")
    return {"asked": asked, "spent": spent, "result": result, "errors": errors}


async def main(kind: str, out: Path) -> dict[str, Any]:
    root = out.parent / f"live-{kind}"
    root.mkdir(parents=True, exist_ok=True)
    ran, work = root / "ran", root / "work"
    work.mkdir(exist_ok=True)
    memory = FakeMemoryService()
    settings = Settings()
    if kind == "react":
        key = os.environ["LLM_API_KEY"]
        settings = Settings(llm_base_url="https://api.anthropic.com/v1/", llm_api_key=key)
    report: dict[str, Any] = {"adapter": kind, "model": MODEL, "steps": []}
    started = time.monotonic()
    async with Harness(config=settings, memory=memory.client()) as h:
        made, tool = target(kind, ran, work)
        spend = Spend()
        agent = h.wrap(made, id=f"live-{kind}", hooks=[spend])
        first = await agent.run(ASK.format(n=7), user="ada", thread="live-c1")
        step: dict[str, Any] = {
            "conversation": "live-c1",
            "status": first.status.value,
            "error": str(first.error),
        }
        if first.interrupt is not None:
            step |= {
                "asked": first.interrupt.tool_call and first.interrupt.tool_call.tool,
                "question": first.interrupt.question,
                "options": [o.value for o in first.interrupt.options],
            }
            done = await agent.resume(
                first.interrupt.interrupt_id, "answer", answer="always", reviewer="ada"
            )
            step |= {"after_always": done.status.value, "answer": done.answer}
        report["steps"].append(step)
        await h.writes.drain()
        for call in memory.named("put_catalog"):  # kept as the memory service keeps it
            for entry in call.body["tools"]:
                memory.catalog.setdefault(entry["name"], dict(entry))
        report["published"] = memory.catalog.get(tool)
        report["kept_permissions"] = [p["tool"] for p in memory.permissions.values()]
        second = await turn(h.wrap(made, id=f"live-{kind}-2", hooks=[spend]), 8, "live-c2")
        report["steps"].append({"conversation": "live-c2", **second})
    report["ran"] = ran.read_text().splitlines() if ran.exists() else []
    report["seconds"] = round(time.monotonic() - started, 1)
    report["spend"] = spend.runs
    report["usd"] = round(sum(usd(s) for s in spend.runs.values()), 6)
    return report


if __name__ == "__main__":
    found = asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
    Path(sys.argv[2]).write_text(json.dumps(found, indent=2, default=str))
