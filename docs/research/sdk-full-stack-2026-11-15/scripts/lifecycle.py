# ruff: noqa: PLC0415 - a research script: it reports with print, imports an SDK where a case needs it
"""Part B's run lifecycle on the real stack, per SDK: a pause answered in another process (a
restart), a queued run executed by a worker, and a run cancelled while its tool works.

Every target is built at import, the same way in each process (as a worker needs). The tools
leave a mark on disk, so a second process can tell whether a call ran and how often.

Usage: python lifecycle.py <sdk> <step> [state.json]
  step: pause (process 1: run until the approval; write the interrupt to state.json),
        resume (process 2: a new harness answers it), queued, cancel
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from common import MODEL, case, harness, records, uid

MARKS = Path(os.environ.get("SDKFULL_MARKS", "/var/tmp/cons/sdkfull-tmp/marks"))
MARKS.mkdir(parents=True, exist_ok=True)
ASK = "Wire 300 to account ACME-1 now: call the wire tool, do not ask me first."
SLOW = "Call the slow_report tool for region north, then tell me its result."


def mark(name: str, value: str) -> str:
    with (MARKS / name).open("a") as out:
        out.write(value + "\n")
    return f"{name} done for {value}"


def marks(name: str) -> list[str]:
    path = MARKS / name
    return path.read_text().split() if path.exists() else []


async def slow_wait(gate: str) -> None:
    mark("slow-started", gate)
    await asyncio.sleep(60)


# --------------------------------------------------------------------------- targets
def langgraph_target() -> Any:
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from lc_agent import model

    @tool
    def wire(amount: int, to: str) -> str:
        """Wire money to an account."""
        return mark("wire-langgraph", to)

    wire.metadata = {"side_effects": "irreversible"}

    @tool
    async def slow_report(region: str) -> str:
        """A report that takes a while."""
        await slow_wait("langgraph")
        return "report ready"

    slow_report.metadata = {"side_effects": "read"}
    return create_agent(model(), tools=[wire, slow_report])


def deepagents_target() -> Any:
    from deepagents import create_deep_agent
    from langchain_core.tools import tool
    from lc_agent import model

    @tool
    def wire(amount: int, to: str) -> str:
        """Wire money to an account."""
        return mark("wire-deepagents", to)

    wire.metadata = {"side_effects": "irreversible"}

    @tool
    async def slow_report(region: str) -> str:
        """A report that takes a while."""
        await slow_wait("deepagents")
        return "report ready"

    slow_report.metadata = {"side_effects": "read"}
    return create_deep_agent(model(), tools=[wire, slow_report])


def react_target() -> Any:
    from trellis import ReAct, tool

    @tool(side_effects="irreversible")
    def wire(amount: int, to: str) -> str:
        """Wire money to an account."""
        return mark("wire-react", to)

    @tool(side_effects="read")
    async def slow_report(region: str) -> str:
        """A report that takes a while."""
        await slow_wait("react")
        return "report ready"

    return ReAct("You make payments and fetch reports.", MODEL, tools=[wire, slow_report])


def openai_target() -> Any:
    from agents import Agent, function_tool
    from oai import model

    @function_tool(needs_approval=True)
    def wire(amount: int, to: str) -> str:
        """Wire money to an account."""
        return mark("wire-openai_agents", to)

    @function_tool
    async def slow_report(region: str) -> str:
        """A report that takes a while."""
        await slow_wait("openai")
        return "report ready"

    return Agent(
        name="payer",
        instructions="You make payments and fetch reports.",
        model=model(),
        tools=[wire, slow_report],
    )


def claude_target() -> Any:
    from claude import options
    from claude_agent_sdk import create_sdk_mcp_server, tool
    from mcp.types import ToolAnnotations

    @tool(
        "wire",
        "Wire money to an account.",
        {"amount": int, "to": str},
        annotations=ToolAnnotations(destructiveHint=True),
    )
    async def wire(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": mark("wire-claude_agent_sdk", args["to"])}]}

    @tool(
        "slow_report",
        "A report that takes a while.",
        {"region": str},
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def slow_report(args: dict[str, Any]) -> dict[str, Any]:
        await slow_wait("claude")
        return {"content": [{"type": "text", "text": "report ready"}]}

    cwd = Path(tempfile.gettempdir()) / "lifecycle-claude"
    cwd.mkdir(exist_ok=True)
    return options(
        cwd,
        mcp_servers={"pay": create_sdk_mcp_server("pay", tools=[wire, slow_report])},
        allowed_tools=["mcp__pay__wire", "mcp__pay__slow_report"],
    )


def kimi_target() -> Any:
    from kimi_code import env

    from trellis import kimi

    root = Path(tempfile.gettempdir()) / "lifecycle-kimi"
    (root / "cwd").mkdir(parents=True, exist_ok=True)
    (root / "home").mkdir(exist_ok=True)
    return kimi(model=MODEL, cwd=str(root / "cwd"), env=env(root / "home"))


KIMI_ASK = "Run the shell command `echo paid >> ledger.txt` now."
KIMI_SLOW = "Run the shell command `sleep 60 && echo late` and wait for it."
TARGETS = {
    "langgraph": langgraph_target,
    "deepagents": deepagents_target,
    "react": react_target,
    "openai_agents": openai_target,
    "claude_agent_sdk": claude_target,
    "kimi": kimi_target,
}


def wired(sdk: str) -> list[str]:
    if sdk == "kimi":
        f = Path(tempfile.gettempdir()) / "lifecycle-kimi" / "cwd" / "ledger.txt"
        return f.read_text().split() if f.exists() else []
    return marks(f"wire-{sdk}")


# --------------------------------------------------------------------------- steps
def write_state(state: Path, saved: dict[str, Any]) -> None:
    state.write_text(json.dumps(saved))


def read_state(state: Path) -> dict[str, Any]:
    return json.loads(state.read_text())


async def pause(sdk: str, state: Path) -> dict[str, Any]:
    async with harness("stack") as h:
        agent = h.wrap(TARGETS[sdk](), id=f"lc-{sdk}-restart")
        user = f"life-{uid()}"
        r = await agent.run(KIMI_ASK if sdk == "kimi" else ASK, user=user, thread=uid())
        saved = {
            "interrupt": r.interrupt.interrupt_id if r.interrupt else None,
            "run_id": r.run_id,
            "before": wired(sdk),
        }
        write_state(state, saved)
        return {
            "ok": r.interrupt is not None,
            "status": str(r.status),
            "run_id": r.run_id,
            "asked": r.interrupt.question if r.interrupt else None,
            "wired": wired(sdk),
        }


async def resume(sdk: str, state: Path) -> dict[str, Any]:
    saved = read_state(state)
    async with harness("stack") as h:  # a new process, a new harness: nothing in memory
        agent = h.wrap(TARGETS[sdk](), id=f"lc-{sdk}-restart")
        r = await agent.resume(saved["interrupt"], "approve", reviewer="lee")
        rec = records(r.run_id)
    added = len(wired(sdk)) - len(saved["before"])
    return {
        "ok": str(r.status).endswith("SUCCESS") and added == 1,
        "status": str(r.status),
        "answer": str(r.answer)[:300],
        "calls_made": added,
        "error": str(r.error) if r.error else None,
        "records": rec,
    }


async def queued(sdk: str) -> dict[str, Any]:
    async with harness("stack") as h:
        agent = h.wrap(TARGETS[sdk](), id=f"lc-{sdk}-queued")
        handle = await agent.start(
            "What is 17 times 3? Answer with the number only.", user=f"life-{uid()}"
        )
        worker = h.worker([agent])
        for _ in range(30):
            await worker.run_once()
            got = await h.runs.get(handle.id)
            if got is not None and str(got.status).endswith(("SUCCESS", "ERROR", "PARTIAL")):
                break
        rec = records(handle.id)
    out = str((got.output if got else None) or "")
    return {
        "ok": str(got.status if got else "").endswith("SUCCESS") and "51" in out,
        "status": str(got.status if got else None),
        "output": out[:200],
        "records": rec,
    }


async def cancel(sdk: str) -> dict[str, Any]:
    gate = f"started-{sdk}"
    before = len(marks("slow-started"))
    async with harness("stack") as h:
        agent = h.wrap(TARGETS[sdk](), id=f"lc-{sdk}-cancel")
        events: list[Any] = []

        async def consume() -> None:
            async for ev in agent.stream(
                KIMI_SLOW if sdk == "kimi" else SLOW, user=f"life-{uid()}", thread=uid()
            ):
                events.append(ev)

        task = asyncio.create_task(consume())
        for _ in range(240):  # until the tool works (its mark), or Kimi's command started
            await asyncio.sleep(0.5)
            started = len(marks("slow-started")) > before or any(
                "TOOL_CALL_START" in str(e.type) for e in events
            )
            if started or task.done():
                break
        run_id = events[0].run_id if events else None
        cancelled = await agent.cancel(run_id, reason="enough") if run_id else None
        await asyncio.wait_for(task, 60)
        rec = records(run_id) if run_id else None
    last = events[-1] if events else None
    status = rec["run"]["status"] if rec else None
    return {
        "ok": status == "CANCELLED",
        "status": status,
        "gate": gate,
        "cancel_returned": str(cancelled.status) if cancelled else None,
        "last_event": str(last.type) if last else None,
        "last_data": json.dumps(last.data, default=str)[:300] if last else None,
        "records": rec,
    }


async def main() -> None:
    sdk, step = sys.argv[1], sys.argv[2]
    state = Path(sys.argv[3]) if len(sys.argv) > 3 else MARKS / f"{sdk}.json"
    body = {
        "pause": lambda: pause(sdk, state),
        "resume": lambda: resume(sdk, state),
        "queued": lambda: queued(sdk),
        "cancel": lambda: cancel(sdk),
    }[step]
    await case(sdk, f"lifecycle_{step}", "stack", body)


if __name__ == "__main__":
    asyncio.run(main())
