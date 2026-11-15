"""Claude Code (the CLI bundled with claude-agent-sdk) on Haiku through the harness: six probes.

Usage: ANTHROPIC_KEY_FILE=<file> python claude_probe.py <out-dir> [probe ...]

The key is read from the file into the CLI's environment only; never printed. Each probe
writes what it saw to <out-dir>/<probe>.json; the CLI's own transcripts (the provider's usage
per reply) stay under <out-dir>/home/.claude.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

OUT = Path(sys.argv[1]).resolve()
CHOSEN = set(sys.argv[2:])
KEY = Path(os.environ.pop("ANTHROPIC_KEY_FILE")).read_text().strip()
for name in list(os.environ):
    if name.startswith(("CLAUDE", "ANTHROPIC")):
        del os.environ[name]
HOME = OUT / "home"
HOME.mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(HOME)

from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, HookMatcher  # noqa: E402

import trellis  # noqa: E402
from trellis import Harness, Settings, tool  # noqa: E402
from trellis.contracts import RunStatus  # noqa: E402
from trellis.harness import budget  # noqa: E402
from trellis.harness.adapters.base import TOOLS_ATTRIBUTE  # noqa: E402

#: every reply the harness accounted (budget.account, each adapter's path), by run
ACCOUNTED: list[dict[str, Any]] = []
_account = budget.account


def _accounting(runtime: Any, **kw: Any) -> None:
    _account(runtime, **kw)
    ACCOUNTED.append(
        {
            "t": round(time.time(), 3),
            "run": runtime.run_id,
            "parent": getattr(runtime.parent, "run_id", None),
            **{k: v for k, v in kw.items() if k != "model"},
        }
    )


budget.account = _accounting  # type: ignore[assignment]

MODEL = "claude-haiku-5-5"
PRICE = {"input": 0.10, "cached": 0.01, "write": 0.125, "output": 0.50}  # benchmark.cost.prices
ENV = {"ANTHROPIC_API_KEY": KEY, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
if os.environ.get("METER_URL"):  # meter_proxy.py: the provider's usage of every response
    ENV["ANTHROPIC_BASE_URL"] = os.environ["METER_URL"]
CWD = OUT / "cwd"
CWD.mkdir(exist_ok=True)


def options(**kw: Any) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(model=MODEL, setting_sources=[], cwd=str(CWD), env=dict(ENV), **kw)


def hooks(seen: list[dict[str, Any]]) -> dict[str, list[HookMatcher]]:
    def one(event: str) -> list[HookMatcher]:
        async def own(given: Any, tool_use_id: str | None, context: Any) -> Any:
            run = trellis.current()
            seen.append(
                {
                    "event": event,
                    "agent_id": given.get("agent_id"),
                    "tool": given.get("tool_name"),
                    "in_run": run is not None,
                    "at": round(time.monotonic(), 2),
                }
            )
            return {}

        return [HookMatcher(hooks=[own])]

    return {e: one(e) for e in ("SubagentStop", "UserPromptSubmit", "PreToolUse", "Stop")}


async def spent(h: Harness, run_id: str) -> dict[str, Any]:
    """What the harness accounted to the run (its own replies, not its children's)."""
    record = await h.runs.get(run_id, tenant="default")
    mine = [a for a in ACCOUNTED if a["run"] == run_id]
    keys = ("input_tokens", "output_tokens", "cached", "cost", "calls")
    return {
        "status": record.status.value if record else None,
        "accounted": len(mine),
        **{
            k: round(sum(float(a.get(k) or (1 if k == "calls" else 0)) for a in mine), 6)
            for k in keys
        },
    }


async def children(h: Harness, run_id: str) -> list[Any]:
    found = [r async for r in h.runs.iterate(parent_run_id=run_id, tenant="default")]
    return [await h.runs.get(r.run_id, tenant="default") for r in found]


def provider_usage() -> dict[str, Any]:
    """Every reply's usage as the API returned it, from the CLI's own transcripts (one entry a
    message id: a reply of several blocks repeats its usage), summed and priced."""
    replies: dict[str, dict[str, Any]] = {}
    for path in HOME.rglob("projects/*/*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            msg = row.get("message") or {}
            if row.get("type") == "assistant" and msg.get("id") and msg.get("usage"):
                replies[msg["id"]] = msg["usage"]
    keys = (
        "input_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "output_tokens",
    )
    total = {k: sum(int(u.get(k) or 0) for u in replies.values()) for k in keys}
    usd = (
        total["input_tokens"] * PRICE["input"]
        + total["cache_read_input_tokens"] * PRICE["cached"]
        + total["cache_creation_input_tokens"] * PRICE["write"]
        + total["output_tokens"] * PRICE["output"]
    ) / 1e6
    return {"replies": len(replies), **total, "usd_at_list_price": round(usd, 6)}


def text(result: Any) -> str:
    return str(result.answer or "")[:600]


# --------------------------------------------------------------------------- probes
async def approved(h: Harness) -> dict[str, Any]:
    """A governed write: asked once, approved, run once."""
    done: list[str] = []

    @tool(side_effects="irreversible")
    def close_ticket(ticket: str) -> str:
        """Close a support ticket by its id."""
        done.append(ticket)
        return f"ticket {ticket} is closed"

    opts = options(
        system_prompt="You close support tickets with close_ticket. Be brief.",
        tools=[],
        max_turns=4,
    )
    setattr(opts, TOOLS_ATTRIBUTE, [close_ticket])
    agent = h.wrap(opts, id=f"probe-approve-{uuid.uuid4().hex[:6]}")
    first = await agent.run("Close ticket T-1.", user="probe")
    out: dict[str, Any] = {"first": first.status.value, "ran_before_approval": list(done)}
    if first.interrupt is None:
        return out | {
            "note": "no pause: the model did not call close_ticket",
            "answer": text(first),
        }
    out["asked_about"] = first.interrupt.tool_call.tool if first.interrupt.tool_call else None
    done_ = await agent.resume(first.interrupt.interrupt_id, "approve", reviewer="lead")
    out |= {
        "after": done_.status.value,
        "ran": list(done),
        "answer": text(done_),
        "asked_again": done_.interrupt is not None,
    }
    out["spent"] = await spent(h, first.run_id)
    return out


async def rejected(h: Harness) -> dict[str, Any]:
    """A rejected call never runs."""
    done: list[str] = []

    @tool(side_effects="irreversible")
    def refund(order: str) -> str:
        """Refund an order by its id."""
        done.append(order)
        return f"refunded {order}"

    opts = options(system_prompt="You refund orders with refund. Be brief.", tools=[], max_turns=4)
    setattr(opts, TOOLS_ATTRIBUTE, [refund])
    agent = h.wrap(opts, id=f"probe-reject-{uuid.uuid4().hex[:6]}")
    first = await agent.run("Refund order O-9.", user="probe")
    out: dict[str, Any] = {"first": first.status.value}
    if first.interrupt is None:
        return out | {"note": "no pause", "answer": text(first)}
    after = await agent.resume(first.interrupt.interrupt_id, "reject", reviewer="lead")
    out |= {
        "after": after.status.value,
        "ran": list(done),
        "answer": text(after),
        "asked_again": after.interrupt is not None,
    }
    if after.interrupt is not None:  # the model asked again: reject again, once
        again = await agent.resume(after.interrupt.interrupt_id, "reject", reviewer="lead")
        out |= {"again": again.status.value, "ran": list(done), "answer": text(again)}
    out["spent"] = await spent(h, first.run_id)
    return out


async def background(h: Harness) -> dict[str, Any]:
    """A background subagent's outcome reaches the next turn; the live session serves the
    second turn; the team's hooks run between the turns."""
    seen: list[dict[str, Any]] = []

    @tool(side_effects="read")
    def market_note(topic: str) -> str:
        """The research desk's note on a market topic."""
        return f"Note on {topic}: demand up 14% this quarter; reference code MKT-7731."

    opts = options(
        system_prompt="You coordinate research. Delegate lookups to the researcher subagent.",
        tools=["Agent"],
        agents={
            "researcher": AgentDefinition(
                description="Looks up market notes with the market_note tool and reports them.",
                prompt="Look up what you are asked with the market_note tool and report the "
                "note word for word.",
                model="inherit",
            )
        },
        max_turns=6,
        hooks=hooks(seen),
    )
    setattr(opts, TOOLS_ATTRIBUTE, [market_note])
    agent = h.wrap(opts, id=f"probe-bg-{uuid.uuid4().hex[:6]}")
    thread = f"t-{uuid.uuid4().hex[:6]}"
    first = await agent.run(
        "Start the researcher subagent in the background (run_in_background: true) to look up "
        "the market note on 'tents'. Do not wait for it: reply only 'started'.",
        user="probe",
        thread=thread,
    )
    out: dict[str, Any] = {"first": first.status.value, "first_answer": text(first)}
    live = list(agent.adapter.live)  # type: ignore[attr-defined]
    out["live_sessions_after_turn_1"] = len(live)
    kids = await children(h, first.run_id)
    out["child_runs"] = [(k.status.value, str(k.output or "")[:200]) for k in kids]
    in_turn = len(seen)
    # the child run ends when the task does: wait for it (two minutes at most)
    for _ in range(120):
        kids = await children(h, first.run_id)
        if kids and all(k.status is not RunStatus.RUNNING for k in kids):
            break
        await asyncio.sleep(1)
    out["child_runs_ended"] = [(k.status.value, str(k.output or "")[:300]) for k in kids]
    await asyncio.sleep(5)  # the session's own turn for the finished task
    out["hooks_in_turn_1"] = seen[:in_turn]
    out["hooks_between"] = seen[in_turn:]
    between = len(seen)
    second = await agent.run(
        "What did the researcher find? Give its reference code.", user="probe", thread=thread
    )
    out |= {
        "second": second.status.value,
        "second_answer": text(second),
        "code_reached_turn_2": "MKT-7731" in text(second),
        "hooks_in_turn_2": seen[between:],
    }
    after = list(agent.adapter.live)  # type: ignore[attr-defined]
    out["same_live_session"] = bool(live) and bool(after) and live[0] is after[0]
    out["spent_1"] = await spent(h, first.run_id)
    out["spent_2"] = await spent(h, second.run_id)
    out["spent_children"] = [await spent(h, k.run_id) for k in kids]
    return out


async def session(h: Harness) -> dict[str, Any]:
    """A live session's second turn: the same process, what turn 1 said remembered."""
    opts = options(system_prompt="You are a terse assistant.", tools=[], max_turns=2)
    agent = h.wrap(opts, id=f"probe-live-{uuid.uuid4().hex[:6]}")
    thread = f"t-{uuid.uuid4().hex[:6]}"
    one = await agent.run("My code word is PLUM-42. Reply 'noted'.", user="probe", thread=thread)
    live = list(agent.adapter.live)  # type: ignore[attr-defined]
    two = await agent.run("What is my code word?", user="probe", thread=thread)
    after = list(agent.adapter.live)  # type: ignore[attr-defined]
    s1, s2 = await spent(h, one.run_id), await spent(h, two.run_id)
    return {
        "first": one.status.value,
        "second": two.status.value,
        "answer": text(two),
        "remembered": "PLUM-42" in text(two),
        "same_process": bool(live) and bool(after) and live[0] is after[0],
        "spent_1": s1,
        "spent_2": s2,
    }


async def limit(h: Harness) -> dict[str, Any]:
    """At the turn limit the run ends PARTIAL, no query more."""
    seen: list[dict[str, Any]] = []
    pages: list[int] = []

    @tool(side_effects="read")
    def read_page(page: int) -> str:
        """Read one page of the report (pages 1 to 6)."""
        pages.append(page)
        return f"Page {page}: revenue line {page * 111}. Read page {page + 1} next."

    opts = options(
        system_prompt="You read reports page by page with read_page, one call at a "
        "time, never several in one reply.",
        tools=[],
        max_turns=3,
        hooks=hooks(seen),
    )
    setattr(opts, TOOLS_ATTRIBUTE, [read_page])
    agent = h.wrap(opts, id=f"probe-limit-{uuid.uuid4().hex[:6]}")
    thread = f"t-{uuid.uuid4().hex[:6]}"
    res = await agent.run(
        "Read all six pages of the report, one at a time, then add up the revenue lines.",
        user="probe",
        thread=thread,
    )
    out = {
        "status": res.status.value,
        "error": res.error.code if res.error else None,
        "error_message": (res.error.message[:300] if res.error else None),
        "answer": text(res),
        "pages_read": list(pages),
        "prompts_sent": sum(1 for s in seen if s["event"] == "UserPromptSubmit"),
        "stops": sum(1 for s in seen if s["event"] == "Stop"),
        "spent": await spent(h, res.run_id),
    }
    nxt = await agent.run("How many pages did you read?", user="probe", thread=thread)
    out |= {
        "next_turn": nxt.status.value,
        "next_answer": text(nxt),
        "pages_read_in_next_turn": pages[len(out["pages_read"]) :],
    }
    return out


PROBES = {
    "approved": approved,
    "rejected": rejected,
    "session": session,
    "background": background,
    "limit": limit,
}


async def main() -> None:
    config = Settings().model_copy(
        update={"runs_url": None, "memory_url": None, "otlp_endpoint": None}
    )
    async with Harness(config=config, memory=False, gateway=False) as h:
        for name, probe in PROBES.items():
            if CHOSEN and name not in CHOSEN:
                continue
            started, wall = time.monotonic(), time.time()
            try:
                found = await asyncio.wait_for(probe(h), timeout=400)
            except Exception as exc:
                found = {"raised": f"{type(exc).__name__}: {exc}"[:800]}
            found["seconds"] = round(time.monotonic() - started, 1)
            found["wall"] = [round(wall, 3), round(time.time(), 3)]
            (OUT / f"{name}.json").write_text(json.dumps(found, indent=2, default=str))
            sys.stdout.write(f"{name} {json.dumps(found, default=str)[:400]}\n")
    usage = provider_usage()
    keys = ("input_tokens", "output_tokens", "cached", "cost")
    usage["harness_accounted"] = {
        k: round(sum(float(a.get(k) or 0) for a in ACCOUNTED), 6) for k in keys
    } | {"replies_or_results": len(ACCOUNTED)}
    (OUT / "provider_usage.json").write_text(json.dumps(usage, indent=2))
    (OUT / "accounted.json").write_text(json.dumps(ACCOUNTED, indent=2, default=str))
    sys.stdout.write(f"provider usage {usage}\n")


asyncio.run(main())
