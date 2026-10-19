"""Tool choice against a real model, from each kind of source.

**Functions** — the store-operations set of ``examples/06_scenarios/tool_choice.py``: a broad
dashboard beside specific tools, a two- and a three-part request, a change behind approval, a
conclusion to draw, a figure the context remembers out of date, a tool that fails and one that
hangs — among its 10 tools, and again among 50 (40 unrelated ones added). Each item is scored
by ``tool_coverage`` (and, with an expected answer, ``contains`` and ``picks``), ``ReAct`` as it
is and with ``tool_guidance`` and ``answer_check``.

**MCP through Bifrost** — the local MCP servers (``mcp_fixture.py``, declared in the gateway):
the definitions the model is sent are the servers' own (descriptions and schemas), the same
tool on three servers is three names, and the check of the tool set finds what the servers
left undescribed; a change asked of an MCP tool is scored the same way. (Code Mode needs the
memory service's catalog to call the wikis read-only — ``test_live_code_mode.py`` — so it is
not run here.)

What is asserted is the harness's part (``tests/live/proof.py``): every item that ran is scored
from its trajectory, a change that reached its approval counts as made, the guidance is in every
call that offers tools, the check is asked before every answer when it is on, a failing or
hanging tool's outcome reaches the model as a failure. How well the model chooses is reported,
not asserted: per variant and per item, printed and written to ``build/tool_choice_live.json``
(``TRELLIS_LIVE_REPEATS`` runs each set that many times). Memory and agent-runs are not used:
the model's tool choice alone is measured, and no service's data is touched."""

from __future__ import annotations

import importlib
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import pytest

from tests.live.conftest import MODEL, OPS, _virtual_key, live_harness, needs_gateway
from tests.live.proof import MODEL_FAILURES, RUN_SECONDS
from tests.live.support import Sent as Wire
from tests.live.support import gateway_model
from trellis import Hooks, ModelCall, ReAct, current
from trellis.contracts import ToolSpec
from trellis.harness.evals import EvalReport, tool_coverage
from trellis.harness.middleware import ANSWER_CHECK, TOOL_GUIDANCE
from trellis.harness.tools.lint import lint

scenario = importlib.import_module("examples.06_scenarios.tool_choice")

REPEATS: Final = int(os.environ.get("TRELLIS_LIVE_REPEATS", "").strip() or 1)
VARIANTS: Final[dict[str, dict[str, bool]]] = {
    "baseline": {},
    "guidance": {"tool_guidance": True},
    "guidance+check": {"tool_guidance": True, "answer_check": True},
}
#: The tool sets: the store's 10 tools, and the same among 50 (with the variants run there).
SETS: Final[dict[str, tuple[list[Any], tuple[str, ...]]]] = {
    "10 tools": (scenario.TOOLS, tuple(VARIANTS)),
    "50 tools": ([*scenario.TOOLS, *scenario.UNRELATED], ("baseline", "guidance+check")),
}
#: The first line of the check the model is asked, and of the guidance.
CHECK: Final = ANSWER_CHECK.splitlines()[0]
GUIDANCE: Final = TOOL_GUIDANCE.splitlines()[0]
REPORT: Final = Path("build/tool_choice_live.json")
MAX_STEPS: Final = 8
#: The tools whose every call fails (an error; a timeout after 2 s).
FAILING: Final = {"supplier_eta", "transfer_status"}

RUNS: Final = sum(len(v) for _, v in SETS.values()) * REPEATS * len(scenario.SCENARIOS) + 6
pytestmark = [pytest.mark.live, needs_gateway]


class Sent(Hooks):
    """What each run's model calls were sent: the calls offering tools whose instructions
    lacked the guidance, the runs asked the check, and the model calls per run."""

    def __init__(self) -> None:
        self.unguided: dict[str, int] = {}
        self.checked: set[str] = set()
        self.calls: dict[str, int] = {}

    async def before_model(self, call: ModelCall) -> None:
        run = current()
        assert run is not None
        self.calls[run.run_id] = self.calls.get(run.run_id, 0) + 1
        system = call.system.text if call.system is not None else ""
        if GUIDANCE not in system:
            self.unguided[run.run_id] = self.unguided.get(run.run_id, 0) + 1
        if any(CHECK in str(m.content) for m in call.messages):
            self.checked.add(run.run_id)


def rows_of(report: EvalReport, sent: Sent, seconds: float) -> list[dict[str, Any]]:
    rows = []
    for result in report.items:
        scores = {s.name: s for s in result.scores}
        text = result.input if isinstance(result.input, str) else result.input[-1]["content"]
        rows.append(
            {
                "input": text,
                "status": result.status,
                "error": result.error,
                "coverage": scores["tool_coverage"].value if "tool_coverage" in scores else None,
                "comment": scores["tool_coverage"].comment if "tool_coverage" in scores else None,
                "picks": scores["picks"].value if "picks" in scores else None,
                "contains": scores["contains"].value if "contains" in scores else None,
                "model_calls": sent.calls.get(result.run_id or "", 0),
                "answer": result.output if isinstance(result.output, str) else None,
                "run_id": result.run_id,
                "seconds": round(seconds / max(len(report.items), 1), 1),
            }
        )
    return rows


def mean(values: list[Any]) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    return round(sum(numbers) / len(numbers), 3) if numbers else None


def summarized(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    return {
        "model": MODEL,
        "repeats": REPEATS,
        "mean_coverage": mean([r["coverage"] for r in rows]),
        "full_coverage_items": sum(r["coverage"] == 1.0 for r in rows),
        "broad_tool_used": sum("store_overview" in (r["comment"] or "") for r in rows),
        "picks": mean([r["picks"] for r in rows]),
        "contains": mean([r["contains"] for r in rows]),
        "model_calls": mean([r["model_calls"] for r in rows]),
        "statuses": statuses,
        "items": rows,
    }


def saved(section: str, found: dict[str, Any]) -> None:
    REPORT.parent.mkdir(exist_ok=True)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    report[section] = found
    REPORT.write_text(json.dumps(report, indent=2, default=str))


def harness_part(rows: list[dict[str, Any]], sent: Sent, switches: dict[str, bool]) -> None:
    """What the harness must have done, whatever the model chose."""
    for row in rows:
        if row["status"] == "error":  # the model's own failures only
            assert any(f in (row["error"] or "") for f in MODEL_FAILURES), row
    ran = [r for r in rows if r["status"] in ("success", "interrupted")]
    for row in ran:  # every item that ran is scored from its trajectory
        assert row["coverage"] is not None, row
        if row["status"] == "interrupted":  # only a change asks: it was called
            assert row["coverage"] == 1.0, row
        for tool in FAILING:  # a failing tool's failure reached the model
            if tool in row["input"] and (row["comment"] or "").find(tool) == -1:
                assert row["coverage"] == 0.0 or "every call failed" in (row["comment"] or "")
    ids = {r["run_id"] for r in ran}
    if switches.get("tool_guidance"):
        assert not (ids & set(sent.unguided)), sent.unguided
    if not switches.get("answer_check"):
        assert not sent.checked
    else:
        answered = {r["run_id"] for r in ran if r["status"] == "success"}
        assert answered <= sent.checked, answered - sent.checked


@pytest.mark.parametrize("toolset", list(SETS))
@pytest.mark.timeout(int(RUNS * RUN_SECONDS))
async def test_tool_coverage_of_function_tools_with_and_without_the_switches(
    toolset: str,
) -> None:
    tools, variants = SETS[toolset]
    found: dict[str, Any] = {}
    async with live_harness(memory_url=None, runs_url=None) as h:
        for variant in variants:
            switches = VARIANTS[variant]
            sent = Sent()
            target = ReAct(
                scenario.SYSTEM,
                MODEL,
                max_steps=MAX_STEPS,
                tool_guidance=switches.get("tool_guidance", False),
                answer_check=switches.get("answer_check", False),
            )
            agent = h.wrap(
                target,
                id=f"tool-choice-{variant}",
                tools=tools,
                mcp=[],
                hooks=[sent],
                timeout=RUN_SECONDS,
            )
            rows: list[dict[str, Any]] = []
            for _ in range(REPEATS):
                began = time.monotonic()
                report = await h.evaluate(
                    agent,
                    scenario.SCENARIOS,
                    scenario.EVALUATORS,
                    run_name=f"tool-choice-live-{variant}",
                    concurrency=1,
                )
                rows.extend(rows_of(report, sent, time.monotonic() - began))
            harness_part(rows, sent, switches)
            found[variant] = summarized(rows)
    saved(toolset, found)
    for variant, summary in found.items():
        print(
            f"\n{toolset} / {variant}: coverage {summary['mean_coverage']}, "
            f"full {summary['full_coverage_items']}/{len(summary['items'])}, "
            f"broad tool used {summary['broad_tool_used']}, picks {summary['picks']}, "
            f"contains {summary['contains']}, model calls {summary['model_calls']}"
        )
        for row in summary["items"]:
            print(
                f"  {row['status']:<12} {row['coverage']} {row['comment']} <- {row['input'][:60]}"
            )


# --------------------------------------------------------------------------- MCP
@pytest.fixture(scope="module")
def store_key(wikis: list[str], ops: str) -> Iterator[str]:
    """A key allowing every tool of the three wikis and of the ops server."""
    yield from _virtual_key(
        "trellis-live-tool-choice",
        [{"mcp_client_name": c, "tools_to_execute": ["*"]} for c in [*wikis, ops]],
    )


MCP_ITEMS: Final = [
    {
        "input": "Delete every record of the table audit_tmp.",
        "metadata": {"needed_tools": [f"{OPS}-delete_records"]},
    },
    {
        "input": "Write a note saying: restock store S40.",
        "metadata": {"needed_tools": [f"{OPS}-write_note"]},
    },
]


def definitions(wire: Wire) -> dict[str, dict[str, Any]]:
    """Every tool definition the model was sent, by name."""
    return {
        t["function"]["name"]: t["function"]
        for request in wire.requests
        for t in request.get("tools") or []
    }


@pytest.mark.timeout(int(6 * RUN_SECONDS))
async def test_mcp_tools_reach_the_model_as_their_servers_list_them(
    store_key: str, wikis: list[str]
) -> None:
    from tests.live import mcp_fixture

    own = {
        **{f"{w}-{t.name}": t for w in wikis for t in await mcp_fixture.wiki.list_tools()},
        **{f"{OPS}-{t.name}": t for t in await mcp_fixture.ops.list_tools()},
    }
    found: dict[str, Any] = {}
    async with live_harness(store_key, memory_url=None, runs_url=None) as h:
        wire, sent = Wire(), Sent()
        model = gateway_model(h, wire)
        target = ReAct(
            "You run store operations. Be brief.",
            model,
            max_steps=4,
            tool_guidance=True,
            answer_check=True,
        )
        agent = h.wrap(target, id="tool-choice-mcp", hooks=[sent], without={"code_mode"})
        report = await h.evaluate(
            agent, MCP_ITEMS, [tool_coverage()], run_name="tool-choice-mcp", concurrency=1
        )
        rows = rows_of(report, sent, 0.0)
        harness_part(rows, sent, {"tool_guidance": True, "answer_check": True})
        sent_defs = definitions(wire)
        # the same tool on three servers is three names; each as its server lists it
        assert own and set(own) <= set(sent_defs), set(own) - set(sent_defs)
        for name, tool in own.items():
            assert sent_defs[name]["description"] == tool.description, name
            assert sent_defs[name]["parameters"] == tool.input_schema, name
        # the check of the tool set finds what the servers left undescribed
        seen = [
            ToolSpec(name=n, description=d.get("description") or "", input_schema=d["parameters"])
            for n, d in sent_defs.items()
        ]
        codes = {(f.tool, f.code) for f in lint(seen)}
        assert (f"{OPS}-write_note", "undescribed_parameter") in codes
        found["mcp"] = summarized(rows) | {"lint": sorted(map(list, codes))}

    saved("MCP via Bifrost", found)
    print(f"\nMCP: {json.dumps(found['mcp'], default=str)[:2000]}")
