"""Tool choice against a real model, from each kind of source.

**Functions** — the store-operations set of ``examples/06_scenarios/tool_choice.py``: a broad
dashboard beside specific tools, a two- and a three-part request, a change behind approval, a
conclusion to draw, a figure the context remembers out of date, a tool that fails and one that
hangs, a five-part request, a part no tool answers, a change asked for indirectly, a read
that sounds like a change, an unknown record, a change a remembered rule forbids, a question
about what the assistant can do — among its 11 tools, and again among 51 (40 unrelated ones
added). Then the same kinds of item on two **held-out** sets (``heldout.py``: an IT help desk,
travel expenses), written apart and never used to tune the harness, reported apart. Each item
is scored by ``tool_coverage`` (and, with an expected answer, ``contains`` and ``picks``),
``ReAct`` as it is built.

**MCP through Bifrost** — the local MCP servers (``mcp_fixture.py``, declared in the gateway):
the definitions the model is sent are the servers' own (descriptions and schemas), the same
tool on three servers is three names, and the check of the tool set finds what the servers
left undescribed; a change asked of an MCP tool is scored the same way. **Bifrost Code Mode**
— the three wikis, declared read-only by a catalog of the test's own (governance given to the
harness, no memory service), become the meta-tools: the model is sent them as the harness
describes them, and a question is scored by whether its script ran.

What is asserted is the harness's part (``tests/live/proof.py``): every item that ran is scored
from its trajectory, a change that reached its approval counts as made, a failing or hanging
tool's outcome reaches the model as a failure. How well the model chooses is reported, not
asserted: per set and per item, printed and written to ``build/tool_choice_live.json``
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
from langchain.agents.middleware import AgentMiddleware

from tests.live import heldout
from tests.live.conftest import MODEL, OPS, _virtual_key, live_harness, needs_gateway, settings
from tests.live.proof import MODEL_FAILURES, RUN_SECONDS
from tests.live.support import Sent as Wire
from tests.live.support import gateway_model
from tests.support.arm import arm, with_governance
from tests.support.catalog import FakeCatalog
from trellis import Harness, ReAct, current
from trellis.contracts import ToolSpec
from trellis.harness.clients.bifrost import CODE_MODE_TOOLS
from trellis.harness.evals import EvalReport, tool_coverage
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.tools.lint import lint

scenario = importlib.import_module("examples.06_scenarios.tool_choice")

REPEATS: Final = int(os.environ.get("TRELLIS_LIVE_REPEATS", "").strip() or 1)
#: The tool sets — tools, items, instructions, what it can do — the store's (the reproduction
#: set), the same among 51 tools, and the held-out ones (``heldout.py``), reported apart.
Set = tuple[list[Any], list[dict[str, Any]], str, dict[str, bool]]
SETS: Final[dict[str, Set]] = {
    "reproduction: store, 11 tools": (
        scenario.TOOLS,
        scenario.SCENARIOS,
        scenario.SYSTEM,
        scenario.CAN,
    ),
    "reproduction: store, 51 tools": (
        [*scenario.TOOLS, *scenario.UNRELATED],
        scenario.SCENARIOS,
        scenario.SYSTEM,
        scenario.CAN,
    ),
    **heldout.SETS,
}
REPORT: Final = Path("build/tool_choice_live.json")
MAX_STEPS: Final = 8
#: The tools whose every call fails (an error; a timeout after 2 s).
FAILING: Final = {"supplier_eta", "transfer_status"}
RUNS: Final = max(len(v[1]) for v in SETS.values()) * REPEATS + 10
pytestmark = [pytest.mark.live, needs_gateway]


class Sent(AgentMiddleware):
    """The model calls each run made (the agent's own middleware, before the harness's
    ``ModelHooks``)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: dict[str, int] = {}

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        run = current()
        assert run is not None
        self.calls[run.run_id] = self.calls.get(run.run_id, 0) + 1
        return await handler(request)


def rows_of(
    report: EvalReport, sent: Sent, seconds: float, items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for result, given in zip(report.items, items, strict=True):
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
                **{
                    n: scores[n].value if n in scores else None
                    for n in ("says_cannot", "says_no_record", "respects_rule", "plain_answer")
                },
                "needed": given["metadata"]["needed_tools"],
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


def summarized(rows: list[dict[str, Any]], writes: set[str]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    return {
        "model": MODEL,
        "repeats": REPEATS,
        "mean_coverage": mean([r["coverage"] for r in rows]),
        "full_coverage_items": sum(r["coverage"] == 1.0 for r in rows),
        # a tool the item should not need (a broader one) called
        "avoided_called": sum("called though not needed" in (r["comment"] or "") for r in rows),
        "picks": mean([r["picks"] for r in rows]),
        "contains": mean([r["contains"] for r in rows]),
        **{
            n: mean([r[n] for r in rows])
            for n in ("says_cannot", "says_no_record", "respects_rule", "plain_answer")
        },
        # a change asked for, answered without calling its tool (claimed, or not made)
        "writes_not_called": sum(
            bool(writes & set(r["needed"])) and r["status"] == "success" and r["coverage"] == 0.0
            for r in rows
        ),
        # a read that sounds like a change, answered by calling the write tool
        "writes_called_unasked": sum(
            not (writes & set(r["needed"])) and r["status"] == "interrupted" for r in rows
        ),
        "model_calls": mean([r["model_calls"] for r in rows]),
        "statuses": statuses,
        "items": rows,
    }


def saved(section: str, found: dict[str, Any]) -> None:
    REPORT.parent.mkdir(exist_ok=True)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    report[section] = found
    REPORT.write_text(json.dumps(report, indent=2, default=str))


def harness_part(rows: list[dict[str, Any]], writes: set[str]) -> None:
    """What the harness must have done, whatever the model chose."""
    for row in rows:
        if row["status"] == "error":  # the model's own failures only
            # (ReAct's own stops — a stall, the step limit — say "stopped")
            said = row["error"] or ""
            slow = said.startswith("the run worked past its time limit")  # a CPU model's pace
            assert slow or said.startswith("stopped") or any(f in said for f in MODEL_FAILURES), row
    ran = [r for r in rows if r["status"] in ("success", "interrupted")]
    for row in ran:  # every item that needs tools and ran is scored from its trajectory
        assert (row["coverage"] is not None) == bool(row["needed"]), row
        if row["status"] == "interrupted" and writes & set(row["needed"]):  # it was called
            assert row["coverage"] == 1.0, row
        for tool in FAILING & set(row["needed"]):  # a failing tool's failure was recorded
            if row["coverage"] == 1.0:
                assert f"every call failed: {tool}" in (row["comment"] or ""), row


@pytest.mark.parametrize("toolset", list(SETS))
@pytest.mark.timeout(int(RUNS * RUN_SECONDS))
async def test_tool_coverage_of_function_tools(toolset: str) -> None:
    tools, items, system, can = SETS[toolset]
    writes = {t.spec.name for t in tools if t.spec.side_effects != "read"}
    async with live_harness(memory_url=None, runs_url=None) as h:
        sent = Sent()
        target = ReAct(
            # what the assistant can do: in its instructions
            scenario.told(system, can),
            MODEL,
            max_steps=MAX_STEPS,
            middleware=[sent],
        )
        agent = arm(
            h.wrap(target, id="tool-choice", without={"mcp"}, run_timeout=RUN_SECONDS),
            tools,
        )
        rows: list[dict[str, Any]] = []
        for _ in range(REPEATS):
            began = time.monotonic()
            report = await h.evaluate(
                agent, items, scenario.EVALUATORS, run_name="tool-choice-live", concurrency=1
            )
            rows.extend(rows_of(report, sent, time.monotonic() - began, items))
        summary = summarized(rows, writes)
        saved(toolset, summary)  # kept whatever the checks below say
        harness_part(rows, writes)
    print(
        f"\n{toolset}: coverage {summary['mean_coverage']}, "
        f"full {summary['full_coverage_items']}/{len(summary['items'])}, "
        f"unneeded tool called {summary['avoided_called']}, picks {summary['picks']}, "
        f"contains {summary['contains']}, says cannot {summary['says_cannot']}, "
        f"no record {summary['says_no_record']}, rule kept {summary['respects_rule']}, "
        f"plain capability answer {summary['plain_answer']}, "
        f"writes not called {summary['writes_not_called']}, "
        f"writes called unasked {summary['writes_called_unasked']}, "
        f"model calls {summary['model_calls']}"
    )
    for row in summary["items"]:
        print(f"  {row['status']:<12} {row['coverage']} {row['comment']} <- {row['input'][:60]}")


# --------------------------------------------------------------------------- MCP
@pytest.fixture(scope="module")
def store_key(wikis: list[str], ops: str) -> Iterator[str]:
    """A key allowing every tool of the three wikis and of the ops server."""
    yield from _virtual_key(
        "trellis-live-tool-choice",
        [{"mcp_client_name": c, "tools_to_execute": ["*"]} for c in [*wikis, ops]],
    )


MCP_WRITES: Final = {f"{OPS}-delete_records", f"{OPS}-write_note"}
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


def untitled(schema: Any) -> Any:
    """A JSON Schema without its ``title`` keywords or an empty ``properties``, ``required`` in
    name order: what the gateway normalizes (seen live) — the same schema."""
    if isinstance(schema, dict):
        return {
            k: sorted(v) if k == "required" else untitled(v)
            for k, v in schema.items()
            if k != "title" and not (k == "properties" and v == {})
        }
    return [untitled(v) for v in schema] if isinstance(schema, list) else schema


def definitions(wire: Wire) -> dict[str, dict[str, Any]]:
    """Every tool definition the model was sent, by name."""
    return {
        t["function"]["name"]: t["function"]
        for request in wire.requests
        for t in request.get("tools") or []
    }


@pytest.mark.timeout(int(10 * RUN_SECONDS))
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
            middleware=[sent],
        )
        agent = h.wrap(target, id="tool-choice-mcp", without={"code_mode"})
        report = await h.evaluate(
            agent, MCP_ITEMS, [tool_coverage()], run_name="tool-choice-mcp", concurrency=1
        )
        rows = rows_of(report, sent, 0.0, MCP_ITEMS)
        harness_part(rows, MCP_WRITES)
        sent_defs = definitions(wire)
        # the same tool on three servers is three names; each as its server lists it
        assert own and set(own) <= set(sent_defs), set(own) - set(sent_defs)
        for name, tool in own.items():
            assert sent_defs[name]["description"] == tool.description, name
            # as the server lists it, but for what the gateway normalizes (``untitled``)
            assert untitled(sent_defs[name]["parameters"]) == untitled(tool.input_schema), name
        # the check of the tool set finds what the servers left undescribed
        seen = [
            ToolSpec(name=n, description=d.get("description") or "", input_schema=d["parameters"])
            for n, d in sent_defs.items()
        ]
        codes = {(f.tool, f.code) for f in lint(seen)}
        assert (f"{OPS}-write_note", "undescribed_parameter") in codes
        found["mcp"] = summarized(rows, MCP_WRITES) | {"lint": sorted(map(list, codes))}

        # Bifrost Code Mode: the wikis, read-only by the catalog, behind the meta-tools
        rules = {name: Rule(risk="read") for name in own if not name.startswith(OPS)}
        coded = with_governance(
            Harness(
                config=settings(
                    llm_api_key=store_key, memory_url=None, runs_url=None, api_key=None
                ),
            ),
            Governance(FakeCatalog(rules)),
        )
        async with coded:
            wire, sent = Wire(), Sent()
            target = ReAct(
                "You answer questions about code repositories from their wikis, by code.",
                gateway_model(coded, wire),
                max_steps=6,
                middleware=[sent],
            )
            agent = coded.wrap(target, id="tool-choice-code-mode")
            item = {
                "input": "What does the facebook/react wiki say about hooks?",
                "metadata": {"needed_tools": ["execute_tool_code"]},
            }
            report = await coded.evaluate(
                agent, [item], [tool_coverage()], run_name="tool-choice-code-mode"
            )
            rows = rows_of(report, sent, 0.0, [item])
            harness_part(rows, set())
            sent_defs = definitions(wire)
            for spec in CODE_MODE_TOOLS:  # the meta-tools, as the harness describes them
                assert sent_defs[spec.name]["description"] == spec.description
                assert sent_defs[spec.name]["parameters"] == spec.input_schema
            assert not any(n.startswith(tuple(wikis)) for n in sent_defs)  # behind them
            found["code_mode"] = summarized(rows, set()) | {"offered": sorted(sent_defs)}
    saved("MCP via Bifrost", found)
    for part in ("mcp", "code_mode"):
        print(f"\n{part}: {json.dumps(found[part], default=str)[:2000]}")
