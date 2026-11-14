"""More harness features end to end against the real memory service and agent-runs:

* HAR-TIME a run's time limit and a stalled model: on every adapter, a model that never answers
  ends the run ``TIMEOUT`` within its limit, and agent-runs records the ending; ``ReAct``'s
  ``call_timeout`` ends a stalled step with a ``TIMEOUT`` error (the stall guard);
* HAR-STEP the step limit: at ``max_steps`` the model is asked once more with no tool, its answer
  is the run's, a ``step_limit`` warning is said, and no call past the limit ran;
* HAR-MCP an MCP server of the agent's own over stdio: its tools governed by their annotations,
  recorded in memory as the harness's tools are, on every adapter that takes them;
* HAR-EVAL ``h.evaluate`` with the grounding evaluator: the memory service verifies the answers
  against the contexts the runs were given (``/v1/verify``);
* BG background work: the parts pending on unmerged branches are strict xfails, listed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from tests.features.conftest import ADAPTERS, eventually, harness, memory_sql
from tests.features.kit import armed, ledger, lookup, record, tool_records
from tests.support.planned import PlannedChat
from trellis import ReAct, mcp, stdio
from trellis.contracts import RunStatus
from trellis.harness.evals import grounding

pytestmark = pytest.mark.timeout(300)
SERVER = Path(__file__).parents[1] / "support" / "mcp_server.py"


@pytest.mark.plan("HAR-TIME", "LR-03", "F09")
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_a_model_that_never_answers_ends_the_run_timeout_within_its_limit(
    adapter: str, tmp_path: Path, uid: str
) -> None:
    from tests.support.adapters import BUILDERS
    from tests.support.arm import arm

    async with harness() as h:
        if adapter == "react_da":
            target, tools = ReAct(system_prompt="You work.", model=PlannedChat([], stall=60)), []
        else:
            target, tools = await BUILDERS["react" if adapter == "react_da" else adapter](
                h, [lookup], tmp_path, [("lookup", {"sku": "A-1"})], stall=60
            )
        agent = arm(h.wrap(target, id=f"stall-{adapter}-{uid}"), tools)
        import time

        started = time.monotonic()
        done = await agent.run("Check A-1.", user=f"ty-{uid}", thread="c", run_timeout=4)
        took = time.monotonic() - started
    assert done.status is RunStatus.TIMEOUT, done
    assert took < 40, took
    assert (await record(done.run_id)).status is RunStatus.TIMEOUT
    assert "lookup:A-1" not in ledger()


@pytest.mark.plan("HAR-TIME")
async def test_the_stall_guard_ends_a_stalled_step(tmp_path: Path, uid: str) -> None:
    async with harness() as h:
        target = ReAct(system_prompt="You work.", model=PlannedChat([], stall=60), call_timeout=2)
        agent = h.wrap(target, id=f"guard-{uid}")
        done = await agent.run("Hello?", user=f"ul-{uid}", thread="c")
    assert done.status in (RunStatus.TIMEOUT, RunStatus.ERROR), done
    assert done.error is not None and done.error.category.value == "TIMEOUT", done.error
    assert (await record(done.run_id)).status is done.status


@pytest.mark.plan("HAR-STEP", "LR-02")
async def test_the_step_limit_ends_with_the_models_answer_and_runs_nothing_past_it(
    tmp_path: Path, uid: str
) -> None:
    plan = [("lookup", {"sku": f"S-{n}"}) for n in range(6)]
    async with harness() as h:
        target = ReAct(system_prompt="You work.", model=PlannedChat(plan), max_steps=3,
                       tools=[lookup])
        agent = h.wrap(target, id=f"steps-{uid}")
        events = [e async for e in agent.stream("Check six.", user=f"vi-{uid}", thread="c")]
    ran = [e for e in ledger() if e.startswith("lookup:S-")]
    assert len(ran) <= 3, ran
    said = str([e.data for e in events])
    assert "step_limit" in said, said[:2000]
    run_id = events[0].run_id
    stored = await record(run_id)
    assert stored.status in (RunStatus.SUCCESS, RunStatus.PARTIAL), stored


@pytest.mark.plan("HAR-MCP", "F18s")
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_an_mcp_servers_tools_are_governed_and_recorded(
    adapter: str, tmp_path: Path, uid: str
) -> None:
    notes = mcp(stdio(sys.executable, str(SERVER)), name="notes")
    plan = [("notes-read_note", {"path": f"/inbox/{uid}"})]
    async with harness() as h:
        agent, _ = await armed(adapter, h, tmp_path, plan, id=f"mcp-{adapter}-{uid}",
                               tools=[notes])
        done = await agent.run("Read my note.", user=f"wy-{uid}", thread="c")
        await h.writes.drain()
    assert done.status is RunStatus.SUCCESS, done  # a read: unasked
    assert f"note at /inbox/{uid}" in str(done.answer), done.answer

    async def recorded() -> list[dict]:
        return tool_records(done.run_id)

    rows = await eventually(recorded, within=60)
    assert [r["tool_name"] for r in rows] == ["notes-read_note"], rows
    catalog = memory_sql("select name from tools where name = 'notes-read_note' limit 1")
    assert catalog, "the server's tool is published to the catalog"


@pytest.mark.plan("HAR-EVAL", "F54", "F55")
async def test_evaluate_verifies_answers_against_their_contexts_in_memory(
    tmp_path: Path, uid: str
) -> None:
    user = f"xu-{uid}"
    async with harness() as h:
        ctx = h.memory.client.bind(tenant_id=await h._tenant(), user_id=user)  # type: ignore[union-attr]
        await ctx.remember("The Rotterdam warehouse opens at 7 am.")

        async def answer(question: str, agent: Any) -> str:
            return "The Rotterdam warehouse opens at 7 am."

        agent = h.wrap(answer, id=f"eval-{uid}")
        report = await h.evaluate(
            agent, [{"input": "When does the Rotterdam warehouse open?"}],
            evaluators=[grounding()], user=user,
        )
    scores = [s for item in report.items for s in item.scores if s.name == "grounding"]
    assert scores and scores[0].value is not None and scores[0].value > 0.5, report


#: Background work pending on unmerged branches: each test fails until its branch is merged
#: (the module or name the branch adds is missing here), then passes and must be replaced by the
#: branch's own end-to-end proof — strict, so the merge cannot go unnoticed.
PENDING = {
    "react-background": ("trellis.harness.background", None),
    "bg-kimi": ("trellis.harness.bridges.acp_live", None),
    "bg-claude-da": ("trellis.harness.adapters.claude", "WORKFLOW"),
}


@pytest.mark.plan("BG", "BG-01", "BG-02", "BG-04", "BG-05", "BG-07", "BG-09", "BG-10", "WF-08")
@pytest.mark.parametrize("branch", list(PENDING))
@pytest.mark.xfail(strict=True, reason="background work pending on an unmerged branch "
                   "(react-background: harness-native tasks, L4; bg-kimi: Kimi work across its "
                   "process exit, L5; bg-claude-da: a Claude workflow as one child run, L6)")
def test_background_work_of_an_unmerged_branch(branch: str) -> None:
    import importlib

    module, name = PENDING[branch]
    loaded = importlib.import_module(module)
    assert name is None or hasattr(loaded, name)
