"""Way 1 and Way 2 together against the same services: one administrator's ``approve_when`` in
the memory service's catalog gates the same tool for a wrapped agent (``h.wrap`` of a plain
function) and for a team's own LangGraph graph that governs it with ``governed()``
(``tests/live/team.py``). Both pauses are in the harness's inbox and in agent-runs' listing;
both are approved and finish in agent-runs; both turns are in the memory service, and both
approvals are the tool call's feedback there; one evaluator scores both runs on their traces in
Langfuse (the suite's local stand-in, which records what reaches Langfuse's API)."""

from __future__ import annotations

import uuid
from typing import Any, Final

import pytest

from tests.live.conftest import live_harness, needs_memory, needs_runs
from tests.live.support import StubLangfuse, eventually, memory_scope
from tests.live.team import Team, graph
from tests.support.planned import PlannedChatModel
from trellis import Runtime, tool
from trellis.contracts import InterruptDecision, InterruptResolution, RunStatus
from trellis.harness import telemetry
from trellis.harness.evals import EvalCase, EvalScore, EvalServices, judge
from trellis.harness.governance import governed
from trellis.harness.journal import content_key

pytestmark = [pytest.mark.live, needs_memory, needs_runs]

#: Two runs, each waiting on both services, and the background writes drained.
TIMEOUT_SECONDS: Final = 180
RULE: Final = "qty > 100"


async def ordered(case: EvalCase) -> EvalScore:
    """A deterministic evaluator both ways share: whether the answer is a purchase order."""
    return EvalScore("ordered", "PO for" in str(case.output))


@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_one_rule_gates_a_wrapped_agent_and_a_plain_graph() -> None:
    suffix = uuid.uuid4().hex[:8]
    user, name, sku = f"live-mixed-{suffix}", f"place_order_{suffix}", f"A-{suffix}"
    args = {"sku": sku, "qty": 500}
    placed: list[str] = []

    @tool(name=name, side_effects="write")
    def place_order(sku: str, qty: int) -> str:
        """Place a purchase order."""
        placed.append("wrapped")
        return f"PO for {qty} x {sku}"

    def place_order_plain(sku: str, qty: int) -> str:
        placed.append("plain")
        return f"PO for {qty} x {sku}"

    async def buyer(input: dict[str, Any], agent: Runtime) -> Any:
        return await agent.tools.call(name, **input)

    team = await Team.open(f"live-mixed-plain-{suffix}")
    with StubLangfuse() as langfuse:
        async with live_harness(
            judges=[ordered], judge_sample=1.0, grounding_sample=0.0, **langfuse.settings()
        ) as h:
            try:
                wrapped = h.wrap(buyer, id=f"live-mixed-wrapped-{suffix}", tools=[place_order])
                # the administrator's one rule, in the tenant's catalog
                catalog = team.scope(user).advanced.tools
                await catalog.put_catalog(
                    [{"name": name, "side_effects": "write", "approve_when": RULE}]
                )

                # Way 1: the wrapped agent's call waits
                w_thread = f"live-mixed-wrapped-{suffix}"
                w_paused = await wrapped.run(args, user=user, thread=w_thread)
                assert w_paused.status is RunStatus.PAUSED and w_paused.interrupt is not None
                # Way 2: the graph's governed call waits on the same rule
                p_thread = f"live-mixed-plain-{suffix}"
                record = await team.start(f"Order 500 x {sku}", user=user, thread=p_thread)
                tools = {
                    name: governed(place_order_plain, team.governance, name=name, on_ask=team.ask)
                }
                model = PlannedChatModel(plan=[(name, args)], final="{last}")
                memory = team.scope(user, p_thread, record.run_id)
                app = graph(memory, model, tools, team.checkpointer)
                p_paused = await team.advance(app, record)
                assert p_paused.status is RunStatus.PAUSED and p_paused.awaiting is not None
                assert placed == []
                question = f"Approve {name}? {RULE}."
                assert w_paused.interrupt.question == p_paused.awaiting.question == question

                # both wait in the harness's inbox, and in agent-runs' listing of each agent
                waiting = {r.run_id for r in await h.inbox()}
                assert {w_paused.run_id, record.run_id} <= waiting
                for agent_id, run_id in (
                    (wrapped.id, w_paused.run_id),
                    (team.agent_id, record.run_id),
                ):
                    listed = team.runs.iterate(status=RunStatus.PAUSED, agent_id=agent_id)
                    assert [r.run_id async for r in listed] == [run_id]

                # both approved: each continues its own way
                w_done = await wrapped.resume(
                    w_paused.interrupt.interrupt_id, "approve", reviewer="live-lee"
                )
                resumed = await team.runs.resume(
                    InterruptResolution(
                        interrupt_id=p_paused.awaiting.interrupt_id,
                        run_id=record.run_id,
                        decision=InterruptDecision.APPROVE,
                        reviewer="live-lee",
                    )
                )
                p_done = await team.advance(app, resumed, resume=True)
                await team.governance.decided(
                    team.asked[0], "approve", reviewer="live-lee", run_id=record.run_id, user=user
                )
                answer = f"PO for 500 x {sku}"
                assert w_done.status is RunStatus.SUCCESS and w_done.answer == answer
                assert p_done.status is RunStatus.SUCCESS and p_done.output == answer
                assert sorted(placed) == ["plain", "wrapped"]

                # both finished in agent-runs
                for run_id in (w_done.run_id, record.run_id):
                    run = await team.runs.get(run_id)
                    assert run is not None and run.status is RunStatus.SUCCESS
                    assert run.output == answer

                # both turns in the memory service, and both approvals the call's feedback
                await h.writes.drain()
                assert h.writes.failed == 0
                w_scope = await memory_scope(h, user=user, agent_id=wrapped.id, thread=w_thread)
                assert [m.content for m in await w_scope.history()][-1] == answer
                p_history = await team.scope(user, p_thread).history()
                assert [m.content for m in p_history][-1] == answer
                target = content_key("call", name, args)

                async def both_learned() -> bool:
                    learned = await w_scope.feedback.list_for("tool_call", target)
                    runs = {f.agent_run_id for f in learned if f.verdict == "approve"}
                    return runs == {w_done.run_id, record.run_id}

                assert await eventually(both_learned)

                # one evaluator scores both runs on their traces: the wrapped agent's online
                # judge by itself, the plain graph's from its own code
                async with EvalServices.from_env(langfuse.environ()) as services:
                    case = EvalCase(input=args, output=p_done.output, run_id=record.run_id)
                    scores, failed = await judge(case, [ordered], services=services)
                assert failed == {} and scores == [EvalScore("ordered", True)]
            finally:
                await team.aclose()
    posted = {(s["name"], s["traceId"], s["value"]) for s in langfuse.posted("/api/public/scores")}
    for run_id in (w_done.run_id, record.run_id):
        assert ("ordered", telemetry.trace_hex(run_id), 1.0) in posted
