"""Who may answer a paused run, against the real memory service and agent-runs: the
application's key (it may act for anyone) answers any run; a key restricted to one person
answers only that person's runs, as that person, and never a run assigned to a group. The
restricted key is issued for the test with ``TRELLIS_ADMIN_KEY`` (an admin of the tenant of
``TRELLIS_API_KEY``, or the platform or development key) and revoked after it."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Final

import pytest

from tests.live.conftest import MEMORY_URL, RUNS_URL, live_harness, needs_memory, needs_runs
from trellis import Runtime
from trellis.contracts import (
    Interrupt,
    InterruptDecision,
    InterruptResolution,
    RunRecord,
    RunStart,
    RunStatus,
    new_id,
)
from trellis.memory import MemoryClient
from trellis.runs import AuthorizationError, RunsClient

ADMIN_KEY: Final = os.environ.get("TRELLIS_ADMIN_KEY", "").strip() or None
#: Issuing a key, five runs and their answers, against services on one development machine.
TIMEOUT_SECONDS: Final = 120

pytestmark = [
    pytest.mark.live,
    needs_runs,
    needs_memory,
    pytest.mark.skipif(
        ADMIN_KEY is None, reason="needs TRELLIS_ADMIN_KEY to issue a key restricted to a person"
    ),
    pytest.mark.timeout(TIMEOUT_SECONDS),
]


@dataclass(frozen=True)
class People:
    """This test's tenant, its people and agent (unique per run), and a key restricted to
    ``priya`` and the agent: a harness holding it runs the agent's memory calls as both."""

    tenant: str
    priya: str
    raj: str
    finance: str
    agent: str
    priya_key: str


@pytest.fixture
async def people() -> AsyncIterator[People]:
    suffix = uuid.uuid4().hex[:8]
    async with live_harness() as h:
        tenant = await h.tenant()
    priya, agent = f"priya-{suffix}", f"live-answering-{suffix}"
    async with MemoryClient(MEMORY_URL, api_key=ADMIN_KEY) as admin:
        keys = admin.administer(tenant).keys
        issued = await keys.issue("service", agent, may_act_as=[f"user:{priya}", f"agent:{agent}"])
        assert issued.token is not None
        try:
            yield People(
                tenant, priya, f"raj-{suffix}", f"role:finance-{suffix}", agent, issued.token
            )
        finally:
            await keys.revoke(issued.key_id)


async def waiting(runs: RunsClient, tenant: str, assignee: str) -> RunRecord:
    """A run recorded in process, paused on a question for ``assignee``."""
    run = await runs.start(
        RunStart(run_id=new_id("run_"), tenant_id=tenant, agent_id="live-answering", input="x")
    )
    asked = Interrupt(
        tenant_id=tenant, run_id=run.run_id, question="Approve the refund?", assignee=assignee
    )
    return await runs.pause(asked)


def approve(run: RunRecord, reviewer: str) -> InterruptResolution:
    assert run.awaiting is not None
    return InterruptResolution(
        interrupt_id=run.awaiting.interrupt_id,
        run_id=run.run_id,
        decision=InterruptDecision.APPROVE,
        reviewer=reviewer,
    )


async def test_a_restricted_key_answers_only_its_persons_runs(people: People) -> None:
    async with (
        RunsClient(RUNS_URL, api_key=os.environ["TRELLIS_API_KEY"]) as app,
        RunsClient(RUNS_URL, api_key=people.priya_key) as priya,
    ):
        hers = await waiting(app, people.tenant, f"user:{people.priya}")
        also_hers = await waiting(app, people.tenant, f"user:{people.priya}")
        his = await waiting(app, people.tenant, f"user:{people.raj}")
        finance = await waiting(app, people.tenant, people.finance)

        answered = await priya.resume(approve(hers, people.priya))  # a bare id is user:<id>
        assert answered.status is RunStatus.RUNNING
        assert answered.last_resolution is not None
        assert answered.last_resolution.reviewer == people.priya  # kept as given

        refusals = {
            f"the run is assigned to user:{people.raj}, not user:{people.priya}": approve(
                his, people.priya
            ),
            f"the run is assigned to {people.finance}, a group": approve(finance, people.priya),
            f"this key may not act for user:{people.raj}": approve(also_hers, people.raj),
        }
        for detail, answer in refusals.items():
            with pytest.raises(AuthorizationError) as refused:
                await priya.resume(answer)
            assert refused.value.status == 403 and detail in refused.value.message
            still = await app.get(answer.run_id)
            assert still is not None and still.status is RunStatus.PAUSED

        # the application's key vouches for whoever it names, a group's run included
        assert (await app.resume(approve(finance, people.raj))).status is RunStatus.RUNNING
        for run in (hers, also_hers, his, finance):
            await app.finish(run.run_id, RunStatus.CANCELLED, tenant=people.tenant)


async def approval(input: Any, agent: Runtime) -> str:
    approved = await agent.ask("Send the quote?", options=["yes", "no"])
    return f"quote {input}: {approved}"


async def test_a_harness_with_a_persons_key_resumes_that_persons_run(people: People) -> None:
    async with live_harness() as app:
        paused = await app.wrap(approval, id=people.agent).run("Q-7", user=people.priya)
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.assignee == f"user:{people.priya}"  # the run's user, by default

    async with live_harness(api_key=people.priya_key) as hers:
        agent = hers.wrap(approval, id=people.agent)
        interrupt_id = paused.interrupt.interrupt_id
        with pytest.raises(AuthorizationError, match=f"may not act for user:{people.raj}"):
            await agent.resume(interrupt_id, "answer", answer="yes", reviewer=people.raj)
        done = await agent.resume(interrupt_id, "answer", answer="yes", reviewer=people.priya)
    assert done.status is RunStatus.SUCCESS, done.error
    assert done.answer == "quote Q-7: yes"
