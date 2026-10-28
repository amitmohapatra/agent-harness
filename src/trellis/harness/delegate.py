"""``delegate``: a ``ReAct``'s helpers — independent jobs handed out at once, each a child run in
a fresh context, their answers merged for the agent to check (the centralized orchestrator
pattern: of the multi-agent ones, the one that amplifies errors least):

    {"jobs": [{"role": "an analyst", "description": "Apple's 2024 revenue, from the filings"},
              {"subagent_type": "research", "description": "Microsoft's 2024 revenue"}]}

A job takes the fields of Deep Agents' ``task`` tool: ``description`` (the part, with everything
the helper needs: it sees nothing else) and ``subagent_type`` (one of the agent's wrapped
sub-agents, ``ReAct(subagents=[h.wrap(...)])``) or ``role`` (a general helper: the agent
itself, told the role). ``ReAct`` offers the tool by itself (``automatic.Delegation``) only to
a wide task on a large model; a single job is one helper, for context isolation.

* **Helpers only read** — every job's run is marked (``Runtime.delegated``): it is given only
  the tools that read, and a call of one that does more is refused, never asked about
  (``tools.bridge``); a sub-agent that may change anything is refused as a job (call it on its
  own). Writes stay with the agent that delegated, one at a time. A helper does not delegate
  again (:data:`MAX_DEPTH`).
* **At once, adaptively** — each job is one of the run's children (``admission.branch``): it
  waits for a slot under the process's adaptive limit, is on the run's task tree (``delegate``,
  ``helper_queued``/``started``/``finished``), and works at most what is left of the run's
  time, less what the run keeps to merge and reply (:data:`MERGE_SECONDS`). In this process
  (``RUNS_URL`` unset) a job is a child run as a sub-agent's call is (``subagents.child_run``:
  its journal kept in the run's, its spending the run's budget's);
  with agent-runs it is queued there (``concurrency_key`` ``delegate:<run>:<slot>``, so
  agent-runs' per-key limit and the slots agree) for any worker with room, its budget an
  equal share of what the run has left (agent-runs keeps no spending), and polled.
* **Early stop, failures** — a job admitted once the run's budget is spent or its time is up is
  skipped; a job that fails, times out or is refused is reported so in the merge, never failing
  the others; cancelling the run cancels every job's run. A job's question to a person pauses
  the run (in this process), as a sub-agent's does.
* **Merged** — one result: each job's helper, status and answer (or why not), in the jobs'
  order; identical jobs run once. The agent checks them against each other and the task.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import RunStatus, stable_id
from trellis.contracts.errors import is_pause_signal
from trellis.harness import admission, budget, pipeline
from trellis.harness.admission import Branch
from trellis.harness.events import DELEGATED as DELEGATED_EVENT
from trellis.harness.result import Result
from trellis.harness.runs import LocalRuns
from trellis.harness.runtime import DELEGATED, Runtime, current
from trellis.harness.subagents import SubAgent, _side_effects, child_run
from trellis.runs import ConflictError, NotFoundError

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from trellis.harness.agent import Agent

#: The tool's name.
DELEGATE: Final = "delegate"
#: The most jobs one call hands out.
MAX_JOBS: Final = 16
#: How deep delegation goes: the agent's helpers do not delegate again (the agent merges).
MAX_DEPTH: Final = 1
#: What a general helper (a job with a ``role``, or neither field) is told.
HELPER_TASK: Final = (
    "You are {role}: one of several helpers working at once, each on one part of a larger "
    "task. Your part:\n\n{description}\n\nOnly look things up and work them out; change "
    "nothing. Answer with what you found, compactly, and say what you could not find."
)
DEFAULT_ROLE: Final = "a helper"
#: What the agent is told with the merged answers.
CHECK: Final = (
    "These are the helpers' own answers: check them against each other and the task before "
    "you answer, and redo or leave out a part that failed."
)
DESCRIPTION: Final = (
    "Hand independent parts of the task to helpers that work at once, each in a fresh "
    "context, and get their answers together. Use it only when the task has several parts of "
    "the same kind that do not depend on each other (one per item: each company, each "
    "document, each question), each needing a few lookups — never for steps that build on "
    "each other, and never to change anything: helpers only read. Each job: `description`, "
    "the part with everything the helper needs to know (it sees nothing else){choice}. You get "
    "each helper's answer, or why it has none."
)
ROLE_ONLY: Final = ", and `role`, what the helper is (an analyst, a fact checker)"
ROLE_OR_AGENT: Final = (
    ", and either `subagent_type`, the agent that does it ({names}), or `role`, what a "
    "general helper is (an analyst, a fact checker)"
)
#: The most of the run's time it keeps for itself, to merge the answers and reply: a job works
#: at most what is left of the run's time less this (or less half of it, when that is less).
MERGE_SECONDS: Final = 5.0
#: What a job's helper is called when it fails before it runs.
REFUSED: Final = "refused"
SKIPPED: Final = "skipped"


def tool(subagents: Sequence[str] = ()) -> BaseTool:
    """The ``delegate`` tool, its jobs' ``subagent_type`` one of ``subagents`` (the agent's
    wrapped sub-agents, by name): the graph's own machinery, which only reads — governed and
    journaled as every call, not of the agent's tool set."""
    from langchain_core.tools import StructuredTool  # noqa: PLC0415 - the react extra

    from trellis.harness.middleware import MACHINERY  # noqa: PLC0415 - the react extra

    names = sorted(set(subagents))
    job: dict[str, Any] = {
        "description": {
            "type": "string",
            "description": "the part of the task, with everything the helper needs to know",
        },
        "role": {"type": "string", "description": "what a general helper is"},
    }
    if names:
        job["subagent_type"] = {
            "type": "string",
            "enum": names,
            "description": "the agent that does this part",
        }
    choice = ROLE_OR_AGENT.format(names=", ".join(names)) if names else ROLE_ONLY
    schema = {
        "type": "object",
        "properties": {
            "jobs": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_JOBS,
                "items": {
                    "type": "object",
                    "properties": job,
                    "required": ["description"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["jobs"],
        "additionalProperties": False,
    }
    return StructuredTool.from_function(
        coroutine=_delegated,
        name=DELEGATE,
        description=DESCRIPTION.format(choice=choice),
        args_schema=schema,
        metadata={MACHINERY: True, "side_effects": "read"},
        handle_tool_error=True,
    )


async def _delegated(jobs: Any) -> str:
    """The tool's call: the jobs, run (:func:`delegate`), as the agent reads them."""
    from langchain_core.tools import ToolException  # noqa: PLC0415 - the react extra

    runtime = current()
    try:
        if runtime is None:
            raise Unrunnable("delegate runs inside a harness run (agent.run/stream)")
        merged = await delegate(runtime, jobs)
    except Unrunnable as exc:
        raise ToolException(str(exc)) from exc
    return json.dumps(merged, ensure_ascii=False, default=str)


class Unrunnable(Exception):
    """Why a ``delegate`` call runs no job at all."""


@dataclass
class Job:
    """One job as given (``n``: its place, from 1) and what came of it."""

    n: int
    by: str
    description: str
    agent: Agent | None = None
    input: str = ""
    run_id: str = ""
    status: str = ""
    answer: Any = None
    error: str | None = None

    def merged(self) -> dict[str, Any]:
        found: dict[str, Any] = {"job": self.n, "helper": self.by, "status": self.status}
        if self.answer is not None:
            found["answer"] = self.answer
        if self.error is not None:
            found["error"] = self.error
        return found


async def delegate(parent: Runtime, given: Any) -> dict[str, Any]:
    """Run ``given`` (the call's ``jobs``) as ``parent``'s helpers, at once, and merge what
    came of them. A job asking a person pauses the run once the others are done (they answer
    from their records when it resumes)."""
    if parent.delegated >= MAX_DEPTH:
        raise Unrunnable(
            "a helper does not delegate: do this part yourself, the agent that gave it to you "
            "merges the parts"
        )
    if not isinstance(given, list) or not given:
        raise Unrunnable("give at least one job: [{description, role or subagent_type}]")
    if len(given) > MAX_JOBS:
        raise Unrunnable(f"at most {MAX_JOBS} jobs at once: {len(given)} were given")
    jobs = [await _job(parent, n, item) for n, item in enumerate(given, start=1)]
    unique: dict[str, Job] = {}
    for job in jobs:
        if job.agent is not None:
            unique.setdefault(job.run_id, job)
    queued = not isinstance(parent.agent.harness.runs, LocalRuns)
    parent.events.custom(
        DELEGATED_EVENT, jobs=len(jobs), runs=len(unique), limit=admission.ADMISSION.limit
    )
    share = _share(parent, len(unique)) if queued else None
    work = [_run(parent, job, queued=queued, share=share) for job in unique.values()]
    raised = await asyncio.gather(*work, return_exceptions=True)
    paused = next((e for e in raised if isinstance(e, BaseException) and is_pause_signal(e)), None)
    if paused is not None:
        raise paused
    for job, problem in zip(unique.values(), raised, strict=True):
        if isinstance(problem, BaseException):
            job.status, job.error = (
                RunStatus.ERROR.value.lower(),
                f"{type(problem).__name__}: {problem}",
            )
    for job in jobs:
        ran = unique.get(job.run_id) if job.agent is not None else None
        if ran is not None and ran is not job:  # the same job again: run once
            job.status, job.answer, job.error = ran.status, ran.answer, ran.error
    done = [job.merged() for job in jobs]
    answered = sum(job["status"] == RunStatus.SUCCESS.value.lower() for job in done)
    return {"answered": answered, "of": len(done), "jobs": done, "note": CHECK}


async def _job(parent: Runtime, n: int, item: Any) -> Job:
    """The job ``item`` (the ``n``-th), its helper found and its run named — or refused, saying
    why."""
    if not isinstance(item, Mapping):
        return _refused(n, "?", "", "a job is an object: {description, role or subagent_type}")
    description = str(item.get("description") or "").strip()
    kind, role = item.get("subagent_type"), item.get("role")
    by = str(kind or role or DEFAULT_ROLE)
    if not description:
        return _refused(n, by, description, "the job has no description")
    if kind and role:
        return _refused(n, by, description, "give subagent_type or role, not both")
    if kind:
        found = _subagent(parent, str(kind))
        if found is None:
            names = sorted(s.name for s in parent.agent.sources if isinstance(s, SubAgent))
            known = ", ".join(names) or "none"
            return _refused(n, by, description, f"no sub-agent {kind} (the sub-agents: {known})")
        if await _side_effects(found) != "read":
            why = f"the {kind} agent may change things, and a helper only reads: call it alone"
            return _refused(n, by, description, why)
        agent, text = found, description
    else:
        agent, text = parent.agent, HELPER_TASK.format(role=by, description=description)
    place = parent.idempotency_key or parent.run_id
    run_id = stable_id(place, DELEGATE, agent.id, text, prefix="run_")
    return Job(n, by, description, agent=agent, input=text, run_id=run_id)


def _refused(n: int, by: str, description: str, why: str) -> Job:
    return Job(n, by, description, status=REFUSED, error=why)


def _subagent(parent: Runtime, name: str) -> Agent | None:
    """The agent's wrapped sub-agent called ``name``."""
    return next(
        (s.agent for s in parent.agent.sources if isinstance(s, SubAgent) and s.name == name),
        None,
    )


async def _run(parent: Runtime, job: Job, *, queued: bool, share: budget.Budget | None) -> None:
    """One job's run, admitted (``admission.branch``): skipped once the run's budget is spent
    or its time is up; in this process, else queued in agent-runs."""
    assert job.agent is not None
    try:
        async with admission.branch(
            parent, job.run_id, job.agent.id, within=_job_time(parent), job=job.n, helper=job.by
        ) as found:
            stopped = _stopped(parent)
            if stopped is not None:
                job.status, job.error = SKIPPED, stopped
                found.ended(SKIPPED, error=stopped)
                return
            if queued:
                result = await _queued(parent, job, found, share)
            else:
                result = await child_run(
                    job.agent,
                    parent,
                    job.input,
                    user=parent.user,
                    run_id=job.run_id,
                    thread=job.run_id,  # its own conversation, not the run's
                    timeout=_job_time(parent),
                    metadata={DELEGATED: parent.delegated + 1},
                    branch=found,
                )
            job.status = result.status.value.lower()
            job.answer = pipeline.jsonable(result.answer)
            job.error = result.error.message if result.error is not None else None
            if result.status is RunStatus.PAUSED:
                job.error = "it asked a person: a queued helper's question is not the run's"
            found.ended(job.status, error=job.error)
    except TimeoutError:
        job.status, job.error = (
            RunStatus.TIMEOUT.value.lower(),
            "the run's time ran out before it ran",
        )


def _stopped(parent: Runtime) -> str | None:
    """Why no more jobs start: the run's budget is spent, or its time is up."""
    spent = budget.reached(parent)
    if spent is not None:
        return f"not run: the budget is spent ({spent})"
    left = parent.remaining()
    return "not run: the run's time is up" if left is not None and left <= 0 else None


def _job_time(parent: Runtime) -> float | None:
    """The most a job may take: what is left of the run's time, less what the run keeps to
    merge the answers and reply (half of it, at most :data:`MERGE_SECONDS`)."""
    left = parent.remaining()
    return None if left is None else left - min(MERGE_SECONDS, left / 2)


def _share(parent: Runtime, jobs: int) -> budget.Budget | None:
    """A queued job's budget: an equal share of what the run has left (``None``: no cap)."""
    cap = parent.budget
    if cap is None:
        return None
    spent = parent.replay.journal.spent

    def left(limit: float | None, used: str) -> float | None:
        return None if limit is None else max(0.0, limit - spent.get(used, 0)) / jobs

    tokens, calls = left(cap.max_tokens, "tokens"), left(cap.max_tool_calls, "tool_calls")
    return budget.Budget(
        max_tokens=None if tokens is None else int(tokens),
        max_cost=left(cap.max_cost, "cost"),
        max_tool_calls=None if calls is None else int(calls),
    )


async def _queued(parent: Runtime, job: Job, found: Branch, share: budget.Budget | None) -> Result:
    """The job queued in agent-runs for a worker (a re-run finds the same run), and waited for
    at most what is left of the run's time; cut short — out of time, cancelled — its run is
    cancelled too."""
    from trellis.harness.agent import RunHandle  # noqa: PLC0415 - the agent imports this

    agent = job.agent
    assert agent is not None
    start = await agent._start(
        job.input,
        user=parent.user,
        thread=job.run_id,
        tenant=None,
        run_id=job.run_id,
        timeout=_job_time(parent),
        budget=share,
        parent=parent,
        concurrency_key=f"{DELEGATE}:{parent.run_id}:{found.slot}",
        metadata={DELEGATED: parent.delegated + 1},
    )
    runs = agent.harness.runs
    await runs.start(start, queue=True)
    try:
        return await RunHandle(agent, job.run_id, tenant=start.tenant_id).result(
            wait=_job_time(parent)
        )
    except (TimeoutError, asyncio.CancelledError) as exc:
        with contextlib.suppress(ConflictError, NotFoundError):  # it ended meanwhile
            await runs.cancel(job.run_id, reason="the delegating run stopped", tenant=parent.tenant)
        if isinstance(exc, TimeoutError):
            return Result(run_id=job.run_id, status=RunStatus.TIMEOUT)
        raise


__all__ = ["DELEGATE", "MAX_DEPTH", "MAX_JOBS", "delegate", "tool"]
