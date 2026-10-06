# Recipe: LangGraph with the blocks (Way 2)

A LangGraph graph as your team writes it, not wrapped, with the blocks plugged in around it by
your own code:

| Block | What it does here |
|---|---|
| memory ([memory.md](memory.md)) | the context for the question goes in as a leading `SystemMessage`; every tool call, the turn and the outcome are recorded |
| governance ([governance.md](governance.md)) | `governed` checks every call of the graph's tools; an irreversible call asks through LangGraph's own `interrupt`, and the graph's checkpointer keeps the pause |
| runs ([runs.md](runs.md)) | the run is recorded in agent-runs; the pause waits in `role:procurement`'s inbox; the reviewer's `InterruptResolution` is what `Command(resume=...)` carries back into the graph |
| evaluation ([evaluation.md](evaluation.md)) | the answer is judged on the run's trace, and graded against the memory context it was given |

The runnable version, offline and deterministic, is
[`examples/03_way2_governance/langgraph_recipe.py`](../../examples/03_way2_governance/langgraph_recipe.py) (with no harness at all, only
the SDKs: [`examples/04_no_harness/langgraph_sdks.py`](../../examples/04_no_harness/langgraph_sdks.py),
[`examples/04_no_harness/deepagents_sdks.py`](../../examples/04_no_harness/deepagents_sdks.py)); the snippets below are
from it. Wrapping the same graph instead (Way 1, no code for any of this):
[frameworks/langgraph.md](../frameworks/langgraph.md).

## Install

```bash
pip install -e '../agent-harness[langgraph]'   # trellis.harness.governance and .evals, LangGraph
pip install -e ../agent-runs/sdk/python         # trellis.runs (the harness brings it too)
```

`trellis.memory` and `trellis.contracts` come with them. Set `MEMORY_URL`, `RUNS_URL` and
`TRELLIS_API_KEY` for the services, and `BIFROST_URL` with `TRELLIS_JUDGE_MODEL` for the judge.

## The tools: governed and recorded

Your tools stay plain functions. Two small wrappers are added where the tools are built:

```python
from trellis.harness.governance import Decision, Governance, governed
from trellis.memory import current_context


async def create_po(sku: str, qty: int) -> str:
    """Order more units of a SKU from the supplier."""
    return await erp.order(sku, qty)


def recorded(fn):
    """Record each call in the run's memory scope (the one `async with scope:` entered)."""
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    async def call(*args, **kwargs):
        output = await fn(*args, **kwargs)
        if (scope := current_context()) is not None:
            named = dict(signature.bind(*args, **kwargs).arguments)
            await scope.record_tool(fn.__name__, named, output=output)
        return output

    return call
```

The approval is LangGraph's own `interrupt`: the graph pauses inside the tool node, its
checkpointer keeps where it stopped, and what `Command(resume=...)` sends comes back as the
return value of `interrupt`. Here that is the reviewer's `InterruptResolution` from agent-runs:

```python
from langgraph.config import get_config
from langgraph.types import interrupt
from trellis.contracts import InterruptDecision, InterruptResolution

VERDICTS = {InterruptDecision.APPROVE: "approve", InterruptDecision.EDIT: "edit"}


async def approval(decision: Decision) -> bool | dict:
    asked = {"question": decision.question, "tool": decision.tool, "args": dict(decision.args)}
    resolution = InterruptResolution.model_validate(interrupt(asked))
    verdict = VERDICTS.get(resolution.decision, "reject")
    await gov.decided(  # the memory service learns approval rules from it
        decision,
        verdict,
        reviewer=resolution.reviewer or "unknown",
        run_id=resolution.run_id,
        user=get_config()["configurable"]["user"],
        edited=resolution.payload,
    )
    if verdict == "edit":
        return dict(resolution.payload or {})  # run with the reviewer's arguments
    return verdict == "approve"


gov = Governance.from_env(agent_id="procurement", tenant="acme")
tools = [
    tool(governed(recorded(stock), gov, side_effects="read", on_ask=approval)),
    tool(governed(recorded(create_po), gov, side_effects="irreversible", on_ask=approval)),
]
```

`tool` is `langchain_core.tools.tool`: `governed` and `recorded` keep the function's signature,
so the tool's schema is the one it always had. `governed(..., timeout=20)` also bounds a call
and retries a read after an error that may pass, as a harness tool call is
([reliability.md](../reliability.md)). The graph is built from `tools` exactly as
before (`create_agent(model, tools=tools, checkpointer=saver)`, or a `StateGraph` with
`ToolNode(tools)` and `model.bind_tools(tools)`), with a checkpointer: the pause lives there.

## One run

```python
run = await runs.start(
    RunStart(tenant_id="acme", agent_id="procurement", user_id=user, input=question)
)

scope = memory.bind(tenant_id="acme", user_id=user).agent("procurement", agent_run_id=run.run_id)
pushed = await scope.context(question, window=False)  # the checkpointer keeps the conversation
messages = [SystemMessage(pushed.rendered), ("user", question)]
config = {"configurable": {"thread_id": run.run_id, "user": user}}  # one thread per run

async with scope:  # what `recorded` records in
    state = await graph.ainvoke({"messages": messages}, config)
    while "__interrupt__" in state:  # governance asked: the run waits in agent-runs
        asked = state["__interrupt__"][0].value
        await runs.pause(
            Interrupt(
                tenant_id="acme",
                run_id=run.run_id,
                reason=InterruptReason.APPROVAL,
                question=asked["question"],
                tool_call=ToolCall(tool=asked["tool"], args=asked["args"]),
                assignee="role:procurement",
            )
        )

        ...  # a reviewer answers from the inbox: runs.resume(InterruptResolution(...))

        record = await runs.get(run.run_id, tenant="acme")
        resume = record.last_resolution.model_dump(mode="json")
        state = await graph.ainvoke(Command(resume=resume), config)  # where it stopped

answer = state["messages"][-1].content
await runs.finish(run.run_id, RunStatus.SUCCESS, output=answer, tenant="acme")
await scope.history.add([("USER", question), ("ASSISTANT", answer)])
await scope.feedback("run", run.run_id, "confirm", source="system")
```

The reviewer, any time later and from any process, reads the inbox and answers:

```python
async for waiting in runs.iterate(
    status=RunStatus.PAUSED, assignee="role:procurement", tenant="acme"
):
    answer = InterruptResolution(
        interrupt_id=waiting.awaiting.interrupt_id,
        run_id=waiting.run_id,
        decision=InterruptDecision.APPROVE,  # or REJECT, or EDIT with payload={...}
        reviewer="user:lead",
    )
    await runs.resume(answer, tenant="acme")
```

Then the answer is judged on the run's trace, and graded against the context it was given:

```python
case = EvalCase(
    input=question, output=answer, run_id=run.run_id, bundle_id=pushed.bundle_id, memory=scope
)
scores, failed = await judge(
    case, [grounding(), llm_judge("Says what was ordered.")], services=services
)
```

## Across processes: a worker continues the graph

In production the process that paused is not the one that continues. Start the run queued,
and let a `trellis.runs.Worker` run the graph: the first claim invokes it, a claim after the
answer resumes it from the checkpointer (one shared by every worker, such as
`langgraph-checkpoint-postgres`'s `AsyncPostgresSaver`) with the reviewer's resolution:

```python
from trellis.runs import Job, Worker


async def handle(job: Job) -> None:
    record = job.record
    config = {"configurable": {"thread_id": record.run_id, "user": record.user_id}}
    if record.last_resolution is None:
        state = await graph.ainvoke({"messages": [("user", record.input)]}, config)
    else:
        resume = record.last_resolution.model_dump(mode="json")
        state = await graph.ainvoke(Command(resume=resume), config)
    if "__interrupt__" in state:
        asked = state["__interrupt__"][0].value
        await job.pause(
            Interrupt(
                tenant_id=record.tenant_id,
                run_id=record.run_id,
                reason=InterruptReason.APPROVAL,
                question=asked["question"],
                tool_call=ToolCall(tool=asked["tool"], args=asked["args"]),
                assignee="role:procurement",
            )
        )
        return
    await job.finish(RunStatus.SUCCESS, output=state["messages"][-1].content)


await runs.start(
    RunStart(tenant_id="acme", agent_id="procurement", user_id="ada", input=question), queue=True
)
await Worker(runs, handle, ["procurement"]).serve()
```

A resumed queued run goes back to the queue, so whichever worker claims it continues the
graph. A webhook on `run.paused` tells the reviewer's UI instead of polling ([runs.md](runs.md#webhooks)).

## What you own, and what Way 1 does for you

| | This recipe (Way 2) | Wrapped (Way 1) |
|---|---|---|
| where the pause lives | your checkpointer, plus the `Interrupt` you send agent-runs | the harness: the graph's checkpointer, or a re-run against its journal without one |
| the tools' governance | `governed(...)` on each tool | every harness tool, automatically |
| time limits, retries | `governed(..., timeout=)` on each tool; your model client's timeout; `RunStart.timeout_seconds` (agent-runs ends the run) | `@tool(timeout=)`, `ReAct(model_timeout=)`, `run(timeout=, deadline=)`; reads retried, unknown writes never re-run, after a crash too |
| recording | your `recorded` and `history.add` | the transcript, every tool call and the outcome, queued in the background with a spool |
| memory push | your `SystemMessage` | a leading system message with a fixed id, one per checkpointed thread |
| evaluation | `judge(...)` where you choose | `Harness(judges=[...])` on a sampled share, in the background |
| serving | yours | `serve_chat` (AG-UI) and `serve_a2a` |

To move a graph between the two ways, or to run both in one deployment, see
[mixing.md](mixing.md).
