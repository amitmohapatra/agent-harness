# Mixing both ways

The two ways are not two platforms. A wrapped agent (Way 1) and your own framework's agent
using the blocks (Way 2) talk to the same services, through the same SDKs, with the same
records, so one deployment can run both: a team that wraps its new agents keeps its existing
LangGraph graph as it is, and both share one inbox, one tool catalog, one memory and one place
for scores.

```mermaid
flowchart LR
  subgraph w1["Way 1: wrapped"]
    wrapped["h.wrap(graph)<br/>agent.run · start · resume"]
  end
  subgraph w2["Way 2: your framework"]
    own["your graph · Runner.run · query()<br/>+ RunsClient · MemoryClient<br/>+ Governance · judge"]
  end
  wrapped --> runs["agent-runs<br/>one inbox, one queue, webhooks"]
  own --> runs
  wrapped --> memory["memory service<br/>one user's memory, one tool catalog"]
  own --> memory
  wrapped --> lf["Langfuse<br/>one trace per run, scores"]
  own --> lf
  own -- "remote(url)" --> served["agent.serve_a2a(app, url)"]
  wrapped --- served
```

## What is shared

| | Shared how |
|---|---|
| runs and the inbox | both start runs in the same agent-runs: `h.inbox()` and `runs.iterate(status=PAUSED, assignee=...)` list paused runs of both kinds; one webhook subscription hears about both, and one receiver checks both with `verify_signature` |
| governance | one catalog per tenant: an `approve_when` an administrator sets on `create_po` governs the wrapped agent's calls and your `governed(create_po, ...)` alike; approvals from both teach the same approval suggestions |
| memory | one user's profile, memories and documents: what a wrapped agent learned about `ada`, your graph's `scope.context(...)` pushes, and the other way round |
| evaluation | the same evaluators, judge configuration and sampling: a run either way is scored on its trace in Langfuse, and the same `sample` rate picks the same runs |
| the vocabulary | the contracts records ([below](#the-records-they-share)): a pause is an `Interrupt` and its answer an `InterruptResolution` in both |

## Answering each run its own way

The inbox shows both kinds; answer each the way its run continues:

* **A wrapped agent's run**: `await agent.resume(interrupt_id, decision, reviewer=...)` (or the
  AG-UI and A2A surfaces). It continues the run (in process, or back to the queue for a harness
  worker) and sends the decision to memory as feedback. A queued wrapped run may also be
  answered with `RunsClient.resume` from a UI that is not the harness: a harness worker
  continues it, but nothing is sent to memory.
* **Your own run**: `RunsClient.resume(InterruptResolution(...))`, then your code (or your
  `Worker` handler) continues the framework from `record.checkpoint` and
  `record.last_resolution`, and calls `gov.decided(...)` for memory.

`RunSummary.agent_id` says whose run it is.

## Workers

Harness workers (`h.worker([...])`, `python -m trellis.harness.worker`) and your own
`trellis.runs.Worker`s claim from the same queue by agent id. A wrapped agent's run is
continued by `agent.execute(job)`, so your own worker runs it too: hand it the agent's id and
`agent.execute` as the handler (`Worker(runs, agent.execute, [agent.id])`), or call
`agent.execute(job)` for the wrapped agents' jobs from a handler that also runs your own. What
it must not do is run a wrapped agent's run with your own handler, or yours with
`agent.execute` (it refuses another agent's job).

## Calling across

* **Your code calls a wrapped agent**: serve it with `agent.serve_a2a(app, url)` and call it
  with `remote(url, tenant=, user=)` ([a2a.md](a2a.md)). Its questions come back as
  `InputRequired`, or go to your `on_input`.
* **A wrapped agent calls your agent**: only through a server your agent has. Either serve it
  with the A2A SDK yourself, or wrap the function that calls it (any async function is a target,
  [frameworks/functions.md](../frameworks/functions.md)) and serve that with `serve_a2a`; the
  wrapped agent then calls it with `a2a(url)` in its tools.

## Do not gate twice

Inside a wrapped agent, the harness already governs, records and pauses every harness tool
call. Do not also wrap those tools with `governed`, record them with `record_tool`, or pause the
run in agent-runs yourself: each call would be approved twice and recorded twice. Inside a
wrapped run, `trellis.current().memory` is the memory SDK already bound to the run, for the
reads and writes your own code needs.

## Moving between the ways

From Way 2 to Way 1: delete what the harness now does (the `governed` and `recorded` wrappers,
the `RunsClient` calls, the context push, `judge`), declare each tool's side effects with
`tool(fn, side_effects=...)`, build the agent's tools with `h.tools(...)`, and call
`h.wrap(graph).run(...)` where you called the graph. Its runs, memory and catalog are the same
services, so nothing is migrated. From Way 1 to Way 2, the recipes show what to add:
[LangGraph](langgraph.md), [OpenAI Agents SDK](openai-agents.md),
[Claude Agent SDK](claude-agent-sdk.md).

## The records they share

`trellis.contracts` (pip `trellis-contracts`) is the vocabulary the blocks share: the records
that cross a process boundary (a run, a pause, its answer, a tool call, a judgement) and the
ports an adapter is written against. It has no runtime and no I/O, so it is never switched on
or off. In Way 1 the harness builds and reads every record; you meet them in what it returns
(`Result.interrupt` is an `Interrupt`, `agent.stream()` yields `RunEvent`s). In Way 2 you build
them once and hand them, as they are, to each block you imported. Every block depends on it, so
installing a block installs it.

| Block | Takes | Returns |
|---|---|---|
| `trellis.runs` ([runs.md](runs.md)) | `RunStart` (`start`), `Interrupt` (`pause`), `InterruptResolution` (`resume`), `RunStatus` and `AgentError` (`finish`), `ScheduleSpec` (`schedules.create`) | `RunRecord` (with `checkpoint`, `awaiting`, `last_resolution`), `RunSummary` (an inbox entry; its `awaiting` is the `Interrupt`), `ArtifactRef` (`artifacts.upload`, for `Interrupt.payload_ref`), `Schedule`; a `Worker`'s `Job.record` is a `RunRecord` |
| `trellis.memory` ([memory.md](memory.md)) | `Feedback` as it is (`feedback(record)`); the scope keywords of `AgentExecutionContext.scope_fields()` (`bind(**ctx.scope_fields())`) | the service's own models (`PromptContext`, `Feedback`, ...) |
| `trellis.harness.governance` ([governance.md](governance.md)) | `ToolSpec` (`publish`, and its `side_effects` for `check`) | a `Decision` (governance's own); `decided(...)` sends the `Feedback` built from an `Interrupt` and its `InterruptResolution` |
| `trellis.harness.evals` ([evaluation.md](evaluation.md)) | its own `EvalCase` | its own `EvalScore`, put on the run's trace by `run_id` |
| `trellis.harness.a2a.remote` ([a2a.md](a2a.md)) | a message | the answer; `spec` is a `ToolSpec`, a failure a `ToolError` |

One record means the same thing to every block because it is defined once, in this package:

* **One pause, three places.** The `Interrupt` your code pauses a run with in agent-runs is the
  inbox entry a reviewer reads (`RunSummary.awaiting`), and with its `InterruptResolution` it
  becomes the `Feedback` the memory service learns approval rules from
  (`resolution.to_feedback(interrupt, ctx)`, which `Governance.decided` builds for you). No
  translation between them.
* **One identity.** An `AgentExecutionContext` is the run (`RunStart.from_request(
  AgentRequest.create(ctx, input))`), the memory scope (`MemoryClient().bind(
  **ctx.scope_fields())`) and the feedback's attribution, so the three services agree on who,
  for whom and in which thread.
* **Both ways in one store.** A run your LangGraph code started and a run a wrapped agent
  started are the same `RunRecord` in agent-runs: one inbox, one webhook receiver, one
  resolution format ([above](#what-is-shared)).
* **Checked on the wire.** agent-runs' OpenAPI document is built from these models, and the
  harness's and the SDK's tests check every request against it: a record that changes is a new
  contracts version the pins must admit.

```python
from trellis.contracts import (
    AgentExecutionContext,
    AgentRequest,
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    RunStart,
    ToolCall,
)
from trellis.memory import MemoryClient
from trellis.runs import RunsClient

ctx = AgentExecutionContext.create(tenant_id="acme", user_id="ada", agent_id="buyer")
async with RunsClient() as runs, MemoryClient() as memory:
    run = await runs.start(RunStart.from_request(AgentRequest.create(ctx, {"sku": "A-1"})))
    asked = Interrupt(
        tenant_id="acme",
        run_id=run.run_id,
        reason=InterruptReason.APPROVAL,
        question="Order 12 of A-1?",
        tool_call=ToolCall(tool="erp.order", args={"sku": "A-1", "qty": 12}),
    )
    await runs.pause(asked)
    answer = InterruptResolution(
        interrupt_id=asked.interrupt_id, run_id=run.run_id, decision=InterruptDecision.APPROVE
    )
    await runs.resume(answer, tenant="acme")
    await memory.bind(**ctx.scope_fields()).feedback(answer.to_feedback(asked, ctx))
```

Which type each group holds, how you meet it in each way, and which to use for which job: the
contracts README's
[Each group, in each way](https://github.com/amitmohapatra/agent-contracts/blob/main/README.md#each-group-in-each-way)
and [Which type do I use?](https://github.com/amitmohapatra/agent-contracts/blob/main/README.md#which-type-do-i-use).
Runnable: [examples/03_way2_contracts/records.py](../../examples/03_way2_contracts/records.py).
