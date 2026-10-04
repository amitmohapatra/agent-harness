# Contracts: `trellis.contracts` (Way 2)

`trellis.contracts` (pip `trellis-contracts`) is the vocabulary the blocks share: the records
that cross a process boundary (a run, a pause, its answer, a tool call, a judgement) and the
ports an adapter is written against. It has no runtime and no I/O, so it is never switched on
or off: it is how the blocks agree on what goes in and what comes out.

In Way 1 the harness builds and reads every record; you meet them in what it returns
(`Result.interrupt` is an `Interrupt`, `agent.stream()` yields `RunEvent`s). In Way 2 you build
them once and hand them, as they are, to each block you imported.

## Install

```bash
pip install -e ../agent-contracts    # pip trellis-contracts; imports as trellis.contracts
```

It depends on `pydantic` only. Every block depends on it, so installing a block installs it.

## What each block takes and returns

| Block | Takes | Returns |
|---|---|---|
| `trellis.runs` ([runs.md](runs.md)) | `RunStart` (`start`), `Interrupt` (`pause`), `InterruptResolution` (`resume`), `RunStatus` and `AgentError` (`finish`), `ScheduleSpec` (`schedules.create`) | `RunRecord` (with `checkpoint`, `awaiting`, `last_resolution`), `RunSummary` (an inbox entry; its `awaiting` is the `Interrupt`), `ArtifactRef` (`artifacts.upload`, for `Interrupt.payload_ref`), `Schedule`; a `Worker`'s `Job.record` is a `RunRecord` |
| `trellis.memory` ([memory.md](memory.md)) | `Feedback` as it is (`feedback(record)`); the scope keywords of `AgentExecutionContext.scope_fields()` (`bind(**ctx.scope_fields())`) | the service's own models (`PromptContext`, `Feedback`, ...) |
| `trellis.harness.governance` ([governance.md](governance.md)) | `ToolSpec` (`publish`, and its `side_effects` for `check`) | a `Decision` (governance's own); `decided(...)` sends the `Feedback` built from an `Interrupt` and its `InterruptResolution` |
| `trellis.harness.evals` ([evaluation.md](evaluation.md)) | its own `EvalCase` | its own `EvalScore`, put on the run's trace by `run_id` |
| `trellis.harness.a2a.remote` ([a2a.md](a2a.md)) | a message | the answer; `spec` is a `ToolSpec`, a failure a `ToolError` |

## Why shared types matter

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
  resolution format ([mixing.md](mixing.md)).
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

Which type each group holds, how you meet it in each way, and which to use for which job:
the contracts README's
[Each group, in each way](https://github.com/amitmohapatra/agent-contracts/blob/main/README.md#each-group-in-each-way)
and [Which type do I use?](https://github.com/amitmohapatra/agent-contracts/blob/main/README.md#which-type-do-i-use).
