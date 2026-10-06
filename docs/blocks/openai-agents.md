# Recipe: OpenAI Agents SDK with the blocks (Way 2)

An OpenAI Agents SDK `Agent`, run with `Runner.run` as your team runs it, not wrapped, with
the blocks plugged in by your own code:

| Block | What it does here |
|---|---|
| memory ([memory.md](memory.md)) | the context for the question is a leading `system` input item; every tool call, the turn and the outcome are recorded |
| governance ([governance.md](governance.md)) | each tool's `needs_approval` asks `Governance.check`; an irreversible call interrupts the run: the SDK's own approval pause |
| runs ([runs.md](runs.md)) | the SDK's `RunState` (JSON) is the paused run's checkpoint in agent-runs, so any process can continue it; the pause waits in `role:procurement`'s inbox, and the reviewer's answer approves or rejects the call on the restored state |
| evaluation ([evaluation.md](evaluation.md)) | the answer is judged on the run's trace |

The runnable version, offline and deterministic, is
[`examples/03_way2_governance/openai_agents_recipe.py`](../../examples/03_way2_governance/openai_agents_recipe.py); the snippets
below are from it. Wrapping the same `Agent` instead (Way 1):
[frameworks/openai-agents.md](../frameworks/openai-agents.md).

## Install

```bash
pip install -e '../agent-harness[openai-agents]'   # governance, evals, the OpenAI Agents SDK
pip install -e ../agent-runs/sdk/python             # trellis.runs (the harness brings it too)
```

Set `MEMORY_URL`, `RUNS_URL` and `TRELLIS_API_KEY` for the services, and `BIFROST_URL` with
`TRELLIS_JUDGE_MODEL` for the judge. Your model is your own
(`OpenAIChatCompletionsModel(model=..., openai_client=AsyncOpenAI(base_url=BIFROST_URL))` to go
through Bifrost).

## The tools: governed and recorded

`needs_approval` is the SDK's own gate, and governance decides it. Say what each tool does
(governance reads it, and the catalog overrides it):

```python
from agents import Agent, RunContextWrapper, function_tool
from trellis.harness.governance import Governance

SIDE_EFFECTS = {"stock": "read", "create_po": "irreversible"}
gov = Governance.from_env(agent_id="procurement", tenant="acme")


def checked(tool: str):
    async def needs_approval(ctx: RunContextWrapper, args: dict, call_id: str) -> bool:
        return (await gov.check(tool, args, side_effects=SIDE_EFFECTS[tool])).asks

    return needs_approval


buyer = Agent(
    name="buyer",
    instructions="Keep stock topped up: check it, and order when it is low.",
    model=model,
    tools=[
        function_tool(recorded(stock), needs_approval=checked("stock")),
        function_tool(recorded(create_po), needs_approval=checked("create_po")),
    ],
)
```

`recorded` records each call in the run's memory scope, the same small decorator as in the
[LangGraph recipe](langgraph.md#the-tools-governed-and-recorded); `function_tool` builds the
same schema from the wrapped function.

## One run

```python
run = await runs.start(
    RunStart(tenant_id="acme", agent_id="procurement", user_id=user, input=question)
)

scope = memory.bind(tenant_id="acme", user_id=user).agent("procurement", agent_run_id=run.run_id)
pushed = await scope.context(question)
items = [{"role": "system", "content": pushed.rendered}, {"role": "user", "content": question}]

async with scope:  # what `recorded` records in
    result = await Runner.run(buyer, items)
    while result.interruptions:  # governance asked: the run waits in agent-runs
        item = result.interruptions[0]
        tool, args = item.name, json.loads(item.arguments or "{}")
        decision = await gov.check(tool, args, side_effects=SIDE_EFFECTS[tool])
        asked = Interrupt(
            tenant_id="acme",
            run_id=run.run_id,
            reason=InterruptReason.APPROVAL,
            question=decision.question,
            tool_call=ToolCall(tool=tool, args=args),
            assignee="role:procurement",
        )
        # the SDK's state is the checkpoint: any process can continue the run
        await runs.pause(
            asked, checkpoint={"state": result.to_state().to_json(), "call_id": item.call_id}
        )

        ...  # a reviewer answers from the inbox: runs.resume(InterruptResolution(...))

        record = await runs.get(run.run_id, tenant="acme")
        resolution = record.last_resolution
        approved = resolution.decision is InterruptDecision.APPROVE
        await gov.decided(
            decision,
            "approve" if approved else "reject",
            reviewer=resolution.reviewer or "unknown",
            run_id=run.run_id,
            user=user,
        )
        restored = await RunState.from_json(buyer, record.checkpoint["state"])
        for pending in restored.get_interruptions():
            if pending.call_id == record.checkpoint["call_id"]:
                if approved:
                    restored.approve(pending)
                else:
                    restored.reject(pending, rejection_message=str(resolution.answer or ""))
        result = await Runner.run(buyer, restored)

answer = str(result.final_output)
await runs.finish(run.run_id, RunStatus.SUCCESS, output=answer, tenant="acme")
await scope.history.add([("USER", question), ("ASSISTANT", answer)])
await scope.feedback("run", run.run_id, "confirm", source="system")
```

The reviewer reads the inbox and answers exactly as in the
[LangGraph recipe](langgraph.md#one-run): `runs.iterate(status=RunStatus.PAUSED,
assignee="role:procurement")`, then `runs.resume(InterruptResolution(...))`. A reject's
`answer` is the message the model reads. The SDK's approvals have no edited arguments: an
`EDIT` is answered here as a reject (the model reads why and may call again).

Then the answer is judged:

```python
case = EvalCase(
    input=question, output=answer, run_id=run.run_id, bundle_id=pushed.bundle_id, memory=scope
)
scores, failed = await judge(
    case, [grounding(), llm_judge("Says what was ordered.")], services=services
)
```

## Across processes

`RunState` is JSON, so the checkpoint in agent-runs is all another process needs: a
`trellis.runs.Worker` handler (as in the [LangGraph recipe](langgraph.md#across-processes-a-worker-continues-the-graph))
runs `Runner.run(buyer, record.input)` on the first claim, and on a claim after the answer
restores `RunState.from_json(buyer, record.checkpoint["state"])`, approves or rejects the call
named by `record.checkpoint["call_id"]` from `record.last_resolution`, and runs it again. The
agent object is built the same way in every process. Set `set_tracing_disabled(True)` unless
you want the SDK's own tracing sent to OpenAI.

## What you own, and what Way 1 does for you

| | This recipe (Way 2) | Wrapped (Way 1) |
|---|---|---|
| where the pause lives | the `RunState` JSON you put in the checkpoint | the harness keeps the `RunState` and continues it on `agent.resume` |
| governance | `needs_approval` asking `check` | every harness tool; the SDK's own `needs_approval` is answered with the harness's decisions |
| recording, memory push | yours | automatic; the tools offered are narrowed per turn by the tool hints |
| evaluation, serving | `judge(...)` where you choose; serving is yours | judges in the background; `serve_chat`, `serve_a2a` |

To run both ways in one deployment, see [mixing.md](mixing.md).
