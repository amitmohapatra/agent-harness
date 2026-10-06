# Evaluation: `trellis.harness.evals` (Way 2)

Evaluation scores what an agent answered, and puts every score on the run's trace in Langfuse.
As a block it evaluates an agent you do not wrap: a LangGraph graph, an OpenAI Agents
`Runner`, a Claude Agent SDK `query`, plain functions. Two calls:

* **offline**, `evaluate(my_agent, dataset, evaluators)`: every item of a dataset run and
  scored, as an item of a Langfuse experiment;
* **online**, `judge(case, judges, services=...)`: one run of yours judged, on its trace.

The evaluators (`grounding`, `exact_match`, `contains`, `llm_judge`, your own), the judge's
model and budget, Langfuse experiments and the setup are in [evaluation.md](../evaluation.md);
they are the same in both ways. This page is the API you call.

## Install and set up

Evaluation ships in the harness distribution (`trellis-harness`, from source:
`pip install -e ../agent-harness`) and needs no `Harness`. What it reaches is an
`EvalServices`, read from the environment:

```python
from trellis.harness.evals import EvalServices

async with EvalServices.from_env() as services:  # Langfuse and the judge: the deployment's
    ...
```

| Variable | |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS` | Langfuse's public API for scores, datasets and dataset runs ([Langfuse setup](../evaluation.md#langfuse-setup)); the endpoint also exports the spans, unless your application installed its own tracer provider |
| `BIFROST_URL`, `TRELLIS_JUDGE_VIRTUAL_KEY` (else `BIFROST_VIRTUAL_KEY`) | the judge's gateway |
| `TRELLIS_JUDGE_MODEL` | the judge's model, unless a judge names its own (`llm_judge(..., model="name")`); with neither, `llm_judge` is a failure that says so (there is no agent model to fall back to) |

An unset variable leaves a service out: no Langfuse means scores are `score` spans only and a
dataset must be given as items; no judge means only `llm_judge` fails.
`EvalServices(langfuse=, judge_gateway=, judge_model=, fallback_model=)` takes the same as
fields; `judge_model` may also be a chat model object (anything with
`async complete(messages, **body)`), which is how the offline examples script it.

## Offline: `evaluate`

```python
from trellis.harness.evals import EvalServices, evaluate, exact_match, llm_judge


async def my_agent(question: str) -> str: ...  # your agent, as it is


async with EvalServices.from_env() as services:
    report = await evaluate(
        my_agent,
        "support-golden",  # a Langfuse dataset, or [{"input": ..., "expected": ...}, ...]
        [exact_match(), llm_judge("Answers the question and cites the policy.")],
        services=services,
        run_name="nightly",
    )
print(report)  # items by status, then each evaluator's mean, count and failures
```

`evaluate(target, dataset, evaluators, *, services=None, user=None, run_name=None,
description=None, metadata=None, concurrency=4, limit=None) -> EvalReport`:

* `target` is any `async (input) -> answer`. Each item is one call, inside a root span of its
  own (`invoke_agent <the function's name>`) in the trace of a run id made for the item,
  carrying the Langfuse experiment attributes; the spans your framework's own OpenTelemetry
  instrumentation makes during the call are its children.
* To be graded for grounding, return `EvalOutput(answer, bundle_id, memory)`: the answer, the
  `bundle_id` of the memory context it was given, and the `MemoryContext` (`trellis.memory`) it
  was built in.
* To be graded on what it did (`called`, `tool_sequence`, an evaluator of yours reading
  `case.trajectory`), return `EvalOutput(answer, trajectory=[(ToolCall, ToolOutcome), ...])`:
  the tool calls it made, in order (contracts types). Returned none, the case's trajectory is
  `None` and those evaluators give no score ([evaluation.md](../evaluation.md#trajectories)).
* An item whose call raises is `error`, never fatal; only `success` items are scored.
* `services` defaults to `EvalServices.from_env()`, opened and closed by the call.

The report, the dataset forms and the Langfuse experiment are described in
[evaluation.md](../evaluation.md#offline-evaluate-and-hevaluate).

## Online: `judge`

```python
from trellis.harness.evals import EvalCase, judge, llm_judge

scores, failed = await judge(
    EvalCase(input=question, output=answer, run_id=run_id),
    [llm_judge("Polite, correct and concise.", name="quality")],
    services=services,
    sample=0.1,
)
```

`judge(case, judges, *, services, sample=None) -> (scores, failed)` scores one `EvalCase` and
returns the `EvalScore`s and each judge that raised, with why (logged, never raised). Each
score goes on the case's trace: `trace_id` (32 hex characters, the trace your own tracing
made), else its run's (`run_id`). A case with neither is scored and kept off any trace.
`sample` (0 to 1) judges only that share of cases, chosen by the run id (else the trace id), so
a run is always or never judged; `None` judges every case. To let the deployment choose, pass
`Settings.from_env().judge_sample` (`TRELLIS_JUDGE_SAMPLE`; `None` when unset).

`judge` runs where it is awaited: run it after the response is sent (your web framework's
background task), so the judge model is never on the request path.

For grounding, give the case what the run was given: `EvalCase(..., bundle_id=pushed.bundle_id,
memory=scope)`, where `pushed = await scope.context(question)` ([memory.md](memory.md)).

## With your framework

A LangGraph graph, offline, graded for grounding against the memory context it was given:

```python
from trellis.harness.evals import EvalOutput, EvalServices, evaluate, grounding, llm_judge
from trellis.memory import MemoryClient

memory = MemoryClient()  # MEMORY_URL and TRELLIS_API_KEY


async def support(question: str) -> EvalOutput:
    scope = memory.bind(user_id="trellis-evaluate").agent("support")
    pushed = await scope.context(question)  # the context your graph is given
    state = await graph.ainvoke({"messages": [("system", pushed.rendered), ("user", question)]})
    return EvalOutput(state["messages"][-1].content, pushed.bundle_id, scope)


async with EvalServices.from_env() as services:
    report = await evaluate(
        support, "support-golden", [grounding(), llm_judge("Cites the policy.")], services=services
    )
```

Online, on the trace your own tracing made for the run:

```python
from opentelemetry import trace

tracer = trace.get_tracer("support")


async def answer(question: str, run_id: str) -> str:
    with tracer.start_as_current_span("support") as span:
        state = await graph.ainvoke({"messages": [("user", question)]})
        trace_id = format(span.get_span_context().trace_id, "032x")
    text = state["messages"][-1].content
    case = EvalCase(input=question, output=text, run_id=run_id, trace_id=trace_id)
    await judge(case, [llm_judge("Polite and correct.")], services=services, sample=0.1)
    return text
```

An OpenAI Agents `Agent` and a Claude Agent SDK `query()` are targets the same way:

```python
from agents import Runner
from claude_agent_sdk import ResultMessage, query


async def concierge(question: str) -> str:
    return (await Runner.run(agent, question)).final_output


async def claude(question: str) -> str:
    async for message in query(prompt=question, options=options):
        if isinstance(message, ResultMessage):
            return message.result or ""
    return ""


report = await evaluate(
    concierge, dataset, [exact_match(), llm_judge("Correct.")], services=services
)
```

Each recipe ends with `judge` on the run it just finished: [LangGraph](langgraph.md),
[OpenAI Agents SDK](openai-agents.md), [Claude Agent SDK](claude-agent-sdk.md).
[`examples/blocks_evaluate.py`](../../examples/blocks_evaluate.py) runs `evaluate` on a plain
function over a dataset and `judge` on one run, with a judge of its own and a judge model.

## With Way 1

`h.evaluate(agent, ...)` is `evaluate` with a wrapped agent as the target: each item runs
through the harness's pipeline (memory, tools, governance) and is graded in its own memory
scope. `Harness(judges=[...])` runs `judge` on a sampled share of every wrapped agent's
successful runs, in the background ([evaluation.md](../evaluation.md)). The evaluators, the
judge configuration and the sampling are the same, so a run your code judges with
`sample=rate` is the same run the harness would judge.
