# Evaluation: offline datasets and online judges

Langfuse is the system of record for evaluation: datasets, dataset runs, scores, dashboards and
annotation queues live there. Trellis fills it two ways:

| | Offline | Online |
|---|---|---|
| What | an agent run over a dataset, every answer scored | live runs scored as they happen |
| Way 1, wrapped (`h.wrap`) | `report = await h.evaluate(agent, dataset, evaluators)` | automatic: `Harness(judges=[...])`, and the sampled grounding check |
| Way 2, your own code | `report = await evaluate(my_agent, dataset, evaluators)` | `await judge(case, judges, services=services)` |
| Which runs | every item of the dataset (or the first `limit=`) | a sampled share of successful runs with a text answer (`TRELLIS_JUDGE_SAMPLE`, or `sample=`) |
| When | now: the call returns an `EvalReport` | after the run — in the background writes queue, never on the request path, when wrapped |
| Where the scores go | each run's trace (Langfuse scores, and a `score` span); a Langfuse dataset's runs are linked to the dataset run | each run's trace (or the trace a case names) |

Every evaluation name is imported from `trellis.harness.evals` (the `trellis` package holds the
wrapped API only). `examples/evaluate_offline.py` and `examples/online_judges.py` run both ways
with no services. Against the real memory service, `tests/live/test_live_matrix.py` runs a
wrapped evaluation with grounding, `exact_match` and a scripted judge, and checks the Langfuse
dataset run, scores and experiment attributes.

## Way 1 (wrapped): automatic, and `h.evaluate`

A wrapped agent is evaluated with nothing more than this:

```python
from trellis import Harness
from trellis.harness.evals import exact_match, grounding, llm_judge

# online, automatic: every sampled successful run is judged in the background
async with Harness(judges=[llm_judge("Polite, correct and concise.", name="quality")]) as h:
    agent = h.wrap(graph, id="support")

    # offline: a Langfuse dataset by name, or the items themselves
    report = await h.evaluate(
        agent,
        "support-golden",  # or [{"input": ..., "expected": ...}, ...]
        [grounding(), exact_match(), llm_judge("Answers the question and cites the policy.")],
        run_name="nightly-2026-10-04",
    )
    print(report)  # items by status, then each evaluator's mean, count, failures
```

* **Automatic.** With memory on, a sampled share of successful runs is checked against the
  context it was given (`TRELLIS_GROUNDING_SAMPLE`, [memory.md](memory.md)), and with `judges=`
  a sampled share is judged (`TRELLIS_JUDGE_SAMPLE`, [Online](#online-judges-and-judge) below).
  Nothing to call.
* **`h.evaluate`** runs each item through the normal pipeline — memory, tools, governance and
  approvals — and scores it ([Offline](#offline-evaluate-and-hevaluate) below).
* What it reaches is `h.evals`, an `EvalServices` the harness builds from its settings (sharing
  its gateway and Langfuse client); each agent has its own copy, `agent.evals`, whose judge
  falls back to a `ReAct` target's own model when `TRELLIS_JUDGE_MODEL` is unset.

This works for every target ([framework pages](README.md#which-target)).

## Way 2 (pluggable): from your own code

A team that keeps its own framework — a LangGraph graph, an OpenAI Agents `Runner`, a Claude
Agent SDK `query`, plain functions — and does not wrap its agent, imports evaluation as a block.
It needs three things:

```python
from trellis.harness.evals import EvalCase, EvalServices, evaluate, exact_match, judge, llm_judge

services = EvalServices.from_env()  # Langfuse and the judge: the deployment's, never the code's


async def my_agent(question: str) -> str: ...  # your agent, as it is


# offline: a dataset, every answer scored, each call an item of a Langfuse experiment
report = await evaluate(my_agent, "support-golden", [exact_match()], services=services)

# online: one run of yours, judged on its trace
scores, failed = await judge(
    EvalCase(input=question, output=answer, run_id=run_id),
    [llm_judge("Polite, correct and concise.", name="quality")],
    services=services,
    sample=0.1,
)
await services.aclose()  # or: async with EvalServices.from_env() as services
```

**`EvalServices.from_env(environ=None)`** reads only the environment:

| Variable | |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS` | Langfuse's public API for scores, datasets and dataset runs ([Langfuse setup](#langfuse-setup)); the endpoint also exports the spans, unless the application installed its own tracer provider |
| `BIFROST_URL`, `TRELLIS_JUDGE_VIRTUAL_KEY` (else `BIFROST_VIRTUAL_KEY`) | the judge's gateway |
| `TRELLIS_JUDGE_MODEL` | the judge's model; unset, `llm_judge` is a failure that says so (there is no agent model to fall back to) |

Unset variables leave a service out: no Langfuse means scores are `score` spans only and a
dataset must be given as items; no judge means only `llm_judge` fails. `EvalServices(langfuse,
judge_gateway, judge_model, fallback_model)` is the same as fields; close what `from_env` opened
with `await services.aclose()` or `async with`.

**`evaluate(target, dataset, evaluators, *, services=None, user=None, ...)`** takes any
`async (input) -> answer` as `target` (or a wrapped `Agent`: that is `h.evaluate`). Each item is
one call, inside a root span of its own (`invoke_agent <the function's name>`) in the trace of a
run id made for the item, carrying the Langfuse experiment attributes; spans your framework's
own OpenTelemetry instrumentation makes during the call are its children. `services` defaults to
`EvalServices.from_env()`, opened and closed by the call. To be graded for grounding, return
`EvalOutput(answer, bundle_id, memory)`: the answer, the `bundle_id` of the memory context it was
given, and the `MemoryContext` (`trellis.memory`) it was built in.

**`judge(case, judges, *, services, sample=None)`** scores one `EvalCase` and returns
`(scores, failed)`: the `EvalScore`s, and each judge that raised with why (logged, never raised).
Each score goes on the case's trace: `trace_id` (32 hex characters — the trace your own tracing
made), else its run's (`run_id`). A case with neither is scored and kept off any trace.
`sample` (0 to 1) judges only that share of cases, chosen by the run id (else the trace id) so a
run is always or never judged; `None` judges every case. To let the deployment choose, pass
`Settings.from_env().judge_sample` (`TRELLIS_JUDGE_SAMPLE`; `None` when it is unset). `judge` runs where it is awaited: run it after the response
is sent (your web framework's background task), so the judge model is never on the request path.

### With LangGraph

```python
import os

from trellis.harness.evals import EvalOutput, EvalServices, evaluate, grounding, llm_judge
from trellis.memory import MemoryClient

memory = MemoryClient(os.environ["MEMORY_URL"], api_key=os.environ["TRELLIS_API_KEY"])


async def support(question: str) -> EvalOutput:
    scope = memory.bind(user_id="trellis-evaluate", agent_id="support")
    pushed = await scope.context(question)  # the context your graph is given
    state = await graph.ainvoke({"messages": [("system", pushed.rendered), ("user", question)]})
    return EvalOutput(state["messages"][-1].content, pushed.bundle_id, scope)


async with EvalServices.from_env() as services:
    report = await evaluate(
        support,
        "support-golden",
        [grounding(), llm_judge("Answers the question and cites the policy.")],
        services=services,
        run_name="nightly",
    )
```

On-line, judge each run after it answers, on the trace your tracing made for it:

```python
from opentelemetry import trace

from trellis.harness.evals import EvalCase, EvalServices, judge, llm_judge

tracer = trace.get_tracer("support")
services = EvalServices.from_env()  # once, at start-up


async def answer(question: str, run_id: str) -> str:
    with tracer.start_as_current_span("support") as span:
        state = await graph.ainvoke({"messages": [("user", question)]})
        trace_id = format(span.get_span_context().trace_id, "032x")
    text = state["messages"][-1].content
    case = EvalCase(input=question, output=text, run_id=run_id, trace_id=trace_id)
    await judge(case, [llm_judge("Polite and correct.")], services=services, sample=0.1)
    return text
```

### With the OpenAI Agents SDK

```python
from agents import Agent, Runner

from trellis.harness.evals import evaluate, exact_match, llm_judge

concierge = Agent(name="concierge", instructions="Answer briefly.")


async def ask(question: str) -> str:
    return (await Runner.run(concierge, question)).final_output


report = await evaluate(ask, dataset, [exact_match(), llm_judge("Correct.")], services=services)
```

### With the Claude Agent SDK

```python
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from trellis.harness.evals import contains, evaluate

options = ClaudeAgentOptions(system_prompt="Answer briefly.")


async def ask(question: str) -> str:
    async for message in query(prompt=question, options=options):
        if isinstance(message, ResultMessage):
            return message.result or ""
    return ""


report = await evaluate(ask, dataset, [contains()], services=services)
```

## Evaluators

An evaluator is any async function of an `EvalCase` that returns an `EvalScore`, or `None` when
it has nothing to say about the case:

```python
from trellis.harness.evals import EvalCase, EvalScore


async def cites_policy(case: EvalCase) -> EvalScore | None:
    if not isinstance(case.output, str):
        return None
    return EvalScore("cites_policy", "policy" in case.output.lower())
```

| `EvalCase` field | |
|---|---|
| `input` | what the agent was asked (an item's `input`; online, the run's question) |
| `output` | the agent's answer |
| `expected` | the item's expected answer (`None` online, or when the item has none) |
| `run_id` | the run, whose trace the score goes on |
| `trace_id` | the trace the score goes on instead (32 hex characters), when it is not the run's: one your own tracing made (`judge`) |
| `bundle_id`, `context` | the memory context pushed into the run (memory on) |
| `memory` | the `MemoryContext` the context was built in, where `grounding` verifies (a wrapped run's own scope; a callable's `EvalOutput.memory`) |
| `metadata` | the item's metadata |

`EvalScore(name, value, comment=None)`: `value` is a number from 0 to 1 (a Langfuse `NUMERIC`
score), a bool (`BOOLEAN`: 1 or 0) or a string (`CATEGORICAL`); `comment` is shown with it.
An evaluator that raises is a failure of that evaluator on that item (counted in the report,
logged), never a failed evaluation; online, it is a `warning` event and a log line, never a
failed run.

Built in:

| Evaluator | Score |
|---|---|
| `grounding(name="grounding")` | the share of the answer's claims the run's memory context supports — `grounding_score(memory, answer, bundle_id)`: the memory service's `/v1/verify` with the case's `bundle_id` in its `memory` scope, the same function as the sampled check every wrapped run gets (which, in a run's scope, also records the run's `judge` feedback there). No score without a memory scope, without a pushed context, or for an answer with no checkable claim |
| `exact_match(name="exact_match", case_sensitive=False)` | whether the answer is `expected` (text trimmed, case-blind by default; anything else compared as JSON); no score without `expected` |
| `contains(name="contains", case_sensitive=False)` | whether the answer contains `expected` — each of them, for a list; the comment names what is missing |
| `llm_judge(criteria, *, name="llm_judge")` | a judge model's grade, 0 to 1, against `criteria` written in plain language; its reasoning is the comment |

### `llm_judge`

The judge reads the criteria, the input, the expected answer and the memory context when there
are any, and the answer (each cut at 8000 characters), and is asked at temperature 0 for only
`{"score": <0..1>, "reasoning": "..."}`. Its reply is read robustly (a code fence or prose
around the object is fine); a reply that is still not that object, or a score outside 0..1, is
answered once with what was wrong, and a second bad reply is no score and a warning. It asks
the model of the services `evaluate` or `judge` was given; called on its own, outside them, it
is a `ConfigurationError`.

**Which model, and whose budget, is configuration — never code:**

| Variable | |
|---|---|
| `TRELLIS_JUDGE_MODEL` | the Bifrost model name the judge asks, through `BIFROST_URL` |
| `TRELLIS_JUDGE_VIRTUAL_KEY` | the virtual key the judge's calls go through; unset: `BIFROST_VIRTUAL_KEY` |

Set `TRELLIS_JUDGE_MODEL` to a **different, stronger model than the agent's**: a model grading
its own answers is biased towards them. Give the judge **its own virtual key**: its spend is then
budgeted, rate-limited and reported in Bifrost apart from the agents' traffic. With
`TRELLIS_JUDGE_MODEL` unset, a wrapped agent's judge falls back to its own model (a `ReAct`'s
model, `agent.evals.fallback_model`), and the harness logs once per agent that the judge shares
the agent's model; a judged agent with no model the harness knows (a graph, a function), and
any code judged through `EvalServices.from_env()`, then gets no judge score, with the reason in
the report's `failed` (`"llm_judge needs a model: set TRELLIS_JUDGE_MODEL"`).

**Cost.** Every judged answer is one judge model call (two when the first reply is malformed),
with a prompt of the criteria plus the case — up to about 32 000 characters. Offline that is
one call per item and judge; online it is one per *sampled* run and judge: at the default 10 %
sample, 1 000 runs a day cost about 100 judge calls a day per judge. Sample down
(`TRELLIS_JUDGE_SAMPLE`), judge offline on a dataset instead, or use a cheaper deterministic
evaluator where one will do.

## Offline: `evaluate` and `h.evaluate`

```python
await h.evaluate(agent, dataset, evaluators, *, run_name=None, description=None, metadata=None,
                 concurrency=4, limit=None, user=None)
await evaluate(target, dataset, evaluators, *, services=None, user=None, run_name=None,
               description=None, metadata=None, concurrency=4, limit=None)
```

`h.evaluate(agent, ...)` is `evaluate(agent, ...)` with the agent's own services
(`agent.evals`); `target` is a wrapped `Agent` or any `async (input) -> answer`
([Way 2](#way-2-pluggable-from-your-own-code)).

* **`dataset`** — a Langfuse dataset's name, or a sequence of items: mappings with `input` (and
  `expected`, `metadata`) or `EvalItem(input, expected=None, metadata={}, id=None)`s. A
  Langfuse dataset is read whole first (`GET /api/public/v2/datasets/{name}`, then
  `GET /api/public/dataset-items?datasetName=&page=&limit=50`, page by page; archived items are
  left out). A name with Langfuse not configured, or one Langfuse does not have, is a
  `ConfigurationError`.
* **Each item of a wrapped agent** runs through the normal pipeline — memory push and pull, the
  toolbox, governance and approvals, records, traces — acting for `user` (default
  `trellis-evaluate`; memory is scoped to it, so give an evaluation its own user when its writes
  should stay apart), as an item of a Langfuse experiment (below). Then the evaluators score the
  answer — `grounding` in the run's own memory scope — and each score goes on the run's trace.
* **Each item of a callable** is one call, `await target(input)`, under a run id made
  for it (`run_…`), inside its `invoke_agent <name>` span (the function's name, else its type's;
  `user.id` is `user`), as an item of the experiment. What it returns is the answer — or, as an
  `EvalOutput`, the answer with the memory context `grounding` checks it against.
* **What cannot stop it**: an item whose run fails (or whose call raises) is `error` (with its
  message), one that pauses for a person is `interrupted` — its run is cancelled so it does not
  wait in an inbox — and one whose run is cancelled is `cancelled`; only `success` items are
  scored.
* **`concurrency`** items run at once (at least 1), **`limit`** keeps the first items only,
  **`run_name`** (default `<agent id or function name>-<UTC time>`) names the experiment, and
  `description` and `metadata` describe it (the metadata always holds `agent_id`: the agent's
  id, or the callable's name).
* **At the end** the background writes are drained (a wrapped agent's) and the spans exported,
  so the scores are in Langfuse when the call returns.

`EvalReport`: `run_name`, `dataset` (the Langfuse dataset's name, or `None`), `experiment_id`
and `dataset_run_url` (below), `items` — in
dataset order, each an `EvalResult(input, expected, output, status, run_id, scores, failed,
error, trace_url)` (`trace_url` is `<langfuse host>/trace/<trace id>` when Langfuse is
configured) — `statuses` (how many items ended each way) and `summary`: each evaluator's
`EvaluatorStats(mean, count, failures)` by name — the mean of its numeric and bool scores
(`None` for categorical ones), how many scores it gave, how many items it failed on.
`print(report)` and `print(report.summary)` are tables.

### Each run is an item of a Langfuse experiment (v3, self-hosted, and v4)

The harness does what Langfuse's own SDK experiment runner does (`langfuse-python`,
`_process_experiment_item`), so an evaluation is an experiment on Langfuse v3 — Cloud and
self-hosted — and on v4 alike:

1. **The dataset run link (v3).** For an item of a Langfuse dataset, before the run, its trace is
   linked to the dataset run `run_name` as that item's result:
   `POST /api/public/dataset-run-items {runName, runDescription?, datasetItemId, traceId,
   metadata}` (the dataset run is created by its first item). Langfuse answers with the dataset
   run's id (`datasetRunId`), which becomes the experiment's id and gives
   `report.dataset_run_url` (`<host>/project/<projectId>/datasets/<datasetId>/runs/<id>`). A link
   Langfuse refuses — Langfuse v4 has no such endpoint — is a warning, and nothing else changes.
2. **The experiment attributes (v4 reads these).** Every span of the run — its `invoke_agent`
   span (the item's root observation; a callable's own root span), `chat`, `execute_tool` and
   `retrieve memory` spans, and the `score` spans of its evaluators — carries:

| Attribute | Value |
|---|---|
| `langfuse.experiment.id` | the dataset run's id; else one id made once per `evaluate` call (16 hex characters), the same for all its items — also `report.experiment_id` |
| `langfuse.experiment.name` | `run_name` |
| `langfuse.experiment.metadata.<key>` | the experiment's `metadata`, flattened (`params.temperature`) and each value serialized (text as is, anything else as JSON) |
| `langfuse.experiment.dataset.id` | the Langfuse dataset's id (not for a local list) |
| `langfuse.experiment.item.id` | the dataset item's id (or an `EvalItem`'s `id`); else the first 16 hex characters of the SHA-256 of the serialized input |
| `langfuse.experiment.item.metadata.<key>` | the item's metadata, flattened and serialized the same way |
| `langfuse.experiment.item.root_observation_id` | the run's `invoke_agent` span id (16 hex characters) |
| `langfuse.environment` | `sdk-experiment`, as the SDK marks an experiment's spans |

The root span also carries `langfuse.experiment.description` (`description`, when given) and
`langfuse.experiment.item.expected_output` (the item's `expected`, serialized). As in the SDK, a
propagated value longer than 200 characters is left out. Spans a framework's own
instrumentation creates (a LangChain or OpenAI Agents tracer) do not carry them; the harness's
spans do.

## Online: judges and `judge()`

```python
h = Harness(judges=[llm_judge("Polite, correct and concise.", name="quality"), cites_policy])
```

After a successful run of a wrapped agent with a text answer, if the run falls in the sample,
each judge is queued in the background writes (`judge.<name>`): the run has already returned
when it runs. Each is one `judge(case, [that judge], services=agent.evals)`; the case is the
run's question, answer, memory context and scope, and run id (no `expected`). Its score goes on
the run's trace; a judge that fails is a `warning` event on the run's stream (`judge_failed`)
and a log line.

**`TRELLIS_JUDGE_SAMPLE`** is the share of runs judged, 0 to 1. Unset, it is 0.1 when the harness
has judges (and nothing is judged without judges). The run id decides — salted apart from the
grounding sample (`TRELLIS_GROUNDING_SAMPLE`), so the two samples are independent — so a run is
either always or never judged, whichever process asks. `judge(..., sample=rate)` from your own
code samples the same way: a run the harness would judge, your code judges too.

## Langfuse setup

Nothing beyond the OTLP variables that already send the traces there
([observability.md](observability.md#export)): `OTEL_EXPORTER_OTLP_HEADERS` with
`Authorization=Basic <base64 public-key:secret-key>`, and the endpoint Langfuse's
(`…/api/public/otel`) or the headers naming its host (`x-langfuse-host`, through a collector).
The same credentials reach its public API for scores, datasets and dataset runs — the harness's
and `EvalServices.from_env()`'s alike. Without them, scores are `score` spans on the run's trace
only (every OTLP backend gets those), and a dataset
must be given as items.

In Langfuse, an offline evaluation is a **dataset run** of the dataset: each item's trace and
its scores side by side, comparable with earlier runs of the same dataset. Online judges'
scores are **scores on the traces**: filter traces by score, chart them per agent (the trace
name is the agent id) and over time, and send low scores to an annotation queue.

The API shapes are those of Langfuse's own definitions (`fern/apis/server/definition` in its
repository: `datasets.yml`, `dataset-items.yml`, `dataset-run-items.yml`, `scores.yml`), and the
experiment attributes those of its Python SDK (`langfuse/_client/attributes.py`,
`propagation.py`). Evaluation works on Langfuse v3 (Cloud and self-hosted), through the dataset
run link, and on v4, through the spans' experiment attributes — the harness sends both, as the
SDK does.
