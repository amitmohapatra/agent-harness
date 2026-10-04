# Evaluation: offline datasets and online judges

Langfuse is the system of record for evaluation: datasets, dataset runs, scores, dashboards and
annotation queues live there. The harness makes both ways of filling it one call:

| | Offline | Online |
|---|---|---|
| What | an agent run over a dataset, every answer scored | live runs scored as they happen |
| How | `report = await h.evaluate(agent, dataset, evaluators)` | `Harness(judges=[...])` |
| Which runs | every item of the dataset (or the first `limit=`) | a sampled share of successful runs with a text answer (`TRELLIS_JUDGE_SAMPLE`) |
| When | now: the call returns an `EvalReport` | after the run, in the background writes queue — never on the request path |
| Where the scores go | each run's trace (Langfuse scores, and a `score` span); a Langfuse dataset's runs are linked to the dataset run | each run's trace |

```python
from trellis import Harness, contains, exact_match, grounding, llm_judge

async with Harness() as h:
    agent = h.wrap(graph, id="support")

    # offline: a Langfuse dataset by name, or the items themselves
    report = await h.evaluate(
        agent,
        "support-golden",  # or [{"input": ..., "expected": ...}, ...]
        [grounding(), exact_match(), llm_judge("Answers the question and cites the policy.")],
        run_name="nightly-2026-10-04",
    )
    print(report)  # items by status, then each evaluator's mean, count, failures

# online: judges on a sample of live runs
h = Harness(judges=[llm_judge("Polite, correct and concise.", name="quality")])
```

`examples/evaluate_offline.py` and `examples/online_judges.py` run both with no services.

## Evaluators

An evaluator is any async function of an `EvalCase` that returns an `EvalScore`, or `None` when
it has nothing to say about the case:

```python
from trellis import EvalCase, EvalScore


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
| `bundle_id`, `context` | the memory context pushed into the run (memory on) |
| `metadata` | the item's metadata |
| `agent` | the `Agent` that answered (the built-ins reach the harness through it) |

`EvalScore(name, value, comment=None)`: `value` is a number from 0 to 1 (a Langfuse `NUMERIC`
score), a bool (`BOOLEAN`: 1 or 0) or a string (`CATEGORICAL`); `comment` is shown with it.
An evaluator that raises is a failure of that evaluator on that item (counted in the report,
logged), never a failed evaluation; online, it is a `warning` event and a log line, never a
failed run.

Built in:

| Evaluator | Score |
|---|---|
| `grounding(name="grounding")` | the share of the answer's claims the run's memory context supports — the memory service's `/v1/verify` with the run's `bundle_id`, the same check as the sampled one every run gets (which also records the run's `judge` feedback there). No score without memory, without a pushed context, or for an answer with no checkable claim |
| `exact_match(name="exact_match", case_sensitive=False)` | whether the answer is `expected` (text trimmed, case-blind by default; anything else compared as JSON); no score without `expected` |
| `contains(name="contains", case_sensitive=False)` | whether the answer contains `expected` — each of them, for a list; the comment names what is missing |
| `llm_judge(criteria, *, name="llm_judge")` | a judge model's grade, 0 to 1, against `criteria` written in plain language; its reasoning is the comment |

### `llm_judge`

The judge reads the criteria, the input, the expected answer and the memory context when there
are any, and the answer (each cut at 8000 characters), and is asked at temperature 0 for only
`{"score": <0..1>, "reasoning": "..."}`. Its reply is read robustly (a code fence or prose
around the object is fine); a reply that is still not that object, or a score outside 0..1, is
answered once with what was wrong, and a second bad reply is no score and a warning.

**Which model, and whose budget, is configuration — never code:**

| Variable | |
|---|---|
| `TRELLIS_JUDGE_MODEL` | the Bifrost model name the judge asks, through `BIFROST_URL` |
| `TRELLIS_JUDGE_VIRTUAL_KEY` | the virtual key the judge's calls go through; unset: `BIFROST_VIRTUAL_KEY` |

Set `TRELLIS_JUDGE_MODEL` to a **different, stronger model than the agent's**: a model grading
its own answers is biased towards them. Give the judge **its own virtual key**: its spend is then
budgeted, rate-limited and reported in Bifrost apart from the agents' traffic. With
`TRELLIS_JUDGE_MODEL` unset, the judge falls back to the judged agent's own model (a `ReAct`'s
model), and the harness logs once that the judge shares the agent's model; a judged agent with
no model the harness knows (a graph, a function) then gets no judge score, with the reason in
the report's `failed` (`"llm_judge needs a model: set TRELLIS_JUDGE_MODEL"`).

**Cost.** Every judged answer is one judge model call (two when the first reply is malformed),
with a prompt of the criteria plus the case — up to about 32 000 characters. Offline that is
one call per item and judge; online it is one per *sampled* run and judge: at the default 10 %
sample, 1 000 runs a day cost about 100 judge calls a day per judge. Sample down
(`TRELLIS_JUDGE_SAMPLE`), judge offline on a dataset instead, or use a cheaper deterministic
evaluator where one will do.

## Offline: `h.evaluate`

```python
await h.evaluate(agent, dataset, evaluators, *, run_name=None, concurrency=4, limit=None, user=None)
```

* **`dataset`** — a Langfuse dataset's name, or a sequence of items: mappings with `input` (and
  `expected`, `metadata`) or `EvalItem(input, expected=None, metadata={}, id=None)`s. A
  Langfuse dataset is read whole first (`GET /api/public/v2/datasets/{name}`, then
  `GET /api/public/dataset-items?datasetName=&page=&limit=50`, page by page; archived items are
  left out). A name with Langfuse not configured, or one Langfuse does not have, is a
  `ConfigurationError`.
* **Each item** runs through the normal pipeline — memory push and pull, the toolbox, risk tiers
  and approvals, records, traces — acting for `user` (default `trellis-evaluate`; memory is
  scoped to it, so give an evaluation its own user when its writes should stay apart). Then the
  evaluators score the answer, each score goes on the run's trace, and, for a Langfuse dataset,
  the run's trace is linked to the dataset run `run_name` as that item's result
  (`POST /api/public/dataset-run-items {runName, datasetItemId, traceId, metadata}`; the run is
  created by its first item). A link Langfuse refuses is a warning.
* **What cannot stop it**: an item whose run fails is `error` (with its message), one that
  pauses for a person is `interrupted` — its run is cancelled so it does not wait in an inbox —
  and one whose run is cancelled is `cancelled`; only `success` items are scored.
* **`concurrency`** items run at once (at least 1), **`limit`** keeps the first items only, and
  **`run_name`** defaults to `<agent id>-<UTC time>`.
* **At the end** the background writes are drained and the spans exported, so the scores are in
  Langfuse when the call returns.

`EvalReport`: `run_name`, `dataset` (the Langfuse dataset's name, or `None`), `items` — in
dataset order, each an `EvalResult(input, expected, output, status, run_id, scores, failed,
error, trace_url)` (`trace_url` is `<langfuse host>/trace/<trace id>` when Langfuse is
configured) — `statuses` (how many items ended each way) and `summary`: each evaluator's
`EvaluatorStats(mean, count, failures)` by name — the mean of its numeric and bool scores
(`None` for categorical ones), how many scores it gave, how many items it failed on.
`print(report)` and `print(report.summary)` are tables.

## Online: `Harness(judges=[...])`

```python
h = Harness(judges=[llm_judge("Polite, correct and concise.", name="quality"), cites_policy])
```

After a successful run with a text answer, if the run falls in the sample, each judge is queued
in the background writes (`judge.<name>`): the run has already returned when it runs. The case
is the run's question, answer, memory context and run id (no `expected`). Its score goes on the
run's trace; a judge that fails is a `warning` event on the run's stream (`judge_failed`) and a
log line.

**`TRELLIS_JUDGE_SAMPLE`** is the share of runs judged, 0 to 1. Unset, it is 0.1 when the harness
has judges (and nothing is judged without judges). The run id decides — salted apart from the
grounding sample (`TRELLIS_GROUNDING_SAMPLE`), so the two samples are independent — so a run is
either always or never judged, whichever process asks.

## Langfuse setup

Nothing beyond the OTLP variables that already send the traces there
([observability.md](observability.md#export)): `OTEL_EXPORTER_OTLP_HEADERS` with
`Authorization=Basic <base64 public-key:secret-key>`, and the endpoint Langfuse's
(`…/api/public/otel`) or the headers naming its host (`x-langfuse-host`, through a collector).
The same credentials reach its public API for scores, datasets and dataset runs. Without them,
scores are `score` spans on the run's trace only (every OTLP backend gets those), and a dataset
must be given as items.

In Langfuse, an offline evaluation is a **dataset run** of the dataset: each item's trace and
its scores side by side, comparable with earlier runs of the same dataset. Online judges'
scores are **scores on the traces**: filter traces by score, chart them per agent (the trace
name is the agent id) and over time, and send low scores to an annotation queue.

The API shapes are those of Langfuse's own definitions (`fern/apis/server/definition` in its
repository: `datasets.yml`, `dataset-items.yml`, `dataset-run-items.yml`, `scores.yml`). Langfuse
marks `POST /api/public/dataset-run-items` deprecated for Langfuse v4 — on Langfuse Cloud it is
to be removed on 16 November 2026, while self-hosted v3 deployments keep it; past that date a
link Langfuse refuses is a warning and the scores still land on the traces.
