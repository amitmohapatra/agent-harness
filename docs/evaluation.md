# Evaluation: offline datasets and online judges

Langfuse is the system of record for evaluation: datasets, dataset runs, scores, dashboards and
annotation queues live there. Trellis fills it two ways:

| | Offline | Online |
|---|---|---|
| What | an agent run over a dataset, every answer scored | live runs scored as they happen |
| Way 1, wrapped (`h.wrap`) | `report = await h.evaluate(agent, dataset, judges)` | automatic: `h.wrap(judges=[...])`, and the sampled grounding check |
| Which runs | every item of the dataset (or the first `limit=`) | a sampled share of successful runs with an answer — a structured one (a pydantic model, a dict) graded as its JSON (each judge's `sample`, a tenth by default) |
| When | now: the call returns an `EvalReport` | after the run — in the background writes queue, never on the request path, when wrapped |
| Where the scores go | each run's trace (Langfuse scores, and a `score` span); a Langfuse dataset's runs are linked to the dataset run | each run's trace (or the trace a case names) |

Every evaluation name is imported from `trellis.harness.evals` (the `trellis` package holds the
wrapped API only). This page is how evaluation works and what a wrapped agent gets (Way 1);
evaluating your own code, not wrapped, with `evaluate` and `judge` is
[blocks/evaluation.md](blocks/evaluation.md) (Way 2): the same evaluators, judge and Langfuse
records. `examples/05_features/trajectory_evals.py` and `examples/05_features/online_judges.py` run Way 1 with no
services. Against the real memory service, `tests/live/test_live_matrix.py` runs a
wrapped evaluation with grounding, `exact_match` and a scripted judge, and checks the Langfuse
dataset run, scores and experiment attributes.

## Way 1 (wrapped): automatic, and `h.evaluate`

A wrapped agent is evaluated with nothing more than this:

```python
from trellis import Harness
from trellis.harness.evals import exact_match, grounding, llm_judge

# online, automatic: each judge's share of successful runs is judged in the background
quality = llm_judge("Polite, correct and concise.", name="quality", sample=0.2)
async with Harness() as h:
    agent = h.wrap(graph, id="support", judges=[quality])

    # offline: a Langfuse dataset by name, or the items themselves
    report = await h.evaluate(
        agent,
        "support-golden",  # or [{"input": ..., "expected": ...}, ...]
        [grounding(), exact_match(), llm_judge("Answers the question and cites the policy.")],
        run_name="nightly-2026-10-04",
    )
    print(report)  # items by status, then each evaluator's mean, count, failures
```

* **Automatic.** With memory on, a fixed tenth of the successful runs is checked against the
  context it was given ([memory.md](memory.md)), and with `judges=` each judge scores its
  share of runs (`llm_judge(sample=)`, [Online](#online-judges) below). Nothing to call.
  `grounding()` is therefore not one of `h.wrap`'s judges (refused, saying so: it would check
  the same runs twice); it is for `h.evaluate` and `judge()`.
* **`h.evaluate`** runs each item through the normal pipeline — memory, tools, governance and
  approvals, the agent's time limit (`h.wrap(run_timeout=)`: a hung item ends `TIMEOUT`, an
  `error` item) — and scores it ([Offline](#offline-evaluate-and-hevaluate) below).
* What it reaches is `h.evals`, an `EvalServices` the harness builds from its settings (sharing
  its gateway; Langfuse from its own keys); `agent.evals` is the harness's, whose judge is the
  gateway's judge alias, never the agent's own model ([The judge's model](#the-judges-model)).

This works for every target ([framework pages](README.md#which-target)).

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
| `trajectory` | the harness tool calls the run made, in order across its attempts, each with its outcome: `(ToolCall, ToolOutcome)` pairs (contracts; `call.tool`, `call.args`, `outcome.status`, `outcome.output`) — the run's journal, never a second store. A wrapped run's always (empty when it called nothing); a callable's `EvalOutput.trajectory`, else `None` (not known) |

`EvalScore(name, value, comment=None)`: `value` is a number from 0 to 1 (a Langfuse `NUMERIC`
score), a bool (`BOOLEAN`: 1 or 0) or a string (`CATEGORICAL`); `comment` is shown with it.
An evaluator that raises is a failure of that evaluator on that item (counted in the report,
logged), never a failed evaluation; online, it is a `warning` event and a log line, never a
failed run.

Built in:

| Evaluator | Score |
|---|---|
| `grounding(name="grounding")` | the share of the answer's claims the run's memory context supports — `grounding_score(memory, answer, bundle_id)`: the memory service's `/v1/verify` with the case's `bundle_id` in its `memory` scope, the same function as the sampled check every wrapped run gets (which, in a run's scope, also records the run's `judge` feedback there). `/v1/verify` takes an answer of at most 8000 characters: of a longer one, the first 8000, cut back to the last sentence or line that ends within them (`verified(answer)`), are checked — the score is that part's. No score without a memory scope, without a pushed context, or for an answer with no checkable claim |
| `exact_match(name="exact_match", case_sensitive=False)` | whether the answer is `expected` (text trimmed, case-blind by default; anything else compared as JSON); no score without `expected` |
| `contains(name="contains", case_sensitive=False)` | whether the answer contains `expected` — each of them, for a list; the comment names what is missing |
| `called(tool, *, before=None, args=None, name="called:<tool>")` | whether the run called `tool` — with `args`, each equal to the call's (its other arguments not compared) — and, with `before`, before its first call of `before` (which it must have made); the comment says what was called instead. No score without a trajectory |
| `tool_sequence(tools, *, exact=False, name="tool_sequence")` | whether the run called `tools` in this order (other calls between and around them allowed), or — `exact` — exactly these calls; the comment lists the calls made. No score without a trajectory |
| `llm_judge(criteria, name="llm_judge", *, model=None, sample=0.1)` | a judge model's grade, 0 to 1, against `criteria` written in plain language; its reasoning is the comment. `model`: this judge's model name, over the gateway's judge alias ([The judge's model](#the-judges-model)). `sample`: the share of runs it scores online, chosen by the run id |

### Trajectories

A run is judged by what it did, not only by what it answered: `case.trajectory` is every harness
tool call the run made — offline in `h.evaluate`, online for the judges — in order, with its
arguments and outcome:

```python
from trellis.harness.evals import EvalCase, EvalScore, called, tool_sequence


async def no_failed_calls(case: EvalCase) -> EvalScore | None:
    if case.trajectory is None:
        return None
    failed = [call.tool for call, outcome in case.trajectory if not outcome.ok]
    return EvalScore("no_failed_calls", not failed, ", ".join(failed) or None)


evaluators = [
    called("lookup_order", before="refund"),  # looked the order up before refunding it
    called("refund", args={"order": "o-17"}),
    tool_sequence(["lookup_order", "refund", "notify"]),
    no_failed_calls,
]
```

It is the run's journal (`Journal.trajectory`): the calls that ran, each once — a resumed run's
calls before its pause included, whether its framework went on where it stopped or ran again
from its input (a call replayed from the journal is not made again, and not listed again).
A call that did not run — denied by a hook, rejected by its approver, of a feature the run is
`without=`, still waiting for its approval — is not on it, and neither are the calls a
framework makes itself (Claude Code's built-ins, Deep Agents' file tools). The arguments and
outputs are as the tool had them, not redacted: the trajectory stays in the process, with the
evaluators. Code that is not wrapped returns its own as `EvalOutput(answer,
trajectory=[(ToolCall, ToolOutcome), ...])`; a callable that returns none has `None`, and the
trajectory evaluators give it no score.

**Tool coverage.** `tool_coverage()` scores each item by the share of the tools it needs that
its run called, 0 to 1 — the tools each item needs written on the item itself, so one dataset
holds requests that need different tools:

```python
from trellis.harness.evals import tool_coverage

dataset = [
    {
        "input": "Which styles have excess stock at S12, and which lost the most sales?",
        "metadata": {
            "needed_tools": ["excess_stock", "lost_sales"],
            "avoid_tools": ["store_overview"],
        },
    },
    {
        "input": "Create a stock rule for ST-101 at S12, min 2, max 6.",
        "metadata": {"needed_tools": ["create_stock_rule"]},
    },
]
evaluators = [tool_coverage()]  # or tool_coverage(["lookup"]) for every item alike
```

The comment names the needed tools not called, and the `avoid_tools` that were (a broader tool
the needed ones are more specific than: said, not counted). A run that paused for approval is
scored too — the call it paused on reached its gate, so it counts as made — while the other
evaluators score only runs that succeeded (the item stays `interrupted`; an evaluator of your
own scores paused runs with a class attribute `scores_paused = True`). Runnable:
[`examples/06_scenarios/tool_choice.py`](../examples/06_scenarios/tool_choice.py).

### `llm_judge`

The judge reads the criteria, the input, the expected answer and the memory context when there
are any, and the answer (each cut at 8000 characters), and is asked at temperature 0 for only
`{"score": <0..1>, "reasoning": "..."}`. Its reply is read robustly (a code fence or prose
around the object is fine); a reply that is still not that object, or a score outside 0..1, is
answered once with what was wrong, and a second bad reply is no score and a warning. It asks
the model of the services `evaluate` or `judge` was given; called on its own, outside them, it
is a `ConfigurationError`.

#### The judge's model

**Which model is the gateway's; whose budget is the deployment's:**

| | |
|---|---|
| `llm_judge(criteria, model="name")` | this judge's own Bifrost model name: two judges of one run may ask two models |
| the judge alias `trellis-judge` | else, with Bifrost (`LLM_GATEWAY=bifrost`), the model name every judge asks: an alias the Bifrost admin routes to the judging model, changed at runtime with no redeploy; on another route a judge names its model |
| `TRELLIS_JUDGE_VIRTUAL_KEY` | the virtual key the judge's calls go through; unset: `LLM_API_KEY` |

The judge never becomes the judged agent's own model. An alias the gateway does not route
(it answers 404) turns the judges off: each judge's failure says so in the report's `failed`,
and the log says it once. Each score says which model gave it — the comment ends
`[judge: <model>]`, its `score` span carries `trellis.score.model`, and a log line names it.
Every judge call carries Bifrost's dimensions of the run it judges (`x-bf-dim-run`, and online
also its agent, tenant and hashed user) with `x-bf-dim-purpose: judge`, so its spend is
attributed in Bifrost's logs and metrics.

Route the alias to a **different, stronger model than the agent's**: a model grading its own
answers is biased towards them. Give the judge **its own virtual key**: its spend is then
budgeted, rate-limited and reported in Bifrost apart from the agents' traffic, so evaluation
can never exhaust production's budget. Bifrost's semantic cache is not used for judges: a
verdict replayed for a merely similar answer would be wrong.

**Cost.** Every judged answer is one judge model call (two when the first reply is malformed),
with a prompt of the criteria plus the case — up to about 32 000 characters. Offline that is
one call per item and judge; online it is one per *sampled* run and judge: at the default 10 %
sample, 1 000 runs a day cost about 100 judge calls a day per judge. Sample down
(`llm_judge(sample=)`), judge offline on a dataset instead, or use a cheaper deterministic
evaluator where one will do.

## Offline: `evaluate` and `h.evaluate`

```text
await h.evaluate(agent, dataset, judges, *, run_name=None, description=None, metadata=None,
                 concurrency=4, limit=None, user=None)
```

`judges` are the evaluators that score each item — the same kind `h.wrap(judges=)` takes (one
word for both; the type a judge implements is `Evaluator`). `h.evaluate(agent, ...)` is `evaluate(agent, ...)` (`trellis.harness.evals`) with the agent's
own services (`agent.evals`). `evaluate` also takes any `async (input) -> answer` as its
target, for code you do not wrap ([blocks/evaluation.md](blocks/evaluation.md#offline-evaluate));
the dataset, the report and the experiment below are the same for both.

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

## Online: judges

```python
quality = llm_judge("Polite, correct and concise.", name="quality", sample=0.25)
agent = h.wrap(graph, id="support", judges=[quality, cites_policy])
```

After a successful run of a wrapped agent with an answer (a structured one as its JSON), if the run falls in the sample,
each judge is queued in the background writes (`judge.<name>`): the run has already returned
when it runs. Each is one `judge(case, [that judge], services=agent.evals)`; the case is the
run's question, answer, memory context and scope, trajectory (across all its attempts), and run
id (no `expected`). Its score goes on
the run's trace; a judge that fails is a `warning` event on the run's stream (`judge_failed`)
and a log line.

**The share of runs.** Each judge scores `sample` of the runs, 0 to 1 (`llm_judge(sample=)`;
a judge of your own with a `sample` attribute says its own; else a tenth); a fixed tenth of the
runs given a memory context is checked for grounding. `without={"judges"}` or `{"grounding"}`
turns one of the two off. The run id decides — salted apart per judge, so the samples are
independent — so a run is either always or never judged, whichever process asks. `judge(..., sample=rate)` from code you
do not wrap samples the same way ([blocks/evaluation.md](blocks/evaluation.md#online-judge)).

## Langfuse setup

Nothing beyond what already sends the traces there
([observability.md](observability.md#export)): Langfuse's own `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL` (unset: Langfuse Cloud). The same keys reach its
public API for scores, datasets and dataset runs — the harness's and
`EvalServices.from_env()`'s alike. Without them, scores are `score` spans on the run's trace
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

## Cost benchmark

`make bench-cost` (`python -m benchmark.cost`) measures what a `ReAct` agent costs and how
well it chooses its tools as its tool catalogue grows, against whatever model the environment
configures (`LLM_BASE_URL`, `LLM_GATEWAY=bifrost`, `LLM_API_KEY`, a gateway model name), so a cost feature's
saving is a measured number before it is switched on by default.

```bash
make bench-cost BENCH_ARGS="--model provider/model --repeats 3"
# a local model with a small window: --window 8192; one size, one set: --sizes 50 --sets travel
```

| | |
|---|---|
| Tasks | ten, from the tool-choice sets the live suite uses — the store set ([`examples/06_scenarios/tool_choice.py`](../examples/06_scenarios/tool_choice.py)) and the held-out help desk and travel sets (`tests/live/heldout.py`): six short (two parts, a change behind approval, a read that sounds like a change), three long (four and five parts, each served by its own tool) and a large result — a help desk export of about 100,000 characters, past what `ReAct` pages. The report gives every figure for all of them and by length |
| Catalogues | 10, 50, 100 and 500 tools (`--sizes`): each set's own tools, then distractors — 500 synthetic but realistic MCP-style tools (`<server>-<tool>`) of 25 servers (code hosting, CRM, billing, HR, monitoring, DNS...), list / get / create / update / delete of four kinds of record each, every one described and with a typed, described JSON schema; none is about the tasks' domains. The order is fixed, so a run is reproducible |
| Variants | `baseline`: plain tools, everything off (`create_agent` with only what runs the harness's tools and bounds the loop; `without={"code_mode", "tool_search"}`). `main`: `ReAct` as it is built — `create_agent` with `middleware()`: context editing, the request fitted to the window, Code Mode where it applies (gateway MCP servers; the catalogues here are the agent's own tools, so it does not). `deep`: `ReAct` on Deep Agents (a backend given) — result paging, summarization, its long-task files and helper. `pieces`: `create_agent` with Deep Agents' pieces attached and offered per call. `deep-paging`, `deep-context_editing`, `deep-summarization`: `deep` with that one piece off. `--suite classes`: ten tasks per class (simple, multi-step, wide, a large result, a chain of two to four dependent calls) and a mixed ten for `--sizes 50,100`; `--memory`: the environment's memory service on. A new feature is one more entry of `benchmark/cost/variants.py`, measured against these (`--variants`) |
| Per run | input, output and cached tokens (as the model reports them), model calls (every chat model call of the run, a summarization's included) and each call's seconds (the report gives their p50 and p95 per cell), wall time, the CPU seconds of the benchmark's own process and of each `--watch name=pid` process (the model server, the gateway; the report adds each one's peak memory), `tool_coverage` (the share of the task's needed tools called; for a run that ended in an error, from the calls it made) and task success (every needed tool called, no broader one, every other check of the task passed — `contains`, `picks`: the store the answer concludes on, not the first it names) |
| Cost | `benchmark/cost/prices.py`: dollars per million tokens, cached input and cache writes priced apart, each price with its source and the date it was read. A model not in the table is reported with its tokens and no dollars; `--price-as` prices the same tokens at listed models' rates as well (an estimate: their tokenizers count differently) |
| Report | `build/bench-cost.json` (every run, each cell's means and 95% intervals — Student's t over the runs) and `build/bench-cost.md` (the table), written after every cell |

Memory is off (the tool hints, which narrow a run from five tools, need the memory service),
and so are the run store and tracing: the benchmark reads and writes no service's data.

### The baseline, measured

`benchmark/cost/results/baseline-local.md` (and `.json`, every run): Granite 3.3 2B on
llama.cpp (an 8,192-token window, 4 CPUs) behind a local Bifrost, 2026-10-21 — the numbers
each new feature is compared with. Per run, mean ± the 95% interval's half-width:

| Tools | Variant | Runs | Input tokens | Output tokens | Model calls | Seconds | Tool coverage | Success |
|---|---|---|---|---|---|---|---|---|
| 10 | baseline | 20 | 2,654 ± 336 | 182 ± 52 | 1.90 ± 0.14 | 38.3 ± 10.2 | 1.00 | 0.90 ± 0.14 |
| 10 | main | 20 | 2,794 ± 335 | 223 ± 85 | 1.85 ± 0.17 | 51.2 ± 20.6 | 1.00 | 0.95 ± 0.10 |
| 50 (travel) | baseline | 6 | 17,742 ± 4,517 | 204 ± 151 | 2.33 ± 0.54 | 159 ± 94 | 1.00 | 1.00 |
| 50 (travel) | main | 6 | 15,046 ± 141 | 168 ± 99 | 2.00 | 146 ± 95 | 1.00 | 0.67 ± 0.54 |
| 100 | both | 20 each | — (does not fit: ~14,500 prompt tokens) | | | | 0 | 0 |
| 500 | both | 20 each | — (does not fit: ~71,000 prompt tokens) | | | | 0 | 0 |

What it shows:

* **Short and long tasks at 10 tools cost the same in both variants** (the same input tokens
  and calls): what `main` adds costs nothing until it is needed.
* **The large result is where `main` pays.** The baseline sent the ~100,000-character export
  whole (about 45,000 prompt tokens, refused by an 8k window: both runs failed); `main` paged
  it and answered from the preview (one run right, the other paused at a write distractor it
  called). With a large-window vendor model the baseline would pay those 45,000 tokens on
  every later call of the run.
* **At 50 tools the tool definitions are the cost**: about 7,000 of every call's ~7,500 prompt
  tokens. `main`'s summarization guard stopped both four-part runs when the history and the
  tools passed its budget — compaction cannot shrink the tools — where the baseline, sent to the
  server, still fit: on a small window the guard is stricter than the server.
* **From 100 tools nothing fits an 8k window** — the definitions alone are ~14,500 tokens
  (100) and ~71,000 (500). The baseline sends each request and the server refuses it; `main`
  refuses before sending. Tool search (deferring the definitions) is what makes these sizes
  work at all.

Not measured here: any vendor model (no `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or vendor
provider in the gateway — the dollar columns of the report are the same token counts priced
at those models' rates, an estimate), vendor prompt caching (the cached tokens are llama.cpp's
own prefix cache), the tool hints (they need the memory service), Code Mode (it needs gateway
MCP servers), and the full task set at 50 tools (about 65 minutes per variant on this
machine; the travel set ran).

### The gate

`make bench-gate` (`python -m benchmark.cost.gate NOW BASELINE`) compares a report with the
stored baseline, cell by cell (catalogue size and variant) on the tasks both ran, and exits
non-zero when a cell is worse past its margin: a success rate lower by more than
`--success-margin` (0.15: a small model's runs vary), or mean input tokens higher by more than
`--token-margin` (10%) on the successful runs of the tasks both reports did (a run that stopped
early sent fewer tokens for less work: counting it would reward failing). Better passes — the gate is non-inferiority, not equality — and a cell
only one report measured is listed, not failed.

`.github/workflows/bench.yml` runs it where a gateway serving the baseline's model is
configured (the repository variable `BENCH_LLM_BASE_URL` and the secret
`BENCH_LLM_API_KEY`, a Bifrost; without them the job is skipped): the smoke set (10 tools, one
repeat) on a pull request, the 10- and 50-tool catalogues nightly.

## Thread-history benchmark

`python -m benchmark.thread --model <name>` (with `MEMORY_URL` and a gateway serving the name)
measures who should keep a conversation's history across runs: a `create_agent` graph with its
own checkpointer (the framework's thread; the harness sends no memory window) against the same
graph without one (the memory service's window), on conversations whose later turns ask back
facts from the first. It reports recall, input tokens per turn and seconds per turn
(`build/bench-thread.json`, `.md`). The measurement and the decision are in
[ADR 0003](adr/0003-automatic-features.md) (§6).

## Load benchmark

`make bench-load` (`python -m benchmark.load`) runs many `ReAct` runs at once — a tool call and
an answer each — on one and on two worker processes (`--workers 1,2`), at each level of
concurrency (`--levels 1,10,50,100`, split across the workers, `--rounds` runs per lane), and
reports per level the throughput (runs a second), the run latency's p50, p95 and p99, the
errors, and each process's CPU seconds and peak memory (the workers, and `--watch name=pid`).
The climb stops for a worker count at the first level that hits a limit — errors past
`--max-errors` (5%), a p95 past `--max-p95`, or under `--min-free-mb` of the machine's memory
free before the level starts — and the report says which.

```bash
make bench-load                                            # no model: the harness alone
make bench-load LOAD_ARGS="--model provider/model --leak-minutes 30"
```

Without `--model` the model is `Paced`, a scripted model that answers after `--pace` seconds:
the latency and the CPU are the harness's own, under load. `--leak-minutes` then runs one
worker at `--leak-concurrency` for that long, sampling its resident memory, and reports the
growth from the end of the warm-up (the first fifth) to the end, in MB and per run.

Each worker is a process with its own harness (memory, the run store and tracing off, as in the
cost benchmark), so the processes share nothing: the run store's queue (`h.worker`) is not in
the measured path (`--queued` puts it there, below). Writes `build/bench-load.json` and `.md`.

### The production path

`--queued MINUTES` measures what a deployment runs instead: this process queues runs in
agent-runs (`RUNS_URL`) — `agent.start`, then the run's result — keeping
`--leak-concurrency` in flight, and `--workers` worker processes claim and execute them
(`h.worker([agent]).run()`), with the memory service (`MEMORY_URL`) on, or, with
`--no-memory`, only as the key registry agent-runs asks. It samples the resident memory of
each worker process and of each `--watch name=pid` process (the memory API and worker,
agent-runs, its ticker) and reports each one's slope after the warm-up, in MB per hour.

```bash
make bench-load LOAD_ARGS="--queued=15 --leak-concurrency=10 --watch=agent-runs=<pid>"
```

Measured 2026-10-21 on a private stack (the memory service and agent-runs at their `main`, a
database of their own each; never a shared one), Paced model, one worker process, 10 runs in
flight, 15 minutes each, 4 CPUs:

| Setup | Runs | Runs/s | p50 / p95 / p99 s | Errors | Slope, MB per hour: harness worker / memory API / memory worker / agent-runs / ticker |
|---|---|---|---|---|---|
| `RUNS_URL` (memory as the key registry) | 3,784 | 4.05 | 2.19 / 4.18 / 4.79 | 0 | 4.0 / 0.8 / 0.9 / 1.3 / 1.0 |
| `MEMORY_URL` + `RUNS_URL` (memory on) | 3,466 | 3.71 | 2.47 / 4.32 / 5.31 | 1 | 4.9 / 4.3 / 2.8 / 3.9 / 1.0 |

Every process is flat after the warm-up: the harness worker held 140–144 MB, the memory API
and worker ~2.1–2.3 GB each (their encoders), agent-runs 138 MB, the ticker 115 MB; the
slopes are a few MB an hour (0.5–1.5 kB a run: allocator noise at this length, not a growth a
longer run continues). The one error was agent-runs' tenant request budget (429) answering a
result poll; `handle.result` now waits what it says and polls again.
