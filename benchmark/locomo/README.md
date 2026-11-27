# LoCoMo through a harness agent

The memory repository's LoCoMo benchmark (`benchmark/locomo.py` there: the **direct arm**)
measures "memory pushes evidence, a model answers from it": retrieval builds one context, one
model call answers from it. This runner measures the product path beside it, the **harness
arm**: a harness agent is asked the same question in a new conversation, with the memory
context the harness pushes and whatever it then looks up itself with `memory_search`.

What it measures: the answers of that agent, per question, with what they cost (model calls,
input and output tokens, seconds) and how memory was used (the searches made, with their
`order` and `kinds`; the size and evidence status of the pushed context).

What it does not measure: it grades nothing. The answers are written in the shape the memory
repository's scorer reads, and that one scorer grades both arms. It also does not ingest:
the corpus is the direct arm's own, written by the direct arm's own code.

## What both arms share

| | where it comes from |
|---|---|
| the stored conversations | the direct arm's ingestion itself (below), never a copy of it |
| the questions | the dataset's, the direct arm's `--conversations N` and `--sample N` (`--limit N` here: the same seeded per-category draw, tested against its `_stratified`) |
| what memory is searched with | the question alone (`--reference instructions`, the default) |
| the answer prompt | the direct arm's `ANSWER_SYSTEM`, read from its source at run time |
| the reference date | the last session's date, in the direct arm's words |
| the judge and its rule | the memory repository's `benchmark.locomo_rescore`, run over both files |

## The corpus

The direct arm stores a turn in process (`_ingest_conversation`, `benchmark/locomo.py`
152-212): an observation whose user is the speaker, with TENANT visibility, the session's date
as `occurred_at`, no thread. No HTTP route stores that, so the memory SDK cannot mirror it and
this runner does not try. Both arms read one corpus only if:

1. the direct arm ran with `--reuse-corpus` (one tenant a conversation, `bench_conv_<index>`,
   kept after the run) against a real PostgreSQL database and a real Qdrant
   (`BENCH_SEARCH=qdrant`: its in-process vector store dies with the run);
2. the memory service at `MEMORY_URL` serves that database and that Qdrant, with the same
   encoders;
3. each conversation's tenant has a key: a memory-service key speaks for one tenant, so give
   conversation `<index>`'s in `LOCOMO_API_KEY_<index>` (or one key that names no tenant in
   `TRELLIS_API_KEY`). A key for another tenant is refused before any question is asked.

Run the direct arm first: it ingests, and its questions leave the store as they found it. A
harness run records its turn in the memory service (the question, the answer, the tool calls),
so the store is not the same afterwards.

`--ingest direct` stores the conversations without the direct arm's judged run: it starts the
direct arm's own entry point in the memory checkout (`python -m benchmark.locomo
--reuse-corpus --sample 1`, no `--judge`: no model call), with this process's environment, so
`MEMORY__DATABASE__URL`, `MEMORY__SEARCH__QDRANT_URL` and the `BENCH_*` variables must be the
ones the direct arm's run will have (its ledger keeps a key of them, and re-ingests on another).

## Commands

Two conversations, each arm. `<...>` are yours; the model is always a parameter.

The direct arm, in the memory repository (its result goes to `benchmark/results/`):

```sh
cd <memory-repo>
MEMORY__DATABASE__URL=<postgresql+psycopg://.../p7_locomo> \
BENCH_SEARCH=qdrant MEMORY__SEARCH__QDRANT_URL=<http://host:16333> BENCH_EMBEDDING=frozen \
BENCH_LLM=on LLM_BASE_URL=<endpoint>/v1 LLM_GATEWAY=<gateway> LLM_API_KEY=<key> \
BENCH_LLM_MODEL=<provider/model> PYTHONPATH=src:. \
  <memory-python> -m benchmark.locomo --judge --reuse-corpus --conversations 2 \
  --out locomo_direct_c2.json
```

The harness arm, here, memory on and then off (the floor). First what it would cost:

```sh
python -m benchmark.locomo --memory-repo <memory-repo> --conversations 2 --dry-run \
  --price-in <usd-per-million-input> --price-out <usd-per-million-output>
```

```sh
LLM_BASE_URL=<endpoint>/v1 LLM_GATEWAY=<gateway> LLM_API_KEY=<key> \
MEMORY_URL=<memory-service> LOCOMO_API_KEY_0=<key of bench_conv_0> \
LOCOMO_API_KEY_1=<key of bench_conv_1> TRELLIS_API_KEY=<either> \
  python -m benchmark.locomo --memory-repo <memory-repo> --model <provider/model> \
  --conversations 2 --arm push --out build/locomo-push-c2 \
  --price-in <in> --price-out <out> --max-usd <cap> --yes

LLM_BASE_URL=<endpoint>/v1 LLM_GATEWAY=<gateway> LLM_API_KEY=<key> \
  python -m benchmark.locomo --memory-repo <memory-repo> --model <provider/model> \
  --conversations 2 --arm off --out build/locomo-off-c2 \
  --price-in <in> --price-out <out> --max-usd <cap> --yes
```

Without `--yes` nothing is sent: the estimate is printed and the command ends with status 2.
A run that stopped (a cap, an error, Ctrl-C) is started again with the same command: a
question `--out` already answers is not asked again, one that failed is. `--dataset` names
`locomo10.json` where the memory checkout has none (it is git-ignored: a worktree has none).

One scorer, both files (in the memory repository, the direct arm's environment):

```sh
<memory-python> -m benchmark.locomo_rescore benchmark/results/locomo_direct_c2.json --judge-ruler strict
<memory-python> -m benchmark.locomo_rescore <harness-repo>/build/locomo-push-c2.json --judge-ruler strict
<memory-python> -m benchmark.locomo_rescore <harness-repo>/build/locomo-off-c2.json --judge-ruler strict
```

Each writes `<name>_strict.json` beside its input, with `by_category` (J per category;
adversarial is scored on abstaining), `answer_recall_at_k` and `judge_failures`. Compare the
three `_strict` files, not the direct arm's own in-run score: its in-run judge read whole
answers, the scorer reads the 400 characters the file kept.

## The files

* `<out>.json` — a LoCoMo result as the scorer reads one: `judged` true, and `records`, each
  with `conversation`, `category` (its name), `question`, `answer` (the gold one) and
  `judged.produced` (the agent's answer, whole). No verdict: `hit` is null and `scored` false.
  A question whose run gave no answer has `judged.error` and no `produced`, as a direct-arm
  row whose generation failed has; the scorer counts it in `judge_failures` and grades it for
  neither arm. Each record also carries `harness`: the run id, the searches with their
  `order` and `kinds`, tokens and model calls. `arm` says how it was asked, `direct` which
  source of the direct arm it was matched to, `usage` what it cost.
* `<out>.rows.jsonl` — one line per question asked, appended as it ends: all of the above,
  every tool call in order with its arguments, status, milliseconds and result size, the text
  sent, the user, thread and tenant. It is what a rerun reads.
* `<out>.ingest.json` — with `--ingest direct`, the direct arm's own (retrieval-only) result.

## Arms and options

`--arm push` is memory as the harness has it by default; `--arm off` is a harness with no
memory service (and `without=memory`): no context, no tool, no record.
`--adapter` is any of `react` (default), `langgraph`, `graph`, `deepagents`, `openai`,
`claude`, `kimi`. Every run is without the schedule tools, `ask_user`, the shell, the gateway's
MCP tools, tool search, and the sampled grounding check and judges: the memory service's tools
are the only ones (`claude` and `kimi` keep their CLI's own).

Whatever else the harness can be told about memory is passed through unchanged, so a mode it
gains later needs no change here: `--without NAME` (a `trellis.harness.features` name),
`--wrap KEY=JSON` (a keyword of `h.wrap`), `--run KEY=JSON` (a keyword of `agent.run`). What
exists today for a smaller push is `--window N`: the pushed context is a quarter of the
window, between 2,000 and 8,000 tokens.

## What still differs between the arms

* **The prompt.** The agent's instructions are `ANSWER_SYSTEM` word for word, then one
  paragraph this runner adds (`--added`: what "the CONTEXT" is for an agent, and to search
  before declining), then the reference date. Around them the harness and the framework add
  their own: the notes on answering from memory and on asking, the framework's base prompt,
  the tools' descriptions. The direct arm's model reads `ANSWER_SYSTEM` and nothing else.
* **Where the reference date is.** After the instructions here; in the user message in the
  direct arm. `--reference question` puts it in the message as the direct arm does, but a
  harness run's message is also the text memory is searched with, which the direct arm keeps
  to the question alone.
* **The context.** The direct arm renders its own bundle in process at the depth of its
  environment: unset, as above, the shipped one (a final depth of 50, 8,000 tokens);
  `BENCH_DEPTH=judged`, which the recorded judged results were taken at, is 100 and 12,000.
  The harness asks the service for a prompt context of at most 8,000 tokens (`--window`:
  less), from whatever configuration the service at `MEMORY_URL` runs with. The direct arm's
  context is the whole bundle rendered (`bundle.render()`); the harness's is the service's
  prompt form of it, with the harness's notes after it.
* **The service.** The direct arm is a container in the benchmark's process with stand-ins
  (cache, queue, authorization, lexical NLI); the harness arm is a deployed API.
* **Who asks.** The direct arm asks as `locomo-<index>` in workspace `ws`. Here each question
  has a user of its own (`locomo-<index>-q<n>`) and no workspace, so that what one question's
  run recorded is not another question's to read; `--asker conversation` is the direct arm's
  one user, with that leak. The corpus is TENANT-visible, so both read all of it.
* **More than one model call.** The agent may search and call the model again; the direct arm
  calls it once. It can also write (`memory_remember`, `memory_update`, `memory_forget`,
  `profile_edit` are memory tools); the rows show every call.
* **The model's settings.** The direct arm's answerer is the memory service's gateway client
  (16,384 output tokens, its own sampling); the agent's is the harness's route to the model.
  Use the same model for both, and for the judge.
* **Answer length.** The direct arm keeps 400 characters of an answer; these are kept whole
  (`answers_over_direct_cut` counts those longer).
* **Questions without an answer.** In its own run the direct arm scores a failed generation
  wrong; the scorer leaves such a row out of both arms' scores. Read `judge_failures`.

Not verified here: a run against a real memory service and a real model. The offline test
(`tests/unit/test_bench_locomo.py`) runs the whole path with a scripted model and the memory
fake, the direct arm's own sampling against this runner's, and the memory repository's real
scorer over a result this runner wrote.
