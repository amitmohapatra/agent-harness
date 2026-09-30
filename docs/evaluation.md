# Evaluation (`trellis.eval`)

## The online judge

A sampled share of successful runs with a text answer (`TRELLIS_EVAL_SAMPLE`, default 0.1,
deterministic in the run id) is scored in the background by `GroundedJudge`:

1. **reference** — offline only: an answer equal (normalised) to the item's
   `expected_output` scores 1.0 without a model;
2. **grounded** — the memory service's verification of the answer against the context the
   run was given: all claims supported scores 1.0, a contradiction scores the supported share.
   When the service says it already consulted its own LLM judge, an undecided report
   abstains rather than paying twice;
3. **LLM** — what is still undecided goes to `openrouter/openai/gpt-4.1-nano` through Bifrost
   with a short rubric (evidence, reference answer, the unsettled claims) and a JSON-schema
   verdict.

A judge that cannot decide abstains; abstaining is never a zero. `JudgeBudget` bounds it per
agent: 60 judgements and 1 USD per rolling hour, the slot taken atomically. A verdict becomes an
`ANSWER` feedback record in the memory service and the `trellis.judge.score` histogram.

## Datasets and experiments

`DatasetBuilder(name).add(record, feedback, evidence=...)` turns runs and what people said into
`DatasetItem`s (`input`, `expected_output`, `evidence`): a correction is the expected output, a
confirmation keeps the run's output, a rejection is a negative example. Judge feedback is not
ground truth by default. `from_store(runs, feedback_reader, run_ids=...)` assembles from a run
store and the feedback store. `Dataset.write/read` for JSON files.

`ExperimentRunner(judge, output_dir=...).run(dataset, candidate, agent_id=, agent_version=)`
replays every item (`candidate` is `async (input) -> answer`, e.g.
`lambda x: agent.run(x, user="eval")`), scores it with the reference answer and the evidence,
and writes `<name>.json` with a `summary` (`mean_score` over judged items only).

## The regression gate

```bash
python -m trellis.eval gate --baseline benchmark-results.json --current build/benchmark-results.json \
    [--judge build/experiment.json --baseline-judge base.json --max-score-drop 0.05 --min-score 0.7] \
    [--max-latency-regression 20] [--allow-missing-baseline]
```

Fails (exit 1) on a latency percentile regressed past the threshold or a judge score drop; a
missing baseline fails unless allowed. CI runs the overhead benchmark
(`tests/performance`, writing `build/benchmark-results.json`) and gates it against the committed
`benchmark-results.json` (`make gate` locally).
