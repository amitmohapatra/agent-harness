# Big-model judge protocol (keep rule for every harness change)

> The keep rule of the [improvement backlog](IMPROVEMENT_BACKLOG.md): a harness change is kept only when it passes this protocol against main. The cases, the captured inputs and the verdicts of each run go beside the change (its PR), so the decision can be checked.

No strong-model API key is available, so the session's own big model acts as judge and as the
"bigger model" reference. Every eval case records the exact model input (captured request) and
the small model's answer.

For each case the judge produces two verdicts, blind to which arm produced it:
1. **Answer correct?** Small model's answer vs ground truth (the run record / Python-computed
   truth). Strict scorer first; judge only for free-text kinds; record made-up claims.
2. **Context sufficient?** The judge answers the question itself using ONLY the captured
   model input (no outside knowledge, no tools beyond those offered in that input). If the
   judge's answer is correct → the harness gave enough; if not → the harness missed something
   (record what was missing).

Diagnosis per case:
- sufficient + small correct → OK.
- sufficient + small wrong → model capability gap (harness may help with format/steps, but the
  context was complete).
- insufficient → HARNESS gap (missing data, cut results, stale/undated values, wrong tools
  offered). Fix the harness.

Keep rule for a harness change (vs main, same cases, same model):
- context-sufficiency rate goes UP (or stays at 100%) and small-model strict accuracy does not
  drop, made-up rate does not rise, controls do not drop;
- tokens / model calls / seconds not materially up (accuracy wins ties against cost);
- otherwise the change is not kept.

Judging hygiene: arms shuffled and labels hidden; the judge prompt is fixed and versioned;
10% of cases double-judged by an independent judge pass to check agreement.
