# Performance

## Budget

Harness-only overhead target: **p50 < 1 ms, p95 < 5 ms**, excluding network and provider
calls. "Harness-only" means the wrapped execution minus the same agent called directly,
with no memory client, no model and no tools configured — what is left is the pipeline:
context creation, the interceptor chain, spans, metrics, result coercion.

## Measured

Development machine: macOS (Darwin 21.6.0, x86_64), CPython 3.12.14, 2000 iterations after
200 warm-up runs. Spans are exported in-process through a `SimpleSpanProcessor`, which is
synchronous and therefore conservative. Three runs, reported as a range, because a laptop
is not a quiet benchmark host:

| Configuration | p50 | p95 |
| --- | --- | --- |
| Telemetry enabled | 0.66 – 0.71 ms | 1.06 – 1.91 ms |
| Telemetry disabled | 0.36 – 0.42 ms | 0.59 – 1.07 ms |
| Sampled out (`sample_rate: 0`) | 0.36 – 0.42 ms | 0.85 – 1.05 ms |

Context creation alone: p50 0.019 – 0.024 ms, p95 0.04 – 0.07 ms.

Against the budget: p50 is inside the 1 ms target in every run; p95 is inside the 5 ms
target with a wide margin. The variance between runs is larger than the difference between
some of the configurations — treat single runs accordingly.

These are *this machine's* numbers. Re-measure on yours; do not quote them as a guarantee.

## Reproducing

```bash
pytest tests/performance -m performance -q -s
UAH_BENCH_ITERATIONS=10000 pytest tests/performance -m performance -q -s
```

The run writes `benchmark-results.json` at the repository root.

## Where the time goes

Roughly, per execution: creating the context and runtime, running ~6 interceptors, opening
and closing one span, recording two metrics, coercing the result. Sampling out an execution
removes the span work but keeps metrics — which is why the "sampled out" row matches the
"telemetry disabled" row.

## Keeping it fast

* Keep the interceptor and listener counts small and bounded — cost is O(I) and O(L).
* Lower `sampling.sample_rate` in high-volume services; errors and critical agents can stay
  at 1.0.
* Leave `memory.writeback: true` so the turn never waits for memory writes.
* Do not put large payloads in `AgentResult.data`; use artifacts (the harness offloads
  anything over `artifacts.inline_max_bytes` and warns).
* In production use a `BatchSpanProcessor` rather than the synchronous one used in these
  measurements.

## Regression gate

`tests/performance/test_overhead.py` asserts loose ceilings (p50 < 5 ms with telemetry on)
so the suite is stable on shared CI hardware. For a real regression gate, record a baseline
on your own runner and compare `benchmark-results.json` between builds.
