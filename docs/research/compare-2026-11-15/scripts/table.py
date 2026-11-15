"""The comparison table from analyse.py's JSON lines (one per run dir, named <arm>-<run>), with
each run dir's services-check.json (runs_check.py) for the memory and agent-runs columns.

Usage: python table.py <analysis.jsonl>
"""

from __future__ import annotations

import collections
import json
import os
import sys

out = sys.stdout.write

runs = collections.defaultdict(list)
with open(sys.argv[1]) as fh:
    lines = fh.readlines()
for line in lines:
    x = json.loads(line)
    arm = x["run"].rstrip("/").split("/")[-1].rsplit("-", 1)[0]
    runs[arm].append(x)

out(
    "| framework | strict right/scored per run | unverifiable | p50 s | p95 s | max s "
    "| in / cached / out tokens per turn | model calls / turn (provider) | tool calls / turn "
    "| $ per run (prices; provider) | not SUCCESS turns | checks failed "
    "| memory: turns read / written of turns, requests per turn, failed "
    "| agent-runs: runs found / turns, events, spend = rows |\n"
)
out("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
for arm, xs in runs.items():
    acc = " + ".join(f"{x['right_counted']}/{x['strict_counted']}" for x in xs)
    if len(xs) == 3:
        acc += f" = {sum(x['right'] for x in xs)}/{sum(x['strict'] for x in xs)}"
    x = xs[0]
    pt = x["per_turn"]
    prov = x["provider"]
    calls = f"{pt['model_calls']}"
    if prov.get("responses"):
        calls += f" ({prov['responses'] / x['turns']:.2f})"
    usd = f"{x['usd_run']:.4f}" + (f"; {prov['usd']:.4f}" if prov.get("usd") is not None else "")
    bad = sum(v for k, v in x["statuses"].items() if k != "SUCCESS")
    sc = os.path.join(x["run"], "services-check.json")
    if os.path.exists(sc):
        with open(sc) as fh:
            c = json.load(fh)
        real = [f for f in c["memory_failed"] if "model-key" not in f and "NotFound" not in f]
        mem = (
            f"{c['turns_with_context_read']} / {c['turns_with_transcript_write']} of "
            f"{c['turns']}, {c['memory_per_turn']}, {len(real)}"
        )
        stored = (
            f"{c['runs_found']} / {c['turns']}, {c['events']}, "
            f"{'yes' if not c['spend_token_mismatches'] else len(c['spend_token_mismatches'])}"
        )
    else:
        mem = stored = "off"
    out(
        f"| {arm} | {acc} | {x['unverifiable']} | {x['p50']} | {x['p95']} | {x['max']} "
        f"| {pt['input']:,} / {pt['cached']:,} / {pt['output']:,} | {calls} | {pt['tool_calls']} "
        f"| {usd} | {bad} | {len(x['checks_failed'])} | {mem} | {stored} |\n"
    )
