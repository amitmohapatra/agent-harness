"""The comparison on one common set: the strict turns every framework ran with the memory
service up (not judge-only or unverifiable in any arm, not inapplicable to any).

Usage: python common_set.py <analysis.jsonl>   (from the repository root)
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from benchmark.cost import complex as cx

SETS = {
    "long57": "cases57.txt",
    "ordinary": "cases-ordinary.txt",
    "sw16": "cases-sw16.txt",
    "sw8": "cases-sw8.txt",
    "swown": "cases-swown.txt",
}
FOLDER = "benchmark/cost/results/haiku-2026-11-15/sets/"
out = sys.stdout.write


def norm(key: str) -> str:
    """A turn's key with its set's rerun suffix taken off (``long57md`` is ``long57``)."""
    name, rest = key.split(":", 1)
    return f"{re.sub(r'(rerun2md|rerun2|rerun|md)$', '', name)}:{rest}"


def arms_of(path: str) -> dict[str, dict[str, Any]]:
    arms = {}
    with open(path) as fh:
        for line in fh:
            x = json.loads(line)
            arms[x["run"].split("/")[-1][:-2]] = {
                "wrong": {norm(w.split()[0]) for w in x["wrong_turns"]},
                "down": {norm(t) for t in x["memory_down_turns"]},
                "unv": {norm(u.split()[0]) for u in x["unverifiable_turns"]},
                "na": {
                    f"{n.split(':')[0]}:{n.split(':')[1].split()[0]}" for n in x["not_applicable"]
                },
            }
    return arms


def scored() -> set[str]:
    cases = {c.id: c for c in cx.cases()}
    every = set()
    for name, f in SETS.items():
        with open(FOLDER + f) as fh:
            ids = fh.read().strip().split(",")
        for cid in ids:
            every |= {f"{name}:{cid}/t{n}" for n, t in enumerate(cases[cid].turns) if t.expect}
    return every


if __name__ == "__main__":
    arms = arms_of(sys.argv[1])
    every = scored()
    left = set()
    for v in arms.values():
        left |= v["down"] | v["unv"]
        left |= {k for k in every if any(k.startswith(n + "/") for n in v["na"])}
    common = every - left
    out(f"scored turns {len(every)}, common {len(common)}, left out {sorted(left)}\n")
    for arm, v in arms.items():
        wrong = v["wrong"] & common
        out(f"{arm}: {len(common) - len(wrong)}/{len(common)}  wrong: {sorted(wrong)}\n")
