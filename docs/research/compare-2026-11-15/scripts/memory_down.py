"""For each arm's run dir: the cases whose turns ran with the memory service down
(``analyse.memory_down``), printed as ``arm|adapter variant port|<set>md:x:cap:window|cases``
lines for run.sh (SETS, CASES_OVERRIDE); with ``--apply`` they are also set aside from their
set's rows (``<set>.replaced-memory-down.jsonl``), so the run again replaces them.

Usage: python memory_down.py <out-dir> [--apply]
"""

from __future__ import annotations

import glob
import json
import os
import sys

from analyse import loaded, memory_down

ARMS = {
    "react-plain": "react main -",
    "react-deep": "react deep -",
    "langgraph": "langgraph main -",
    "graph": "graph main -",
    "deepagents": "deepagents main -",
    "openai": "openai main 18106",
    "claude": "claude main 18107",
    "kimi": "kimi main -",
}
WINDOWS = {"sw16": "16000", "sw8": "8000"}
out = sys.stdout.write

if __name__ == "__main__":
    for d in sorted(glob.glob(sys.argv[1] + "/*-1")):
        arm = os.path.basename(d)[:-2]
        for f in sorted(glob.glob(d + "/*.jsonl")):
            if "requests" in f or "meter" in f or "replaced" in f:
                continue
            name = os.path.basename(f)[:-6]
            hit = memory_down(loaded(f[:-6] + ".services.json"))
            with open(f) as fh:
                rows = fh.readlines()
            keep = [r for r in rows if json.loads(r)["case"] not in hit]
            gone = [r for r in rows if json.loads(r)["case"] in hit]
            if not gone:
                continue
            if "--apply" in sys.argv:
                with open(f[:-6] + ".replaced-memory-down.jsonl", "a") as fh:
                    fh.writelines(gone)
                with open(f, "w") as fh:
                    fh.writelines(keep)
            base = name.rstrip("0123456789").removesuffix("rerun").removesuffix("md")
            cases = ",".join(json.loads(r)["case"] for r in gone)
            out(f"{arm}|{ARMS[arm]}|{name}md:x:0.15:{WINDOWS.get(base, '-')}|{cases}\n")
