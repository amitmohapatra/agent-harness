"""Re-score saved complex-suite rows with the current scorer: each turn's answer against its
case's Expect, with the requests of that turn as sent (the said context)."""

import collections
import glob
import gzip
import json
import os
import sys

from benchmark.cost import approvals
from benchmark.cost import complex as cx

cases = {c.id: c for c in cx.cases()}
for d in sys.argv[1:]:
    tot = ok = agree = 0
    wrong = []
    unver = []
    for f in sorted(glob.glob(d + "/*.jsonl")):
        if "requests" in f:
            continue
        reqs = collections.defaultdict(list)
        rq = f[:-6] + ".requests.jsonl.gz"
        if os.path.exists(rq):
            with gzip.open(rq, "rt") as fh:
                for line in fh:
                    q = json.loads(line)
                    reqs[(q["case"], q["turn"])].append(q)
        rows = {}
        with open(f) as fh:
            for line in fh:
                x = json.loads(line)
                rows[x["case"]] = x
        for cid, x in rows.items():
            case = cases[cid]
            records = []
            for t in x["turns"]:
                e = case.turns[t["turn"]].expect
                if e is not None:
                    q = reqs.get((cid, t["turn"]), [])
                    said = cx.said_in(q, [])
                    used = set(t.get("native") or [])
                    ran = any(
                        c["tool"] in cx.RAN_CODE and c["status"] == "ok" for c in t.get("calls", [])
                    )
                    s = cx.score(
                        e,
                        t.get("answer") or "",
                        records,
                        t["text"],
                        said,
                        ran=ran or bool(used & approvals.SHELLS),
                        searched=bool(used & cx.WEB_SEARCH),
                        queried=cx.queried(q),
                    )
                    if s.get("correct") is not None:
                        tot += 1
                        ok += bool(s["correct"])
                        if not s["correct"]:
                            said = {k: s.get(k) for k in ("missing", "made_up", "forbidden")}
                            wrong.append(f"{cid}/t{t['turn']} {said}")
                    else:
                        unver.append(f"{cid}/t{t['turn']} {s.get('kind')}")
                    agree += s.get("correct") == (t.get("score") or {}).get("correct")
                records.append(t)
    sys.stdout.write(
        f"{os.path.basename(d)} re-scored {ok}/{tot} strict; agrees with the run's own score "
        f"on {agree} turns\n"
    )
    for w in wrong:
        sys.stdout.write(f"   wrong: {w}\n")
    for u in unver:
        sys.stdout.write(f"   not strict: {u}\n")
