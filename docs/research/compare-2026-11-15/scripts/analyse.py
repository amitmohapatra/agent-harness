"""One framework run's figures, from its saved files: re-scored accuracy (today's scorer: each
scored turn against its case's Expect, with what the model was sent — the turn's captured
requests, or for a CLI the context the run kept, ``*.said.json``), latency, tokens, model and
tool calls, cost at ``benchmark.cost.prices``, failures, and the provider's usage (meter.jsonl)
against the harness's.

Usage: python analyse.py <run-dir> [...]   (prints one JSON object per run dir)
"""

from __future__ import annotations

import collections
import glob
import gzip
import json
import os
import statistics
import sys
from typing import Any

from benchmark.cost import approvals, prices
from benchmark.cost import complex as cx

CASES = {c.id: c for c in cx.cases()}
PRICE = prices.price_of("claude-haiku-5-5")
assert PRICE is not None


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 2)


def requests_of(path: str) -> dict[tuple[str, int], list[dict[str, Any]]]:
    reqs: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    if os.path.exists(path):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                q = json.loads(line)
                reqs[(q["case"], q["turn"])].append(q)
    return reqs


def loaded(path: str) -> list[dict[str, Any]]:
    """A saved JSON file's rows, or its gzipped copy's (``<path>.gz``); none when neither is."""
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    if os.path.exists(path + ".gz"):
        with gzip.open(path + ".gz", "rt") as fh:
            return json.load(fh)
    return []


def memory_down(services: list[dict[str, Any]]) -> set[str]:
    """The cases in which the memory service was down for the harness: a request its client's
    circuit refused, or a context read or transcript write that got no answer."""
    out = set()
    for x in services:
        if x.get("kind") != "memory" or not x.get("error"):
            continue
        refused = x["error"] == "CircuitOpenError"
        lost = x["path"] in ("/v1/context", "/v1/messages") and x["error"] in (
            "TimeoutError",
            "DependencyUnavailableError",
            "ConnectError",
        )
        if refused or lost:
            out.add(x["case"])
    return out


def provider(meter: str) -> dict[str, Any]:
    """The provider's usage of every response the meter saw (both wire formats)."""
    out = collections.Counter()
    seconds = 0.0
    if not os.path.exists(meter):
        return {}
    with open(meter) as fh:
        for line in fh:
            x = json.loads(line)
            u = x.get("usage") or {}
            out["responses"] += 1
            out["errors"] += x["status"] >= 400
            seconds += x.get("seconds") or 0
            if "prompt_tokens" in u:  # OpenAI-compatible: prompt includes cached
                details = u.get("prompt_tokens_details") or {}
                cached = details.get("cached_tokens") or 0
                out["input"] += u.get("prompt_tokens") or 0
                out["cached"] += cached
                out["output"] += u.get("completion_tokens") or 0
            else:  # Anthropic messages: input excludes cache reads and writes
                read = u.get("cache_read_input_tokens") or 0
                write = u.get("cache_creation_input_tokens") or 0
                out["input"] += (u.get("input_tokens") or 0) + read + write
                out["cached"] += read
                out["cache_write"] += write
                out["output"] += u.get("output_tokens") or 0
    fresh = out["input"] - out["cached"] - out["cache_write"]
    usd = (
        fresh * PRICE.input
        + out["cached"] * PRICE.cached_input
        + out["cache_write"] * (PRICE.cache_write or PRICE.input)
        + out["output"] * PRICE.output
    ) / 1e6
    return dict(out) | {"usd": round(usd, 6), "model_seconds": round(seconds, 1)}


def analyse(d: str) -> dict[str, Any]:  # noqa: PLR0912, PLR0915 - one report, read top down
    turns: list[dict[str, Any]] = []
    right = wrong = unver = agree = 0
    #: turns of a case in which the memory service was down (a request refused by the
    #: client's open circuit, or a context read or transcript write that got no answer):
    #: not counted (``*_counted``), each listed
    down: list[str] = []
    right_counted = strict_counted = 0
    wrongs: list[str] = []
    unvers: list[str] = []
    na: list[str] = []
    usd = 0.0
    accounted = collections.Counter()
    for f in sorted(glob.glob(d + "/*.jsonl")):
        if "requests" in f or "replaced" in f or f.endswith("meter.jsonl"):
            continue
        name = os.path.basename(f)[:-6]
        reqs = requests_of(f[:-6] + ".requests.jsonl.gz")
        hit = memory_down(loaded(f[:-6] + ".services.json"))
        said_file = f[:-6] + ".said.json"
        said_rows = loaded(said_file)
        said_of = {(s["case"], s["n"]): s["said"] for s in said_rows}
        acc_file = f[:-6] + ".accounted.json"
        if os.path.exists(acc_file):
            for a in loaded(acc_file):
                accounted["input"] += a.get("input_tokens") or 0
                accounted["output"] += a.get("output_tokens") or 0
                accounted["cached"] += a.get("cached_tokens") or a.get("cached") or 0
                accounted["calls"] += 1
        rows = {}
        with open(f) as fh:
            for line in fh:
                x = json.loads(line)
                rows[x["case"]] = x
        for cid, x in rows.items():
            usd += x.get("usd") or 0
            if x.get("not_applicable"):
                na.append(f"{name}:{cid} ({x['not_applicable']})")
                continue
            case = CASES[cid]
            records: list[dict[str, Any]] = []
            n_scored = 0
            for saved in x["turns"]:
                t = dict(saved, set=name, case=cid)
                turns.append(t)
                e = case.turns[t["turn"]].expect
                if e is not None:
                    q = reqs.get((cid, t["turn"]), [])
                    said = cx.said_in(q, []) if q else said_of.get((cid, n_scored), "")
                    n_scored += 1
                    used = set(t.get("native") or [])
                    ran = any(
                        c["tool"] in cx.RAN_CODE and c["status"] == "ok" for c in t.get("calls", [])
                    )
                    s = cx.score(
                        e, t.get("answer") or "", records, t["text"], said,
                        ran=ran or bool(used & approvals.SHELLS),
                        searched=bool(used & cx.WEB_SEARCH), queried=cx.queried(q),
                    )  # fmt: skip
                    tag = f"{name}:{cid}/t{t['turn']}"
                    counted = cid not in hit
                    if not counted:
                        down.append(tag)
                    elif s.get("correct") is not None:
                        strict_counted += 1
                        right_counted += bool(s["correct"])
                    if s.get("correct") is None:
                        unver += 1
                        unvers.append(f"{tag} {s.get('kind')}")
                    elif s["correct"]:
                        right += 1
                    else:
                        wrong += 1
                        why = {k: s.get(k) for k in ("missing", "made_up", "forbidden") if s.get(k)}
                        wrongs.append(f"{tag} {t['status']} {why}")
                    agree += s.get("correct") == (t.get("score") or {}).get("correct")
                records.append(t)
    secs = [t["seconds"] for t in turns]
    n = len(turns) or 1
    tot = {k: sum(t.get(k) or 0 for t in turns) for k in
           ("input_tokens", "cached_tokens", "output_tokens", "model_calls")}  # fmt: skip
    tool_calls = sum(len(t.get("calls") or []) + len(t.get("native") or []) for t in turns)
    statuses = collections.Counter(t["status"] for t in turns)
    errors = [f"{t['set']}:{t['case']}/t{t['turn']} {t['status']} {t.get('error')}"
              for t in turns if t["status"] not in ("SUCCESS",) or t.get("error")]  # fmt: skip
    checks_failed = [
        f"{t['set']}:{t['case']}/t{t['turn']} {c['check']}: {c.get('why')}"
        for t in turns
        for c in t.get("checks") or []
        if not c.get("ok")
    ]
    fresh = tot["input_tokens"] - tot["cached_tokens"]
    harness_usd = (fresh * PRICE.input + tot["cached_tokens"] * PRICE.cached_input
                   + tot["output_tokens"] * PRICE.output) / 1e6  # fmt: skip
    return {
        "run": d,
        "turns": len(turns),
        "strict": right + wrong,
        "right": right,
        "right_counted": right_counted,
        "strict_counted": strict_counted,
        "memory_down_turns": down,
        "wrong": wrong,
        "unverifiable": unver,
        "agrees_with_run": agree,
        "wrong_turns": wrongs,
        "unverifiable_turns": unvers,
        "not_applicable": na,
        "p50": pct(secs, 0.5),
        "p95": pct(secs, 0.95),
        "max": round(max(secs, default=0), 2),
        "mean": round(statistics.fmean(secs), 2) if secs else 0,
        "seconds_total": round(sum(secs), 1),
        "tokens": tot,
        "per_turn": {
            "input": round(tot["input_tokens"] / n),
            "cached": round(tot["cached_tokens"] / n),
            "output": round(tot["output_tokens"] / n),
            "model_calls": round(tot["model_calls"] / n, 2),
            "tool_calls": round(tool_calls / n, 2),
        },
        "tool_calls": tool_calls,
        "usd_run": round(usd, 6),
        "usd_harness_tokens": round(harness_usd, 6),
        "accounted": dict(accounted),
        "provider": provider(os.path.join(d, "meter.jsonl")),
        "statuses": dict(statuses),
        "errors": errors,
        "checks_failed": checks_failed,
    }


if __name__ == "__main__":
    for d in sys.argv[1:]:
        sys.stdout.write(json.dumps(analyse(d.rstrip("/"))) + "\n")
