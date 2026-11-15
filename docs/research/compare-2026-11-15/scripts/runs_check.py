"""Whether every turn's run, its events and its spend landed in agent-runs, and what the
memory service was asked each turn: read back from the stack (RUNS_URL, TRELLIS_API_KEY in
the environment; never printed) against the turns the run's saved rows hold.

Usage: python runs_check.py <run-dir>   (writes <run-dir>/services-check.json, prints a line)

For each case: the top-level runs that started (``*.services.json``, the run hook) against
its turns; for each run, its record (found, ended), its events (``RUN_STARTED``,
``RUN_FINISHED`` with ``spent``), and the spent tokens of its last ``RUN_FINISHED`` summed over
the case against the tokens the rows hold. Memory: the requests the harness made to the
memory service in each case, by kind (read: GET or a search/context POST; write: the rest),
and those that failed (an error status or no answer).
"""

from __future__ import annotations

import asyncio
import collections
import glob
import json
import os
import sys
from typing import Any

from analyse import loaded

from trellis.runs import RunsClient

ENDED = {"SUCCESS", "PARTIAL", "ERROR", "REJECTED", "CANCELLED", "TIMEOUT", "PAUSED"}
READS = ("search", "context", "recall", "query")
PAGE = 500


def kind_of(method: str, path: str) -> str:
    if method == "GET" or any(word in path for word in READS):
        return "read"
    return "write"


def load(d: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Each set's rows and service records, keyed ``<set>:<case>`` (a case may be in two
    sets: the long one and a small-window one)."""
    rows: dict[str, dict[str, Any]] = {}
    services: list[dict[str, Any]] = []
    for f in sorted(glob.glob(d + "/*.jsonl")):
        if "requests" in f or "replaced" in f or f.endswith("meter.jsonl"):
            continue
        name = os.path.basename(f)[:-6]
        with open(f) as fh:
            for line in fh:
                x = json.loads(line)
                if not x.get("not_applicable"):
                    rows[f"{name}:{x['case']}"] = x
        services += [s | {"case": f"{name}:{s['case']}"} for s in loaded(f[:-6] + ".services.json")]
    return rows, services


async def stored(ids: list[str]) -> dict[str, dict[str, Any]]:
    """Each run's record status, whether its stream started and finished, and the tokens its
    last ``RUN_FINISHED`` says it spent (a resumed run's carries the earlier attempt's too)."""
    out: dict[str, dict[str, Any]] = {}
    async with RunsClient() as client:
        for run_id in ids:
            record = await client.get(run_id)
            if record is None:
                out[run_id] = {"found": False}
                continue
            events: list[Any] = []
            after = 0
            while True:
                page = await client.events(run_id, after=after, limit=PAGE)
                events += page
                if len(page) < PAGE:
                    break
                after = page[-1].position
            types = [e.event.type.value for e in events]
            done = [e for e in events if e.event.type.value == "RUN_FINISHED"]
            spent = ((done[-1].event.data or {}).get("spent") or {}) if done else {}
            out[run_id] = {
                "found": True,
                "status": record.status.value,
                "started": "RUN_STARTED" in types,
                "finished": bool(done),
                "events": len(events),
                "tokens": int(spent.get("tokens") or 0),
            }
    return out


def per_turn(runs: list[dict[str, Any]], memory: list[dict[str, Any]]) -> tuple[int, int]:
    """How many turns read the memory context (``POST /v1/context``, answered) and wrote their
    transcript (``POST /v1/messages``, answered 2xx): per case, at most its turns — the reads
    and writes of a case counted against its turns (a read comes before its run's start hook,
    a transcript after its end: a turn's own window is not reliable)."""
    turns: collections.Counter[str] = collections.Counter()
    seen: set[str] = set()
    for s in runs:
        if s["run"] not in seen:
            seen.add(s["run"])
            turns[s["case"]] += 1
    read: collections.Counter[str] = collections.Counter()
    wrote: collections.Counter[str] = collections.Counter()
    for m in memory:
        ok = 200 <= (m.get("status") or 0) < 300
        read[m["case"]] += ok and m["path"] == "/v1/context"
        wrote[m["case"]] += ok and m["method"] == "POST" and m["path"] == "/v1/messages"
    return (
        sum(min(n, read[c]) for c, n in turns.items()),
        sum(min(n, wrote[c]) for c, n in turns.items()),
    )


def report(rows: dict[str, Any], services: list[dict[str, Any]], seen: dict[str, Any]) -> dict:
    turns = sum(len(x["turns"]) for x in rows.values())
    runs = [s for s in services if s["kind"] == "run" and s["parent"] is None and s["case"] in rows]
    memory = [s for s in services if s["kind"] == "memory" and s["case"] in rows]
    by_case: dict[str, list[str]] = collections.defaultdict(list)
    for s in runs:
        if s["run"] not in by_case[s["case"]]:
            by_case[s["case"]].append(s["run"])
    mismatch = []
    for case, x in rows.items():
        kept = sum(seen[r].get("tokens", 0) for r in by_case[case])
        made = sum((t.get("input_tokens") or 0) + (t.get("output_tokens") or 0) for t in x["turns"])
        if kept != made:
            mismatch.append(f"{case}: agent-runs {kept} vs rows {made}")
    problems = [
        f"{r}: {v}"
        for r, v in seen.items()
        if not v["found"] or v["status"] not in ENDED or not (v["started"] and v["finished"])
    ]
    kinds = collections.Counter(kind_of(s["method"], s["path"]) for s in memory)
    failed = [
        f"{s['case']} {s['method']} {s['path']} {s.get('status') or s.get('error')}"
        for s in memory
        if s.get("error") or (s.get("status") or 0) >= 400
    ]
    turn_reads, turn_writes = per_turn(runs, memory)
    touched = collections.defaultdict(set)
    for s in memory:
        touched[s["case"]].add(kind_of(s["method"], s["path"]))
    return {
        "turns": turns,
        "runs_unique": len(seen),
        "runs_found": sum(v["found"] for v in seen.values()),
        "runs_ended": sum(v.get("status") in ENDED for v in seen.values()),
        "runs_with_started_and_finished": sum(
            bool(v.get("started") and v.get("finished")) for v in seen.values()
        ),
        "events": sum(v.get("events", 0) for v in seen.values()),
        "cases_with_fewer_runs_than_turns": [
            c for c, x in rows.items() if len(by_case[c]) < len(x["turns"])
        ],
        "spend_token_mismatches": mismatch,
        "problems": problems,
        "memory_requests": len(memory),
        "memory_per_turn": round(len(memory) / max(turns, 1), 2),
        "memory_reads": kinds["read"],
        "memory_writes": kinds["write"],
        "memory_failed": failed,
        "cases_without_memory_read": [c for c in rows if "read" not in touched[c]],
        "cases_without_memory_write": [c for c in rows if "write" not in touched[c]],
        "turns_with_context_read": turn_reads,
        "turns_with_transcript_write": turn_writes,
        "context_reads": sum(1 for s in memory if s["path"] == "/v1/context"),
        "transcript_writes": sum(
            1 for s in memory if s["method"] == "POST" and s["path"] == "/v1/messages"
        ),
        "memory_paths": dict(
            collections.Counter(f"{s['method']} {s['path']}" for s in memory).most_common()
        ),
    }


if __name__ == "__main__":
    for d in sys.argv[1:]:
        rows, services = load(d.rstrip("/"))
        ids = list(
            dict.fromkeys(
                s["run"]
                for s in services
                if s["kind"] == "run" and s["parent"] is None and s["case"] in rows
            )
        )
        r = report(rows, services, asyncio.run(stored(ids)))
        with open(os.path.join(d, "services-check.json"), "w") as fh:
            json.dump(r, fh, indent=1)
        brief = {k: v for k, v in r.items() if k not in ("memory_paths", "problems")}
        sys.stdout.write(json.dumps(brief)[:2000] + "\n")
