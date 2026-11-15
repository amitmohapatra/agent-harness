# ruff: noqa: T201, PLC0415 - a research script: it reports with print, imports an SDK where a case needs it
"""Check what landed in agent-runs and the memory service against what each stack case saw,
and build the per-SDK tables (``data/summary.json``, ``data/tables.md``).

Per run (and each child run): a final status that matches the result; ``RUN_STARTED`` and
``RUN_FINISHED`` events; as many ``TOOL_CALL_RESULT`` events as recorded tool invocations; the
question as the first ``USER`` message and, for a success with an answer, an ``ASSISTANT``
message; a ``tool_call`` feedback row for each approval asked; children ended.

Usage: python check.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parent.parent / "data"
FINAL = {"SUCCESS", "ERROR", "PARTIAL", "CANCELLED", "TIMEOUT"}
#: the memory tools: their calls are memories, not tool records (``Agent.record_tool``)
PULL = {
    "memory_search",
    "memory_remember",
    "memory_update",
    "memory_forget",
    "profile_edit",
    "tool_search",
}
#: a call's result event that is no call made: a pause, a rejection, a call cut by the cancel
NOT_MADE = {"paused", "rejected", "cancelled"}


def made_calls(run_id: str) -> int:
    """The calls a run's log says were made: its ``TOOL_CALL_RESULT`` events, less pauses,
    rejections and the memory tools (read from agent-runs' API)."""
    from common import _runs_api

    return sum(
        1
        for e in _runs_api(f"/v1/runs/{run_id}/events?limit=500")
        if e["event"]["type"] == "TOOL_CALL_RESULT"
        and (e["event"].get("data") or {}).get("status") not in NOT_MADE
        and (e["event"].get("data") or {}).get("tool") not in PULL
    )


def run_problems(rec: dict[str, Any], *, status: str | None, asked: int) -> list[str]:
    out: list[str] = []
    run = rec.get("run") or {}
    rid = run.get("run_id")
    if run.get("status") not in FINAL:
        out.append(f"run {rid} not ended: {run.get('status')}")
    if status and not status.endswith(str(run.get("status"))):
        out.append(f"run {rid}: result {status} but record {run.get('status')}")
    for kid in rec.get("children") or []:
        if kid["status"] not in FINAL:
            out.append(f"child {kid['run_id']} not ended: {kid['status']}")
    kids = {k["run_id"] for k in rec.get("children") or []}
    for r, ev in (rec.get("events") or {}).items():
        if r in kids and not ev:
            continue  # a helper's run: its calls are on its parent's log (counted below)
        if not ev.get("RUN_STARTED") or not ev.get("RUN_FINISHED"):
            out.append(f"{r}: events lack RUN_STARTED/RUN_FINISHED: {ev}")
    # every call made, on the family's logs, is one tool record (under whichever run made it)
    made = sum(made_calls(r) for r in (rec.get("events") or {}))
    tools = sum(len(v) for v in (rec.get("tools") or {}).values())
    if made != tools:
        out.append(f"{rid}: {made} calls made on the logs, {tools} tool records")
    msgs = (rec.get("messages") or {}).get(rid, [])
    if not msgs or msgs[0]["role"] != "USER":
        out.append(f"run {rid}: no USER message first in memory ({len(msgs)} messages)")
    if run.get("status") == "SUCCESS" and not any(
        m["role"] == "ASSISTANT" and m["content"] for m in msgs
    ):
        out.append(f"run {rid}: a success with no ASSISTANT text in memory")
    approvals = [f for f in rec.get("feedback") or [] if f["target_kind"] == "tool_call"]
    if asked and len(approvals) < asked:
        out.append(f"run {rid}: {asked} questions asked, {len(approvals)} tool_call feedback rows")
    return out


def checked(case: dict[str, Any]) -> list[str] | None:
    recs = case.get("records")
    if not recs:
        return None
    recs = recs if isinstance(recs, list) else [recs]
    statuses = case.get("statuses") or [case.get("status")] * len(recs)
    given = case.get("asked")
    given = given if isinstance(given, list) else []
    asked = len([a for a in given if isinstance(a, dict) and a.get("tool_call")])
    problems: list[str] = []
    for i, rec in enumerate(recs):
        st = statuses[i] if i < len(statuses) else None
        problems += run_problems(
            rec, status=str(st) if st else None, asked=asked if i == len(recs) - 1 else 0
        )
    return problems


def main() -> None:
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(DATA.glob("*/*.json")):
        case = json.loads(path.read_text())
        feature, arm = case["feature"], case["arm"]
        row = summary.setdefault(case["sdk"], {}).setdefault(feature, {})
        cell: dict[str, Any] = {
            "ok": case.get("ok"),
            "error": case.get("error"),
            "usd": case["spend"]["usd"],
            "file": str(path.relative_to(DATA)),
        }
        if arm == "stack":
            problems = checked(case)
            cell["records_ok"] = None if problems is None else not problems
            cell["record_problems"] = problems
        row[arm] = cell
    (DATA / "summary.json").write_text(json.dumps(summary, indent=1))
    lines: list[str] = []
    mark = {True: "pass", False: "FAIL", None: "-"}
    for sdk, feats in summary.items():
        lines += [
            f"### {sdk}",
            "",
            "| feature | bare | off (same?) | stack (records) | on |",
            "|---|---|---|---|---|",
        ]
        for f, arms in feats.items():
            b, o, s, n = (arms.get(a) for a in ("bare", "off", "stack", "on"))

            def c(x: dict[str, Any] | None) -> str:
                if x is None:
                    return "-"
                return "error" if x["error"] and x["ok"] is None else mark[bool(x["ok"])]

            same = (
                "" if b is None or o is None else (" =" if bool(b["ok"]) == bool(o["ok"]) else " ≠")
            )
            rec = "" if s is None else f" ({mark[s.get('records_ok')]})"
            lines.append(f"| {f} | {c(b)} | {c(o)}{same} | {c(s)}{rec} | {c(n)} |")
        lines.append("")
    (DATA / "tables.md").write_text("\n".join(lines))
    total = sum(
        c["usd"] for feats in summary.values() for arms in feats.values() for c in arms.values()
    )
    print(
        f"cases: {sum(len(a) for f in summary.values() for a in f.values())}, "
        f"usd (last run of each case): {total:.4f}"
    )
    for sdk, feats in summary.items():
        for f, arms in feats.items():
            for arm, cell in arms.items():
                if cell.get("record_problems"):
                    print(sdk, f, arm, cell["record_problems"][:3])


if __name__ == "__main__":
    main()
