"""python todos_report.py <run dir>...: one JSON line per run of ``sweep.sh``, read from its
``run.jsonl`` and ``run.requests.jsonl``: the strict score and each wrong turn, whether
``write_todos`` was offered on every request, the turns that called it, whether each later
turn's request sees the todo list (a ``write_todos`` call in its messages, or the harness's
note), and the tokens, model calls, seconds and dollars."""

import json
import sys
from pathlib import Path
from typing import Any

NOTE = "Your todo list (write_todos)"


def names(tools: list[dict[str, Any]] | None) -> set[str]:
    return {t["function"]["name"] if "function" in t else t.get("name", "") for t in tools or []}


def todo_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        call
        for m in messages
        if m.get("role") == "assistant"
        for call in m.get("tool_calls") or []
        if call["function"]["name"] == "write_todos"
    ]


def noted(messages: list[dict[str, Any]]) -> bool:
    return any(NOTE in json.dumps(m.get("content")) for m in messages if m.get("role") == "user")


def report(d: Path) -> dict[str, Any]:
    rows = [json.loads(x) for x in (d / "run.jsonl").read_text().splitlines()]
    reqs = [json.loads(x) for x in (d / "run.requests.jsonl").read_text().splitlines()]
    turns = [(x["case"], t) for x in rows for t in x["turns"]]
    scored = [(c, t) for c, t in turns if (t.get("score") or {}).get("correct") is not None]
    wrong: list[Any] = [
        (c, t["turn"], {k: v for k, v in t["score"].items() if v and k != "kind"}, t["answer"])
        for c, t in scored
        if not t["score"]["correct"]
    ]
    wrong += [(c, t["turn"], k["check"]) for c, t in turns for k in t["checks"] if not k["ok"]]
    # a write_todos call is made in the turn of the first request whose messages hold it
    made: dict[tuple[str, str], int] = {}
    for q in reqs:
        for call in todo_calls(q["messages"]):
            made.setdefault((q["case"], call["id"]), q["turn"])
    used = sorted({(case, turn) for (case, _), turn in made.items()})
    first: dict[str, int] = {}
    for case, turn in used:
        first[case] = min(first.get(case, turn), turn)
    later = [q for q in reqs if q["case"] in first and q["turn"] > first[q["case"]]]
    by_call = [q for q in later if todo_calls(q["messages"])]
    by_note = [q for q in later if not todo_calls(q["messages"]) and noted(q["messages"])]
    unseen = [q for q in later if q not in by_call and q not in by_note]
    offered = sum("write_todos" in names(q["tools"]) for q in reqs)
    return {
        "run": d.name,
        "strict": f"{sum(t['score']['correct'] for _, t in scored)}/{len(scored)}",
        "wrong": wrong,
        "write_todos_offered": f"{offered}/{len(reqs)}",
        "turns_calling_write_todos": used,
        "later_requests": len(later),
        "list_seen_by_call": len(by_call),
        "list_seen_by_note": len(by_note),
        "list_not_seen": [(q["case"], q["turn"], q["call"]) for q in unseen],
        "input_tokens": sum(t["input_tokens"] for _, t in turns),
        "output_tokens": sum(t["output_tokens"] for _, t in turns),
        "model_calls": sum(t["model_calls"] for _, t in turns),
        "seconds": round(sum(t["seconds"] for _, t in turns), 1),
        "usd": round(sum(x.get("usd", 0) for x in rows), 4),
    }


if __name__ == "__main__":
    for path in sys.argv[1:]:
        sys.stdout.write(json.dumps(report(Path(path)), default=str) + "\n")
