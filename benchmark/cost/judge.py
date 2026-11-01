"""The big-model judge pass over a ``complex`` run (``docs/research/EVAL_PROTOCOL.md``): two
verdicts per scored turn — is the answer correct, and was the context sufficient (could the
answer be given from the model's input alone) — and a report that splits the failures into
harness gaps and model gaps.

    python -m benchmark.cost.judge packets --run build/cx --to build/cx.packets.jsonl
    # the judge writes one verdict line per packet into build/cx.verdicts.jsonl
    python -m benchmark.cost.judge report --packets build/cx.packets.jsonl \\
        --verdicts build/cx.verdicts.jsonl --out build/cx.judged

**A packet** (one JSON line per scored turn, shuffled; the run, model and arm are not in it —
they are in ``<to>.key.jsonl``, by ``packet``):

* ``packet``, ``prompt_version``, ``category``, ``tier``, ``kind``;
* ``model_input`` — every request the turn made, exactly as sent: ``messages`` (OpenAI form)
  and ``tools`` offered, in order (the last is the one the answer came from); a message longer
  than ``--max-chars`` is cut *for the judge only* and says so (``cut_for_judge``);
* ``answer`` — the answer under test;
* ``ground_truth`` — the value (or rubric) and how it was computed;
* ``strict`` — the strict scorer's verdict (``correct``, ``missing``, ``made_up``);
* ``harness_checks`` — the harness's own checks of the turn (``complex.checked``: how its run
  ended, what the model was and was not sent);
* ``verdict`` — the form the judge fills: ``answer_correct`` and ``context_sufficient``
  (true/false), ``missing`` (what the input lacked), ``made_up`` (claims the input does not
  support).

**A verdict** line: ``{"packet": ..., "answer_correct": bool, "context_sufficient": bool,
"missing": str, "made_up": [...], "judge": "a"}`` (``judge``: which judge pass; the 10% judged
twice are compared). Diagnosis: sufficient and correct — OK; sufficient and wrong — a model
gap; insufficient and wrong — a harness gap; insufficient and correct — right without the
evidence (a guess or outside knowledge: counted apart).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from benchmark.cost import stats

PROMPT_VERSION: Final = "complex-judge-v1"
#: The fixed instructions every packet carries (versioned by :data:`PROMPT_VERSION`).
PROMPT: Final = """You are judging one turn of an assistant. Use ONLY `model_input` (the exact
requests the assistant's model was sent: messages and the tools offered) — no outside knowledge,
and no tool beyond the results already in the messages.
1. answer_correct: does `answer` match `ground_truth` (numbers within the stated rounding; every
   field; for a rubric, every point)? List in `made_up` any claim the input does not support.
2. context_sufficient: answer the user's last question yourself from `model_input` alone. If you
   can reach the ground truth from it (computing is allowed), true; if something needed is not
   there (a result cut off or missing, an earlier turn or conversation not shown, a stale value
   with no fresh one, a needed tool not offered), false — and say what in `missing`.
A message marked cut_for_judge was cut for you, not for the model: judge as if it were whole.
Return one JSON line: {"packet", "answer_correct", "context_sufficient", "missing", "made_up"}."""
#: The share of packets a second judge pass also judges, for agreement.
DOUBLE: Final = 0.10
SEED: Final = 11


def _lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _cut(messages: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        content = m.get("content")
        if max_chars and isinstance(content, str) and len(content) > max_chars:
            out.append(
                m
                | {
                    "content": content[:max_chars]
                    + f"\n[cut for the judge: {len(content) - max_chars} more characters]",
                    "cut_for_judge": True,
                }
            )
            continue
        out.append(m)
    return out


def packets(run: str, *, max_chars: int = 60_000, arm: str = "") -> tuple[list[dict], list[dict]]:
    """The judge packets of a run (``<run>.jsonl`` and ``<run>.requests.jsonl``) and their key."""
    rows = _lines(Path(run).with_suffix(".jsonl"))
    asked = _lines(Path(f"{run}.requests.jsonl"))
    by_turn: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for q in asked:
        by_turn.setdefault((q.get("model", ""), q["case"], q["turn"]), []).append(q)
    out, key = [], []
    for row in rows:
        for t in row["turns"]:
            if not t.get("expect"):
                continue
            requests = sorted(
                by_turn.get((row.get("model", ""), row["case"], t["turn"]), []),
                key=lambda q: q["call"],
            )
            ident = f"{arm}|{row.get('model')}|{row['history']}|{row['case']}|{t['turn']}"
            packet = hashlib.sha256(ident.encode()).hexdigest()[:12]
            e = t["expect"]
            out.append(
                {
                    "packet": packet,
                    "prompt_version": PROMPT_VERSION,
                    "category": row["category"],
                    "tier": row["tier"],
                    "kind": e["kind"],
                    "question": t["text"],
                    "model_input": [
                        {"messages": _cut(q["messages"], max_chars), "tools": q.get("tools")}
                        for q in requests
                    ],
                    "answer": t["answer"],
                    "status": t["status"],
                    "ground_truth": {
                        "value": e["value"],
                        "tolerance": e["tol"],
                        "forbidden": e["forbidden"],
                        "rubric": e["rubric"],
                        "computed": e["truth"],
                        "of_turn_calls": (
                            row["turns"][e["of_turn"]]["calls"]
                            if e.get("of_turn") is not None
                            else None
                        ),
                    },
                    "strict": t.get("score"),
                    # the harness's own checks of the turn (how it ended, what the model was
                    # sent): what the context lacked, before the judge looks
                    "harness_checks": t.get("checks", []),
                    "verdict": {
                        "answer_correct": None,
                        "context_sufficient": None,
                        "missing": "",
                        "made_up": [],
                    },
                }
            )
            key.append(
                {"packet": packet, "arm": arm, "model": row.get("model"), "history": row["history"],
                 "case": row["case"], "turn": t["turn"]}
            )  # fmt: skip
    random.Random(SEED).shuffle(out)
    doubled = {p["packet"] for p in out[: max(1, round(len(out) * DOUBLE))]} if out else set()
    for p in out:
        p["double_judge"] = p["packet"] in doubled
    return out, key


def _order(category: str) -> tuple[int, str]:
    from benchmark.cost.complex import CATEGORIES

    return (CATEGORIES.index(category) if category in CATEGORIES else len(CATEGORIES), category)


def diagnosis(correct: bool, sufficient: bool) -> str:
    if sufficient:
        return "ok" if correct else "model_gap"
    return "unsupported_correct" if correct else "harness_gap"


def report(packet_rows: Sequence[dict], verdicts: Sequence[dict]) -> dict[str, Any]:
    """Per packet its diagnosis, and the counts overall, by category and by tier; with two
    judges' verdicts on a packet, their agreement (the first pass is the one counted)."""
    first: dict[str, dict] = {}
    second: dict[str, dict] = {}
    for v in verdicts:
        (second if v["packet"] in first else first)[v["packet"]] = v
    judged = []
    for p in packet_rows:
        v = first.get(p["packet"])
        if v is None or v.get("answer_correct") is None or v.get("context_sufficient") is None:
            continue
        strict = (p.get("strict") or {}).get("correct")
        judged.append(
            {
                "packet": p["packet"],
                "category": p["category"],
                "tier": p["tier"],
                "kind": p["kind"],
                "strict": strict,
                "answer_correct": bool(v["answer_correct"]),
                "context_sufficient": bool(v["context_sufficient"]),
                "diagnosis": diagnosis(bool(v["answer_correct"]), bool(v["context_sufficient"])),
                "missing": v.get("missing") or "",
                "made_up": v.get("made_up") or [],
                "strict_disagrees": strict is not None and strict != bool(v["answer_correct"]),
            }
        )

    def cell(rows: Sequence[dict]) -> dict[str, Any]:
        found = {
            d: sum(r["diagnosis"] == d for r in rows)
            for d in ("ok", "model_gap", "harness_gap", "unsupported_correct")
        }
        return {
            "n": len(rows),
            "correct": stats.interval([float(r["answer_correct"]) for r in rows]),
            "sufficient": stats.interval([float(r["context_sufficient"]) for r in rows]),
            "strict": stats.interval([float(r["strict"]) for r in rows if r["strict"] is not None]),
            "made_up": stats.interval([float(bool(r["made_up"])) for r in rows]),
            **found,
        }

    both = [k for k in second if k in first]
    agree = [
        first[k]["answer_correct"] == second[k]["answer_correct"]
        and first[k]["context_sufficient"] == second[k]["context_sufficient"]
        for k in both
    ]
    return {
        "prompt_version": PROMPT_VERSION,
        "packets": len(packet_rows),
        "judged": len(judged),
        "all": cell(judged),
        "by_category": {
            c: cell([r for r in judged if r["category"] == c])
            for c in sorted({r["category"] for r in judged}, key=_order)
        },
        "by_tier": {
            str(t): cell([r for r in judged if r["tier"] == t])
            for t in sorted({r["tier"] for r in judged})
        },
        "agreement": {
            "double_judged": len(both),
            "agree": stats.interval([float(a) for a in agree]),
        },
        "harness_gaps": [r for r in judged if r["diagnosis"] == "harness_gap"],
        "model_gaps": [r for r in judged if r["diagnosis"] == "model_gap"],
        "strict_disagreements": [r["packet"] for r in judged if r["strict_disagrees"]],
        "rows": judged,
    }


def markdown(found: dict[str, Any]) -> str:
    lines = [
        f"# complex: judged ({found['judged']} of {found['packets']} packets, "
        f"{found['prompt_version']})",
        "",
        "| group | n | correct | context sufficient | strict | OK | model gap | harness gap "
        "| right without evidence |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    groups = (
        [("all", found["all"])]
        + list(found["by_category"].items())
        + [(f"tier {k}", v) for k, v in found["by_tier"].items()]
    )
    for name, c in groups:
        lines.append(
            f"| {name} | {c['n']} | {stats.shown(c['correct'], 2)} "
            f"| {stats.shown(c['sufficient'], 2)} | {stats.shown(c['strict'], 2)} | {c['ok']} "
            f"| {c['model_gap']} | {c['harness_gap']} "
            f"| {c['unsupported_correct']} |"
        )
    a = found["agreement"]
    lines += [
        "",
        f"Double-judged: {a['double_judged']}, agreement {stats.shown(a['agree'], 2)}.",
        "",
    ]
    if found["harness_gaps"]:
        lines += ["## Harness gaps (the input lacked what the answer needs)", ""]
        lines += [
            f"- `{r['packet']}` {r['category']} (tier {r['tier']}): {r['missing'] or '?'}"
            for r in found["harness_gaps"]
        ]
        lines.append("")
    if found["model_gaps"]:
        lines += ["## Model gaps (the input was enough; the answer was wrong)", ""]
        lines += [
            f"- `{r['packet']}` {r['category']} (tier {r['tier']})" for r in found["model_gaps"]
        ]
    return "\n".join(lines) + "\n"


def _write(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m benchmark.cost.judge",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)
    pk = sub.add_parser("packets", help="write the judge packets of a run")
    pk.add_argument("--run", required=True, help="the run's --out (its .jsonl and .requests.jsonl)")
    pk.add_argument("--to", required=True, help="the packets' JSONL (the key goes beside it)")
    pk.add_argument("--arm", default="", help="the arm's name, kept in the key only")
    pk.add_argument(
        "--max-chars", type=int, default=60_000, help="cut longer messages for the judge (0: never)"
    )
    rp = sub.add_parser("report", help="the diagnosis report from the judge's verdicts")
    rp.add_argument("--packets", required=True)
    rp.add_argument("--verdicts", required=True)
    rp.add_argument("--out", required=True, help="writes <out>.json and <out>.md")
    args = p.parse_args(argv)
    if args.command == "packets":
        out, key = packets(args.run, max_chars=args.max_chars, arm=args.arm)
        to = Path(args.to)
        _write(to, out)
        _write(to.with_suffix(".key.jsonl"), key)
        (to.with_suffix(".prompt.txt")).write_text(f"{PROMPT_VERSION}\n\n{PROMPT}\n")
        sys.stdout.write(f"{len(out)} packets -> {to}\n")
        return 0
    found = report(_lines(Path(args.packets)), _lines(Path(args.verdicts)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(found, indent=2, default=str))
    out.with_suffix(".md").write_text(markdown(found))
    sys.stdout.write(markdown(found))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
