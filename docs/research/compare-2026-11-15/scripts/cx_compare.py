"""Run benchmark.cost.complex, also writing (1) every reply the harness accounts to a run's
budget (``budget.account``) to $ACCOUNTED_OUT, each with its case; (2) for a CLI adapter,
whose requests the suite cannot see, what each scored turn's scorer read as its context
(``said_in``: what the harness gave the run) to $SAID_OUT, so the turn can be re-scored from
the saved files; (3) every run that started (its id, parent, thread, time) and every request
the harness made to the memory service (method, path with ids folded, status or error, time)
to $SERVICES_OUT, for the memory and agent-runs checks. Nothing the run does is changed."""

import atexit
import json
import os
import re
import sys
import time
from pathlib import Path

from benchmark.cost import complex as cx
from trellis.harness import budget
from trellis.memory import transport

ACCOUNTED: list[dict] = []
SAID: list[dict] = []
SERVICES: list[dict] = []
ID = re.compile(r"/(?=[^/]*\d)[^/]{8,}")
CASE: list[str] = [""]
_account = budget.account
_said_in = cx.said_in
_run_case = cx.run_case
_perform = transport.Transport._perform
_on_run_start = cx.Trace.on_run_start


async def _performed(self, method, path, **kw):
    row = {"kind": "memory", "case": CASE[0], "t": round(time.time(), 3), "method": method}
    row["path"] = ID.sub("/{id}", path)
    SERVICES.append(row)
    try:
        response = await _perform(self, method, path, **kw)
    except Exception as exc:
        row["error"] = type(exc).__name__
        row["said"] = str(exc)[:300]
        raise
    row["status"] = response.status_code
    return response


async def _started(self, run):
    SERVICES.append(
        {
            "kind": "run",
            "case": CASE[0],
            "t": round(time.time(), 3),
            "run": run.run_id,
            "parent": getattr(getattr(run, "parent", None), "run_id", None),
            "thread": run.thread,
        }
    )
    await _on_run_start(self, run)


def _accounting(runtime, **kw):
    _account(runtime, **kw)
    ACCOUNTED.append(
        {
            "t": round(time.time(), 3),
            "case": CASE[0],
            "run": runtime.run_id,
            "parent": getattr(getattr(runtime, "parent", None), "run_id", None),
            **{k: v for k, v in kw.items() if k != "model"},
        }
    )


def _said(requests, told):
    out = _said_in(requests, told)
    if not requests:
        SAID.append({"case": CASE[0], "n": sum(s["case"] == CASE[0] for s in SAID), "said": out})
    return out


async def _case(h, case, **kw):
    CASE[0] = case.id
    return await _run_case(h, case, **kw)


def _flush():
    Path(os.environ["ACCOUNTED_OUT"]).write_text(json.dumps(ACCOUNTED))
    if os.environ.get("SERVICES_OUT"):
        Path(os.environ["SERVICES_OUT"]).write_text(json.dumps(SERVICES))
    if os.environ.get("SAID_OUT"):
        Path(os.environ["SAID_OUT"]).write_text(json.dumps(SAID))


budget.account = _accounting
cx.said_in = _said
cx.run_case = _case
transport.Transport._perform = _performed
cx.Trace.on_run_start = _started
atexit.register(_flush)

sys.exit(cx.main(sys.argv[1:]))
