"""Run benchmark.cost.complex with every reply the harness accounts to a run's budget
(budget.account) also written to $ACCOUNTED_OUT."""

import atexit
import json
import os
import sys
from pathlib import Path

from trellis.harness import budget

ROWS = []
_account = budget.account


def _accounting(runtime, **kw):
    _account(runtime, **kw)
    ROWS.append(
        {
            "run": runtime.run_id,
            "parent": getattr(getattr(runtime, "parent", None), "run_id", None),
            **{k: v for k, v in kw.items() if k != "model"},
        }
    )


budget.account = _accounting
atexit.register(lambda: Path(os.environ["ACCOUNTED_OUT"]).write_text(json.dumps(ROWS)))
from benchmark.cost.complex import main  # noqa: E402 - after the accounting is wrapped

sys.exit(main(sys.argv[1:]))
