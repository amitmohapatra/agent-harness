"""The run stores' contract (``tests/support/store_contract.py``) on the stores of this
process: ``LocalRuns`` and ``FileRuns``. agent-runs runs the same checks in
``tests/live/test_live_chat_order.py``."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from tests.support.shared_runs import FileRuns
from tests.support.store_contract import CHECKS
from trellis.harness.runs import LocalRuns, RunStore


@pytest.mark.parametrize("store", ["local", "file"])
@pytest.mark.parametrize("check", CHECKS, ids=lambda check: check.__name__)
async def test_a_store_keeps_the_contract(
    tmp_path: Path, store: str, check: Callable[[RunStore], Awaitable[None]]
) -> None:
    runs = LocalRuns() if store == "local" else FileRuns(tmp_path / "runs")
    await check(runs)
    await runs.aclose()
