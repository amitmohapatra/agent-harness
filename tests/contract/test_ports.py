"""Every implementation the harness ships satisfies the contracts port it stands for, and
the public API is exactly the documented one."""

from __future__ import annotations

import subprocess
import sys

from trellis.contracts import ArtifactClient, Judge, RunStore, TelemetryRedactor

import trellis
from trellis.eval import GroundedJudge, JudgeBudget
from trellis.harness.artifacts import Artifacts
from trellis.harness.clients.runs import HttpRuns, LocalRuns
from trellis.harness.redaction import Redactor


def test_the_run_stores_are_run_stores() -> None:
    assert isinstance(LocalRuns(), RunStore)
    assert isinstance(HttpRuns("http://runs", None), RunStore)


def test_the_rest_of_the_ports() -> None:
    assert isinstance(Artifacts(), ArtifactClient)
    assert isinstance(Redactor(), TelemetryRedactor)
    assert isinstance(GroundedJudge(budget=JudgeBudget(0.1)), Judge)


def test_the_public_api_is_the_documented_one() -> None:
    assert trellis.__all__ == [
        "Agent",
        "Harness",
        "ReAct",
        "Result",
        "RunHandle",
        "Runtime",
        "Settings",
        "a2a",
        "current",
        "mcp",
        "openapi",
        "tool",
    ]
    for name in trellis.__all__:
        assert getattr(trellis, name) is not None


def test_importing_trellis_loads_no_framework_and_contracts_stay_cheap() -> None:
    code = (
        "import sys, trellis.contracts, trellis.memory\n"
        "assert 'trellis.harness' not in sys.modules\n"
        "from trellis import Harness, mcp, tool, a2a, openapi, ReAct, current\n"
        "Harness()\n"
        "loaded = {m.split('.')[0] for m in sys.modules}\n"
        "assert not loaded & {'langgraph', 'langchain_core', 'agents', 'claude_agent_sdk', 'a2a', 'fastapi'}, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env={"PATH": ""})
