"""The harness's implementations of contracts ports, and the public API is exactly the
documented one. (The run stores are not the contracts ``RunStore``: agent-runs lists runs as
summaries, ``Runs.inbox``, where the port still lists whole records.)"""

from __future__ import annotations

import subprocess
import sys

import pytest

import trellis
from trellis.contracts import TelemetryRedactor
from trellis.harness.redaction import Redactor


def test_the_redactor_is_the_contracts_redactor() -> None:
    assert isinstance(Redactor(), TelemetryRedactor)


def test_the_public_api_is_the_documented_one() -> None:
    assert trellis.__all__ == [
        "Agent",
        "EvalCase",
        "EvalItem",
        "EvalReport",
        "EvalResult",
        "EvalScore",
        "Evaluator",
        "Harness",
        "ReAct",
        "Result",
        "RunHandle",
        "RunSummary",
        "Runtime",
        "Settings",
        "a2a",
        "contains",
        "current",
        "exact_match",
        "grounding",
        "llm_judge",
        "openapi",
        "tool",
    ]
    for name in trellis.__all__:
        assert getattr(trellis, name) is not None
    assert dir(trellis) == trellis.__all__
    with pytest.raises(AttributeError, match="no attribute 'Missing'"):
        trellis.Missing  # noqa: B018


def test_importing_trellis_loads_no_framework_and_contracts_stay_cheap() -> None:
    code = (
        "import sys, trellis.contracts, trellis.memory\n"
        "assert 'trellis.harness' not in sys.modules\n"
        "from trellis import Harness, tool, a2a, openapi, ReAct, current\n"
        "Harness()\n"
        "loaded = {m.split('.')[0] for m in sys.modules}\n"
        "assert not loaded & {'langgraph', 'langchain_core', 'agents', 'claude_agent_sdk', 'a2a', 'fastapi'}, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env={"PATH": ""})
