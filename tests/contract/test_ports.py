"""The harness's implementations of contracts ports, its run store is the agent-runs SDK's
client (or the in-process one), and the public API is exactly the documented one."""

from __future__ import annotations

import subprocess
import sys

import pytest

import trellis
from trellis import Harness, Settings
from trellis.contracts import TelemetryRedactor
from trellis.harness.redaction import Redactor
from trellis.harness.runs import LocalRuns, RunStore
from trellis.runs import RunsClient, WorkerStore


def test_the_redactor_is_the_contracts_redactor() -> None:
    assert isinstance(Redactor(), TelemetryRedactor)


async def test_the_runs_client_and_the_local_store_are_run_stores() -> None:
    client = RunsClient("http://runs.test", api_key="key")
    local = LocalRuns()
    # pyright checks these assignments: the signatures are the SDK's, so either one is the
    # harness's run store, and the worker loop's store
    stores: list[RunStore] = [client, local]
    workers: list[WorkerStore] = [client, local]
    assert all(isinstance(store, RunStore) for store in stores) and len(workers) == 2
    assert not isinstance(object(), RunStore)
    await client.aclose()


async def test_the_harness_runs_on_agent_runs_when_it_is_configured() -> None:
    assert isinstance(Harness(config=Settings()).runs, LocalRuns)
    remote = Settings(memory_url="http://memory.test", runs_url="http://runs.test", api_key="k")
    async with Harness(config=remote) as h:
        assert isinstance(h.runs, RunsClient)


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
        "import sys, trellis.contracts, trellis.memory, trellis.runs\n"
        "assert 'trellis.harness' not in sys.modules\n"
        "from trellis import Harness, tool, a2a, openapi, ReAct, current\n"
        "Harness()\n"
        "assert not {'trellis.harness.a2a', 'trellis.harness.agui'} & set(sys.modules)\n"
        "loaded = {m.split('.')[0] for m in sys.modules}\n"
        "assert not loaded & {'langgraph', 'langchain_core', 'agents', 'claude_agent_sdk', 'a2a', 'fastapi'}, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env={"PATH": ""})
