"""The user-scenario suite: the harness against the real memory service and the real agent-runs
(with its ticker), each in a process of its own, every model scripted (``tests/scenarios/brains.py``):
nothing but the model is a stand-in, and no model is paid for.

``make test-scenarios`` (``pytest -m scenario tests/scenarios``) runs it with the deployment's
variables: ``MEMORY_URL``, ``RUNS_URL``, ``TRELLIS_API_KEY`` (a service key of the tenant),
``TRELLIS_ADMIN_KEY`` (its administrator's), ``TRELLIS_API_KEY_OTHER`` (a service key of a second
tenant: the tenant boundary). Without them every test is skipped, never failed.
``scripts/scenarios/stack.sh`` starts such a stack on this machine (its own databases, its own
keys) and runs the suite against it.

Every test's teardown checks the test plan's invariants INV-1..INV-10
(``tests/support/invariants.py``): among them INV-7, every request the harness sent the two
services and every answer against the services' committed OpenAPI documents, as it happened —
the real path, no stand-in in between: scenario 10 is every other scenario's traffic.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from trellis import Harness, Settings
from trellis.memory import MemoryClient

#: How long a memory call may take here (the services share one busy development machine).
TIMEOUT: Final = 60.0


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _reachable(url: str | None) -> bool:
    if url is None:
        return False
    try:
        return httpx.get(url.rstrip("/") + "/health/live", timeout=3.0).status_code < 500
    except httpx.HTTPError:
        return False


MEMORY_URL = _env("MEMORY_URL")
RUNS_URL = _env("RUNS_URL")
API_KEY = _env("TRELLIS_API_KEY")
OTHER_KEY = _env("TRELLIS_API_KEY_OTHER")
UP = API_KEY is not None and _reachable(MEMORY_URL) and _reachable(RUNS_URL)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    here = Path(__file__).parent
    skip = pytest.mark.skip(
        reason="needs the memory service and agent-runs (MEMORY_URL, RUNS_URL, TRELLIS_API_KEY)"
    )
    for item in items:
        if Path(str(item.fspath)).is_relative_to(here):
            item.add_marker(pytest.mark.scenario)
            named = PLAN.get(getattr(item, "originalname", item.name), ())
            ids = [*named, *(i for m in item.iter_markers("plan") for i in m.args)]
            item.user_properties.append(("plan", " ".join(dict.fromkeys(ids))))
            if not UP:
                item.add_marker(skip)


#: the test plan's ids (docs/testing/TEST_PLAN.md) of the tests that do not say theirs with
#: ``@pytest.mark.plan``
PLAN: dict[str, tuple[str, ...]] = {
    "test_messages_one_after_another": ("CONV-01", "MEM-01"),
    "test_a_burst_of_messages": ("CONV-02",),
    "test_a_message_to_a_busy_chat": ("CONV-01", "CONV-04", "CONV-05"),
    "test_three_chats_at_once_stay_apart": ("CONV-07",),
    "test_recall_across_chats_is_user_level": ("CONV-07", "MEM-02"),
    "test_two_users_on_one_thread_id": ("CONV-08", "MEM-03"),
    "test_two_users_approvals_files_schedules_and_runs_stay_apart": ("CONV-08", "MEM-03", "SEC-01"),
    "test_a_pause_while_another_chat_works": ("APR-01", "APR-07", "MEM-08", "CONV-09"),
    "test_always_allow_in_one_chat_applies_in_another": ("APR-03",),
    "test_one_question_per_call_under_concurrency": ("APR-02",),
    "test_resumed_on_another_worker": ("APR-05",),
}


def settings(**changes: object) -> Settings:
    return Settings(**{**Settings.from_env().model_dump(), **changes})


def harness(*, key: str | None = None, memory: bool = True, **changes: object) -> Harness:
    """A harness for the environment's deployment (``key``: another tenant's service key)."""
    if key is not None:
        changes["api_key"] = key
    s = settings(**changes)
    client: Any = (
        MemoryClient(s.memory_url, api_key=s.api_key, timeout=TIMEOUT)
        if memory and s.memory_url is not None
        else (False if not memory else None)
    )
    return _Owned(config=s, memory=client)


class _Owned(Harness):
    """A harness that closes the memory client it was given (a deployment's would be its own)."""

    async def aclose(self) -> None:
        await super().aclose()
        if self.memory is not None:
            await self.memory.client.aclose()


@pytest.fixture
def uid() -> str:
    """A name of this test's own: the services keep everything across tests."""
    return uuid.uuid4().hex[:8]


@pytest.fixture
async def h() -> AsyncIterator[Harness]:
    async with harness() as made:
        yield made


# --------------------------------------------------------------------------- the invariants
#: INV-1..INV-10 at every scenario's teardown (INV-7 is the contract: every exchange with the
#: services checked against their OpenAPI documents as it happens)
from tests.support.invariants import invariants  # noqa: E402,F401
