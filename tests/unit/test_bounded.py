"""Every piece of the harness's own client-side state is bounded (invariant I9): per tenant
(governance, toolboxes, sandbox reaping), per tool (the catalog's rules, what was published),
per prompt and skill, per thread outside a run, per model name said once, the h.tools builds
and the spool on disk — a long-lived process serving many tenants, tools and threads keeps a
bounded amount of each, the least recently used forgotten (read again, sent again, said
again)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from tests.support.gateway import FakeGateway
from trellis import Harness, Settings
from trellis.contracts import ToolSpec
from trellis.harness import budget, react, writes
from trellis.harness.bounded import LRU, SAID_ONCE, LRUSet
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Published, Rule, Rules
from trellis.harness.middleware import RunCheckpointer
from trellis.harness.writes import SPOOL_FILE, Writes


def test_an_lru_forgets_the_least_recently_used_past_its_bound() -> None:
    lru: LRU[str, int] = LRU(2)
    lru["a"], lru["b"] = 1, 2
    assert lru["a"] == 1  # a used: b is now the oldest
    lru["c"] = 3
    assert list(lru) == ["a", "c"] and "b" not in lru and len(lru) == 2
    assert lru.get("b") is None and lru.get("a") == 1 and lru.values() == [3, 1]
    assert lru.setdefault("d", lambda: 4) == 4 and lru.setdefault("d", lambda: 5) == 4
    assert lru.pop("d") == 4 and lru.pop("zz") is None
    with pytest.raises(ValueError, match="at least one"):
        LRU(0)
    seen = LRUSet(["x", "y", "z"], maxsize=2)
    assert list(seen) == ["y", "z"] and "x" not in seen and len(seen) == 2
    seen.discard("y")
    assert list(seen) == ["z"]


async def test_per_tenant_state_is_bounded() -> None:
    async with Harness(config=Settings()) as h:
        h._governances = LRU(3)
        for n in range(10):
            h._governance(f"t{n}")
        assert len(h._governances) == 3 and "t9" in h._governances

        async def answer(input: str, agent: Any) -> str:
            return input

        agent = h.wrap(answer, id="a")
        agent._toolboxes = LRU(3)
        for n in range(10):
            agent._toolbox(f"t{n}")
        assert len(agent._toolboxes) == 3


class Catalog:
    def __init__(self) -> None:
        self.read_names: list[list[str]] = []
        self.published: list[list[dict[str, object]]] = []

    async def read(
        self, names: list[str], *, etag: str | None = None
    ) -> tuple[dict[str, Rule], str]:
        self.read_names.append(list(names))
        return {}, "e"

    async def publish(self, entries: list[dict[str, object]]) -> None:
        self.published.append(entries)


async def test_the_catalogs_rules_and_what_was_published_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = Catalog()
    rules = Rules(catalog)  # type: ignore[arg-type]
    rules._names = LRUSet(maxsize=3)
    for n in range(10):
        await rules.get([f"tool{n}"])
    assert catalog.read_names[-1] == ["tool7", "tool8", "tool9"]  # the most recently asked
    published = Published()
    published.done = LRUSet(maxsize=2)
    governance = Governance(catalog, tenant="t")  # type: ignore[arg-type]
    governance._published = published
    specs = [ToolSpec(name=f"tool{n}", description="d", side_effects="read") for n in range(5)]
    await governance.publish(specs)
    assert len(published.done) == 2  # the rest: sent again, if ever published again


def test_threads_outside_a_run_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    saver = RunCheckpointer()
    saver._threads = LRU(2)
    for n in range(5):
        saver._slot(f"thread{n}")
    assert list(saver._threads) == ["thread3", "thread4"]


async def test_the_gateways_prompts_kept_are_bounded() -> None:
    fake = FakeGateway(prompts={name: [[{"role": "system", "content": name}]] for name in "abc"})
    gateway = fake.gateway()
    gateway._prompts = LRU(2)
    for name in "abc":
        assert (await gateway.prompt(name)).name == name
    assert list(gateway._prompts) == ["b", "c"]
    await gateway.aclose()


def test_a_warning_said_once_per_name_remembers_a_bounded_number_of_names() -> None:
    budget._no_cost.cache_clear()
    react._unknown.cache_clear()
    for n in range(SAID_ONCE + 50):
        budget._no_cost(f"model{n}")
        react._unknown(f"model{n}")
    assert budget._no_cost.cache_info().currsize == SAID_ONCE
    assert react._unknown.cache_info().currsize == SAID_ONCE


async def test_a_full_spool_loses_the_write_and_says_so_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(writes, "WRITE_BACKOFF_SECONDS", 0.001)
    monkeypatch.setattr(writes, "MAX_SPOOL_BYTES", 150)

    async def down() -> None:
        raise ConnectionError("memory is down")

    spooling = Writes(spool=tmp_path)
    record = {"op": "record_messages", "scope": {"tenant_id": "t"}, "args": {"n": 1}}
    with caplog.at_level(logging.ERROR, logger="trellis.writes"):
        for _ in range(4):
            await spooling.submit("memory.transcript", down, record=record)
        await spooling.drain()
    assert spooling.spooled == 1 and spooling.failed == 3
    assert len((tmp_path / SPOOL_FILE).read_text().splitlines()) == 1
    assert caplog.text.count("is full") == 1
    await spooling.aclose()
