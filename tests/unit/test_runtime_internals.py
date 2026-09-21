"""Writeback bounds, metric labels, artifact stores, context factory, sync bridge."""

from __future__ import annotations

import asyncio

import pytest

from universal_agent_harness.artifacts.stores import InMemoryArtifactStore
from universal_agent_harness.execution.context_factory import ContextFactory
from universal_agent_harness.execution.sync import in_event_loop, run_sync
from universal_agent_harness.memory.writeback import WritebackQueue
from universal_agent_harness.telemetry.metrics import ALLOWED_LABELS, labels

# --------------------------------------------------------------------------- writeback


async def test_writeback_runs_work_in_the_background():
    queue = WritebackQueue()
    done = asyncio.Event()

    async def work():
        done.set()

    assert queue.submit(work()) is not None
    await queue.drain()
    assert done.is_set()
    assert queue.pending == 0


async def test_writeback_refuses_work_when_saturated():
    queue = WritebackQueue(max_pending=1)

    async def slow():
        await asyncio.sleep(0.05)

    assert queue.submit(slow()) is not None
    second = slow()
    assert queue.submit(second) is None, "a saturated queue must refuse, not grow (§74)"
    second.close()
    await queue.drain()


async def test_writeback_errors_reach_the_handler_not_the_caller():
    seen: list[BaseException] = []
    queue = WritebackQueue(on_error=seen.append)

    async def failing():
        raise ConnectionError("memory down")

    queue.submit(failing())
    await queue.drain()
    assert isinstance(seen[0], ConnectionError)


# --------------------------------------------------------------------------- metrics


def test_only_bounded_labels_survive():
    out = labels(agent_id="inv", tool="search", thread_id="chat-1", user_id="u1", status="ok")
    assert out == {"agent_id": "inv", "tool": "search", "status": "ok"}
    assert "thread_id" not in ALLOWED_LABELS and "user_id" not in ALLOWED_LABELS


def test_label_values_are_coerced_and_empties_dropped():
    assert labels(agent_id="a", retry=3, cached=True, model=None, provider="") == {
        "agent_id": "a",
        "retry": 3,
        "cached": True,
    }


# --------------------------------------------------------------------------- artifacts


async def test_in_memory_store_is_content_addressed_and_bounded():
    store = InMemoryArtifactStore(max_items=2)
    first = await store.put("same")
    again = await store.put("same")
    assert first.artifact_id == again.artifact_id
    assert await store.get(first.artifact_id) == b"same"

    await store.put("second")
    await store.put("third")
    assert len(store._items) == 2  # the oldest was evicted rather than growing (§65)


async def test_artifact_checksum_and_size():
    ref = await InMemoryArtifactStore().put(
        b"12345", type="blob", mime_type="application/octet-stream"
    )
    assert ref.size_bytes == 5
    assert ref.checksum.startswith("sha256:")
    assert ref.mime_type == "application/octet-stream"


# --------------------------------------------------------------------------- context factory


def test_factory_requires_a_tenant():
    with pytest.raises(ValueError, match="tenant_id"):
        ContextFactory().build(agent_id="inv")


def test_factory_derives_stable_run_ids_from_a_durable_position():
    factory = ContextFactory({"tenant_id": "acme"})
    first = factory.build(agent_id="inv", overrides={"thread_id": "t1", "turn_id": "turn-1"})
    second = factory.build(agent_id="inv", overrides={"thread_id": "t1", "turn_id": "turn-1"})
    assert first.agent_run_id == second.agent_run_id


def test_factory_falls_back_to_random_ids_without_a_position():
    factory = ContextFactory({"tenant_id": "acme"})
    assert factory.build(agent_id="inv").agent_run_id != factory.build(agent_id="inv").agent_run_id


def test_factory_can_disable_deterministic_ids():
    factory = ContextFactory({"tenant_id": "acme"}, deterministic_run_ids=False)
    overrides = {"thread_id": "t1", "turn_id": "turn-1"}
    assert (
        factory.build(agent_id="inv", overrides=overrides).agent_run_id
        != factory.build(agent_id="inv", overrides=overrides).agent_run_id
    )


def test_explicit_context_for_the_same_agent_is_reused_as_is(context):
    factory = ContextFactory({"tenant_id": "acme"})
    assert factory.build(agent_id=context.agent_id, context=context) is context


def test_explicit_context_for_another_agent_becomes_a_child(context):
    child = ContextFactory({"tenant_id": "acme"}).build(agent_id="other", context=context)
    assert child.parent_agent_run_id == context.agent_run_id
    assert child.trace_id == context.trace_id


# --------------------------------------------------------------------------- sync bridge


def test_run_sync_outside_a_loop():
    async def work():
        return "done"

    assert not in_event_loop()
    assert run_sync(work()) == "done"


async def test_run_sync_inside_a_loop_uses_a_worker_thread():
    async def work():
        await asyncio.sleep(0)
        return "done"

    assert in_event_loop()
    assert await asyncio.to_thread(run_sync, work()) == "done"
