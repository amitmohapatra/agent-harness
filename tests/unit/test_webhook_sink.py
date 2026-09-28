"""The notifier signs what it sends, refuses targets it must not reach, and never holds
the run."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx
from trellis.contracts import AgentExecutionContext, RunEventType
from trellis.memory.webhooks import verify_signature

from trellis.harness.events import RunEventStream, WebhookEventSink
from trellis.harness.events.targets import TargetRefused
from trellis.harness.memory.writeback import WritebackQueue

CTX = AgentExecutionContext.create(tenant_id="acme", user_id="u1", agent_id="ref")
SECRET = "s" * 64
HOOK = "https://hooks.example/run"


def sink(url: str | None = HOOK, **options) -> WebhookEventSink:
    # respx answers for hosts that do not resolve; name resolution is checked on its own
    return WebhookEventSink(url, secret=SECRET, verify_targets=False, **options)


@respx.mock
async def test_finished_and_interrupt_events_are_delivered_signed() -> None:
    route = respx.post(HOOK).respond(200)
    notifier = sink(attempts=1)
    stream = RunEventStream(CTX, [notifier])
    await stream.emit(RunEventType.RUN_STARTED)  # not a notified type
    await stream.emit(RunEventType.RUN_FINISHED, outcome="success", result={"ok": True})
    assert route.call_count == 1 and notifier.delivered == 1
    request = route.calls.last.request
    assert request.headers["X-Trellis-Event"] == "run.run_finished"
    assert request.headers["X-Trellis-Delivery"].startswith("evt_")
    assert request.headers["X-Request-ID"].startswith("req_")
    assert verify_signature(SECRET, request.headers["X-Trellis-Signature"], request.content)
    assert not verify_signature("other", request.headers["X-Trellis-Signature"], request.content)
    await notifier.aclose()


@respx.mock
async def test_the_runs_own_webhook_wins_and_failures_are_retried_then_counted() -> None:
    own = respx.post("https://own.example/hook").mock(
        side_effect=[httpx.Response(500), httpx.Response(200)]
    )
    default = respx.post(HOOK).respond(503)
    notifier = sink(attempts=2)
    stream = RunEventStream(CTX, [notifier])
    await stream.emit(
        RunEventType.RUN_FINISHED, outcome="success", webhook_url="https://own.example/hook"
    )
    assert own.call_count == 2 and notifier.delivered == 1 and default.call_count == 0
    await stream.emit(RunEventType.RUN_FINISHED, outcome="error")
    assert default.call_count == 2 and notifier.failed == 1
    await notifier.aclose()


@respx.mock
async def test_a_runs_own_url_is_held_to_the_same_rules_as_the_default() -> None:
    """A request can name where to be told, not what the process may reach: a private
    address or a plain-http target is refused and the deployment's default is used."""
    default = respx.post(HOOK).respond(200)
    notifier = sink(attempts=1)
    stream = RunEventStream(CTX, [notifier])
    for bad in ("http://169.254.169.254/latest", "https://localhost/hook", "ftp://x.example/y"):
        await stream.emit(RunEventType.RUN_FINISHED, outcome="success", webhook_url=bad)
    assert notifier.refused == 3 and default.call_count == 3
    with pytest.raises(TargetRefused):
        WebhookEventSink("http://hooks.example/run", secret=SECRET)
    local = WebhookEventSink(
        "http://localhost:9/hook", secret=SECRET, allow_local_targets=True, verify_targets=False
    )
    assert local.url == "http://localhost:9/hook"  # development only, and said so
    await notifier.aclose()
    await local.aclose()


@respx.mock
async def test_delivery_never_holds_the_turn_once_a_queue_is_attached() -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.3)
        return httpx.Response(200)

    route = respx.post(HOOK).mock(side_effect=slow)
    notifier = sink(attempts=1)
    queue = WritebackQueue(8)
    notifier.attach_queue(queue)
    stream = RunEventStream(CTX, [notifier])
    loop = asyncio.get_running_loop()
    started = loop.time()
    await stream.emit(RunEventType.RUN_FINISHED, outcome="success")
    assert loop.time() - started < 0.2  # emitted, not delivered, inside the turn
    assert route.call_count == 0
    await queue.drain(5)
    assert route.call_count == 1 and notifier.delivered == 1
    await notifier.aclose()


@respx.mock
async def test_a_saturated_queue_drops_the_delivery_and_counts_it() -> None:
    route = respx.post(HOOK).respond(200)
    notifier = sink(attempts=1)
    queue = WritebackQueue(1)
    blocker = asyncio.Event()

    async def hold() -> None:
        await blocker.wait()

    queue.submit(hold(), name="occupant")  # the one slot
    notifier.attach_queue(queue)
    await RunEventStream(CTX, [notifier]).emit(RunEventType.RUN_FINISHED, outcome="success")
    assert notifier.failed == 1 and route.call_count == 0
    blocker.set()
    await queue.drain(5)
    await notifier.aclose()


async def test_no_url_means_nothing_is_sent() -> None:
    notifier = sink(None)
    await RunEventStream(CTX, [notifier]).emit(RunEventType.RUN_FINISHED, outcome="success")
    assert notifier.delivered == 0 and notifier.failed == 0
    with pytest.raises(ValueError, match="secret"):
        WebhookEventSink(HOOK, secret="")
    await notifier.aclose()
