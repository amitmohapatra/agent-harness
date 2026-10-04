"""Way 2, pluggable blocks: your own workers on agent-runs' queue, with ``trellis.runs.Worker``
and no harness — the handler is plain code (it could call a LangGraph graph, an OpenAI Agents
``Runner`` or a Claude ``query()`` the same way).

* Queue: three refunds are started with ``queue=True``: ``QUEUED`` until a worker claims one.
* ``Worker(runs, handle, ["refunds"])`` claims them under a lease and runs ``handle`` on each;
  ``job.checkpoint`` saves progress, ``job.pause`` and ``job.finish`` write as this worker.
* A refund over the limit pauses for a person (``role:finance``'s inbox); the answer resumes it
  ``QUEUED`` again, and the worker that claims it — any worker — reads the reviewer's
  ``last_resolution`` and the run's ``checkpoint``.
* Webhooks: a receiver checks a ``run.paused`` delivery with ``verify_signature`` before it reads
  it, and refuses one whose body was changed.

Offline (no ``RUNS_URL``) the store is the harness's in-process one, which takes the same calls,
and the webhook delivery is built and signed here with ``sign`` — agent-runs' own signing — as
the service would send it. In production: ``await Worker(RunsClient(), handle, [...]).serve()``
(it stops gracefully on SIGTERM) and a subscription from ``runs.webhooks.create(url, events)``.

    .venv/bin/python examples/blocks_worker.py
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from _offline import Runs, runs_store

from trellis.contracts import (
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    RunStart,
    RunStatus,
    ToolCall,
    new_id,
)
from trellis.runs import Job, RunSummary, WebhookData, WebhookDelivery, WebhookEvent, Worker
from trellis.runs.webhooks import parse_delivery, sign, verify_signature

TENANT = "default"  # the tenant your key speaks for (a development key's: default)
AGENT = "refunds"
LIMIT = 100  # a refund above it waits for a person
SECRET = "whsec_example"  # offline: the secret a subscription's create returns once


# --------------------------------------------------------------------------- the handler
async def handle(job: Job) -> None:
    """One claimed run: refund it, or ask finance first. The same code runs the run's first
    attempt and the attempt after a person answered (``record.last_resolution``)."""
    record = job.record
    refund: dict[str, Any] = record.input
    answered = record.last_resolution
    if refund["amount"] > LIMIT and answered is None:
        await job.checkpoint({"checked": refund["order"]})  # progress, and the lease extended
        asked = Interrupt(
            tenant_id=record.tenant_id,
            run_id=record.run_id,
            reason=InterruptReason.APPROVAL,
            question=f"Refund {refund['amount']} on {refund['order']}?",
            tool_call=ToolCall(tool="refund", args=refund),
            assignee="role:finance",
        )
        await job.pause(asked, checkpoint={"checked": refund["order"]})  # the lease ends here
        return
    if answered is not None and answered.decision is not InterruptDecision.APPROVE:
        await job.finish(RunStatus.REJECTED, output={"refunded": False})
        return
    approved = {"approved_by": answered.reviewer} if answered is not None else {}
    await job.finish(RunStatus.SUCCESS, output={"refunded": refund["amount"], **approved})


# --------------------------------------------------------------------------- the receiver
def receive(header: str | None, body: bytes) -> None:
    """Your webhook endpoint's body: the raw bytes are verified before they are read."""
    if not verify_signature(SECRET, header, body):
        print("webhook: refused (401)")
        return
    delivery = parse_delivery(body)
    awaiting = delivery.data.run.awaiting
    print("webhook:", delivery.type.value, delivery.data.run.run_id, awaiting and awaiting.question)


def delivered(summary: RunSummary) -> tuple[str, bytes]:
    """Offline: the ``run.paused`` delivery agent-runs would POST for ``summary``, signed."""
    delivery = WebhookDelivery(
        event_id=new_id("evt"),
        type=WebhookEvent.PAUSED,
        tenant_id=TENANT,
        occurred_at=summary.updated_at,
        data=WebhookData(run=summary),
    )
    body = delivery.model_dump_json().encode()
    return sign(SECRET, int(time.time()), body), body


# --------------------------------------------------------------------------- one day
async def drain(worker: Worker) -> None:
    """Run the worker until the queue is empty (``serve()`` in production)."""
    while await worker.run_once():
        pass


async def main() -> None:
    runs: Runs = runs_store()
    started: list[str] = []
    for order, amount in (("A-1", 20), ("A-2", 450), ("A-3", 60)):
        start = RunStart(tenant_id=TENANT, agent_id=AGENT, input={"order": order, "amount": amount})
        started.append((await runs.start(start, queue=True)).run_id)  # QUEUED: for any worker

    worker = Worker(runs, handle, [AGENT], concurrency=2, tenant=TENANT)
    await drain(worker)

    # finance's inbox: the paused refund, and the webhook that told the UI about it
    inbox = runs.iterate(status=RunStatus.PAUSED, assignee="role:finance", tenant=TENANT)
    async for waiting in inbox:
        assert waiting.awaiting is not None
        print("inbox:", waiting.run_id, waiting.awaiting.question)
        if waiting.run_id not in started:  # another day's: not this example's to answer
            continue
        header, body = delivered(waiting)
        receive(header, body)
        receive(header, body.replace(b"450", b"45"))  # changed on the way: refused
        answer = InterruptResolution(
            interrupt_id=waiting.awaiting.interrupt_id,
            run_id=waiting.run_id,
            decision=InterruptDecision.APPROVE,
            reviewer="user:cfo",
        )
        resumed = await runs.resume(answer, tenant=TENANT)
        print("resumed:", resumed.status.value, "attempt", resumed.attempt)  # QUEUED again

    await drain(worker)  # a worker claims it again and finishes it
    for run_id in started:
        record = await runs.get(run_id, tenant=TENANT)
        assert record is not None
        print(record.status.value, record.input, "->", record.output)
    await runs.aclose()


if __name__ == "__main__":
    asyncio.run(main())
