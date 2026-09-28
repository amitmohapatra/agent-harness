"""``WebhookEventSink``: the notifier for disconnected clients (design §10).

A run that pauses at 3 a.m. or finishes after the caller hung up is announced with a signed
POST: the body is the ``RunEvent`` as JSON, ``X-Trellis-Signature`` is
``t=<unix seconds>,v1=<hex hmac-sha256 of "t.body">`` (the scheme the Memory Service uses,
so one receiver verifies both with ``trellis.memory.webhooks.verify_signature``). A run's own
``webhook_url`` (from the request metadata) wins over the sink's default; deliveries go
through the writeback queue with retries and never hold the turn.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Iterable
from typing import Any, Final

import httpx
from trellis.contracts.runs import RunEvent, RunEventType
from trellis.memory.transport import HEADER_REQUEST_ID
from trellis.memory.webhooks import DELIVERY_HEADER, EVENT_HEADER, SIGNATURE_HEADER, sign

from trellis.harness.events.stream import WEBHOOK_URL_FIELD
from trellis.harness.events.targets import TargetRefused, resolved_addresses, validate_url
from trellis.harness.runtime.logging import get_logger

log = get_logger(__name__)

DEFAULT_TYPES: Final = (RunEventType.RUN_FINISHED, RunEventType.INTERRUPT)
USER_AGENT: Final = "trellis-harness-webhooks"


class WebhookEventSink:
    name = "webhook"

    def __init__(
        self,
        url: str | None,
        *,
        secret: str,
        types: Iterable[RunEventType | str] = DEFAULT_TYPES,
        attempts: int = 3,
        timeout: float = 10.0,
        queue: Any = None,
        client: httpx.AsyncClient | None = None,
        allow_local_targets: bool = False,
        verify_targets: bool = True,
    ) -> None:
        """``allow_local_targets`` permits http and local addresses (development only);
        ``verify_targets=False`` skips name resolution (tests with a mocked transport)."""
        if not secret:
            raise ValueError("WebhookEventSink needs the secret the receiver verifies with")
        self.allow_local_targets = allow_local_targets
        self.verify_targets = verify_targets
        self.url = validate_url(url, allow_local=allow_local_targets) if url else None
        self.secret = secret
        self.types = frozenset(RunEventType(t) for t in types)
        self.attempts = max(1, attempts)
        self.queue = queue
        self.refused: int = 0
        self._client = client or httpx.AsyncClient(
            timeout=timeout, follow_redirects=False, headers={"User-Agent": USER_AGENT}
        )
        self._owns_client = client is None
        self.delivered: int = 0
        self.failed: int = 0

    def attach_queue(self, queue: Any) -> None:
        """The harness hands over its writeback queue, so deliveries never hold a turn."""
        if self.queue is None:
            self.queue = queue

    async def publish(self, event: RunEvent) -> None:
        if event.type not in self.types:
            return
        url = self._target(event)
        if not url:
            return
        delivery = self.deliver(url, event)
        if self.queue is not None:
            if self.queue.submit(delivery, name=f"webhook:{event.event_id}") is None:
                # saturated: a backlog of notifications must not become a stalled run
                delivery.close()
                self.failed += 1
                log.warning(
                    "webhook.dropped", url=url, event_id=event.event_id, reason="queue saturated"
                )
            return
        await delivery

    def _target(self, event: RunEvent) -> str | None:
        """The run's own URL (request metadata) wins over the sink's default, but only
        when it passes the same checks; a refused one is logged and the default is used."""
        own = event.data.get(WEBHOOK_URL_FIELD)
        if own:
            try:
                return validate_url(str(own), allow_local=self.allow_local_targets)
            except TargetRefused as exc:
                self.refused += 1
                log.warning("webhook.target_refused", event_id=event.event_id, error=str(exc))
        return self.url

    async def deliver(self, url: str, event: RunEvent) -> bool:
        if self.verify_targets:
            try:
                await resolved_addresses(url, allow_local=self.allow_local_targets)
            except TargetRefused as exc:
                self.refused += 1
                log.warning("webhook.target_refused", url=url, error=str(exc))
                return False
        body = event.model_dump_json().encode()
        headers = {
            "Content-Type": "application/json",
            EVENT_HEADER: f"run.{event.type.value.lower()}",
            DELIVERY_HEADER: event.event_id,
            HEADER_REQUEST_ID: f"req_{secrets.token_hex(12)}",
        }
        for attempt in range(1, self.attempts + 1):
            headers[SIGNATURE_HEADER] = sign(self.secret, int(time.time()), body)
            try:
                response = await self._client.post(url, content=body, headers=headers)
            except httpx.HTTPError as exc:
                failure = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code < 300:
                    self.delivered += 1
                    return True
                failure = f"HTTP {response.status_code}"
            if attempt < self.attempts:
                await asyncio.sleep(min(2.0, 0.2 * 2 ** (attempt - 1)))
            log.warning("webhook.attempt_failed", url=url, attempt=attempt, error=failure)
        self.failed += 1
        return False

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
