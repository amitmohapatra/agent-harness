"""A2A push notifications on the harness's webhook rules (design §10: "one notifier used by
three things").

A2A lets a caller register a URL to be told about a task it is not watching. That is the same
problem the harness already solved for runs — an untrusted URL, a signature the receiver can
check, retries that never hold the turn — so this sender reuses those parts rather than growing a
second webhook stack:

* **Targets** go through ``trellis.harness.events.targets``: https only, no credentials or
  fragment, no local names, and every address the host resolves to re-checked at delivery. The
  same function is handed to the SDK as its ``push_url_validator``, so a URL is refused when it is
  *registered*, not only when it is used.
* **Signatures** are ``trellis.memory.webhooks.sign`` — the scheme the Memory Service and the
  harness's own ``WebhookEventSink`` use, so one receiver verifies all three with
  ``verify_signature``.
* **The body** is the protocol's: the event as a ``StreamResponse``, plus A2A's own
  ``X-A2A-Notification-Token`` when the caller registered one.

A refused or failing target is counted and logged, never raised: a caller that registered a bad
URL has broken its own notifications, not the run.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from typing import Any, Final

import httpx
from a2a.server.tasks import PushNotificationConfigStore, PushNotificationSender
from a2a.server.tasks.push_notification_sender import PushNotificationEvent
from a2a.utils.proto_utils import to_stream_response
from google.protobuf.json_format import MessageToDict
from trellis.memory.transport import HEADER_REQUEST_ID
from trellis.memory.webhooks import DELIVERY_HEADER, EVENT_HEADER, SIGNATURE_HEADER, sign

from trellis.harness.events.targets import TargetRefused, resolved_addresses, validate_url
from trellis.harness.events.webhook import WebhookEventSink
from trellis.harness.runtime.logging import get_logger

log = get_logger("trellis.harness_a2a.push")

USER_AGENT: Final = "trellis-harness-a2a"
#: A2A's own header for the token a caller registered with its push config.
TOKEN_HEADER: Final = "X-A2A-Notification-Token"
EVENT_NAME: Final = "a2a.task_update"


class HarnessPushNotifier(PushNotificationSender):
    """Signed, target-checked delivery of A2A task updates."""

    def __init__(
        self,
        config_store: PushNotificationConfigStore,
        *,
        secret: str,
        client: httpx.AsyncClient | None = None,
        attempts: int = 3,
        timeout: float = 10.0,
        allow_local_targets: bool = False,
        verify_targets: bool = True,
    ) -> None:
        """``secret`` is what the receiver verifies the signature with (the same secret a
        ``WebhookEventSink`` uses, if the deployment has one: see :meth:`from_sink`).
        ``verify_targets=False`` skips name resolution, for tests with a mocked transport."""
        if not secret:
            raise ValueError("HarnessPushNotifier needs the secret the receiver verifies with")
        self._configs = config_store
        self.secret = secret
        self.attempts = max(1, attempts)
        self.allow_local_targets = allow_local_targets
        self.verify_targets = verify_targets
        self._client = client or httpx.AsyncClient(
            timeout=timeout, follow_redirects=False, headers={"User-Agent": USER_AGENT}
        )
        self._owns_client = client is None
        self.delivered = 0
        self.failed = 0
        self.refused = 0

    @classmethod
    def from_sink(
        cls, sink: WebhookEventSink, config_store: PushNotificationConfigStore, **overrides: Any
    ) -> HarnessPushNotifier:
        """A notifier that agrees with the deployment's run webhooks: one secret, one target
        policy, so a receiver verifies A2A pushes and run notifications the same way."""
        options: dict[str, Any] = {
            "secret": sink.secret,
            "attempts": sink.attempts,
            "allow_local_targets": sink.allow_local_targets,
            "verify_targets": sink.verify_targets,
        }
        options.update(overrides)
        return cls(config_store, **options)

    async def validate_url(self, url: str) -> bool:
        """The SDK's ``push_url_validator``: refuse a URL at registration, not at delivery.

        Async and boolean because that is the seam the SDK offers
        (``DefaultRequestHandler(push_url_validator=...)``), and refusing early is what stops a
        caller storing an internal address on a task and learning something from the timing.
        """
        try:
            target = validate_url(url, allow_local=self.allow_local_targets)
            if self.verify_targets:
                await resolved_addresses(target, allow_local=self.allow_local_targets)
        except TargetRefused as exc:
            self.refused += 1
            log.warning("a2a.push_target_refused", error=str(exc))
            return False
        return True

    async def send_notification(self, task_id: str, event: PushNotificationEvent) -> None:
        configs = await self._configs.get_info_for_dispatch(task_id)
        if not configs:
            return
        body = MessageToDict(to_stream_response(event))
        # return_exceptions: one unusable config must not abort its siblings, and a notification
        # must never raise into the task's own event stream
        results = await asyncio.gather(
            *(
                self.deliver(config.url, task_id, body, token=config.token or None)
                for config in configs
            ),
            return_exceptions=True,
        )
        for outcome in results:
            if isinstance(outcome, BaseException):
                self.failed += 1
                log.warning("a2a.push_failed", task_id=task_id, error=str(outcome))

    async def deliver(
        self, url: str, task_id: str, body: dict[str, Any], *, token: str | None = None
    ) -> bool:
        """One delivery, with the harness's checks, signature and backoff."""
        try:
            target = validate_url(url, allow_local=self.allow_local_targets)
            if self.verify_targets:
                await resolved_addresses(target, allow_local=self.allow_local_targets)
        except TargetRefused as exc:
            self.refused += 1
            log.warning("a2a.push_target_refused", task_id=task_id, error=str(exc))
            return False
        payload = _json(body)
        delivery = f"a2a_{secrets.token_hex(8)}"
        headers = {
            "Content-Type": "application/json",
            EVENT_HEADER: EVENT_NAME,
            DELIVERY_HEADER: delivery,
            HEADER_REQUEST_ID: f"req_{secrets.token_hex(12)}",
        }
        if token:
            headers[TOKEN_HEADER] = token
        for attempt in range(1, self.attempts + 1):
            headers[SIGNATURE_HEADER] = sign(self.secret, int(time.time()), payload)
            try:
                response = await self._client.post(target, content=payload, headers=headers)
            except Exception as exc:
                # httpx.InvalidURL is not an HTTPError, and a receiver is not worth a raised
                # notification whatever it does
                failure = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code < 300:
                    self.delivered += 1
                    return True
                failure = f"HTTP {response.status_code}"
            log.warning("a2a.push_attempt_failed", task_id=task_id, attempt=attempt, error=failure)
            if attempt < self.attempts:
                await asyncio.sleep(min(2.0, 0.2 * 2 ** (attempt - 1)))
        self.failed += 1
        return False

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _json(body: dict[str, Any]) -> bytes:
    """The exact bytes that are signed and sent: one serialisation, so the signature matches."""
    return json.dumps(body, separators=(",", ":"), sort_keys=True).encode()


__all__ = ["EVENT_NAME", "TOKEN_HEADER", "HarnessPushNotifier"]
