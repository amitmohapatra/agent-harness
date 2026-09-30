"""A2A push notifications: signed, and only ever to a public https address.

A caller registers a URL (and a token) to hear about a task it is not watching. The URL is
somebody else's text, so it is checked as a server-side request forgery would be — https, no
credentials or fragment, never a local name or a non-public address, every resolved address
re-checked at delivery — both when it is registered (the SDK's ``push_url_validator``) and when
it is used. The body is the protocol's ``StreamResponse``; ``X-Trellis-Signature:
t=<unix>,v1=<hmac-sha256("t.body")>`` signs it with the token the caller registered — the
scheme agent-runs' webhooks use, which a receiver checks with :func:`verify_signature`. A
config without a token cannot be signed and is not delivered to. A failing receiver is logged,
never raised into the task.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import socket
import time
from hashlib import sha256
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import httpx
from a2a.server.tasks import PushNotificationConfigStore, PushNotificationSender
from a2a.server.tasks.push_notification_sender import PushNotificationEvent
from a2a.utils.proto_utils import to_stream_response
from google.protobuf.json_format import MessageToDict

log = logging.getLogger("trellis.a2a.push")

EVENT_NAME: Final = "a2a.task_update"
SIGNATURE_HEADER: Final = "X-Trellis-Signature"
EVENT_HEADER: Final = "X-Trellis-Event"
DELIVERY_HEADER: Final = "X-Trellis-Delivery"
#: How old a signature a receiver accepts.
SIGNATURE_TOLERANCE_SECONDS: Final = 300
#: A2A's own header for the token a caller registered.
TOKEN_HEADER: Final = "X-A2A-Notification-Token"
URL_MAX_CHARS: Final = 2048
LOCAL_SUFFIXES: Final = (".localhost", ".local", ".internal")
ATTEMPTS: Final = 3
TIMEOUT_SECONDS: Final = 10.0


def sign(secret: str, timestamp: int, body: bytes) -> str:
    digest = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, sha256).hexdigest()
    return f"t={timestamp},v1={digest}"


def verify_signature(
    secret: str, header: str | None, body: bytes, *, now: int | None = None
) -> bool:
    """Whether ``header`` signs ``body`` with ``secret`` and is recent (a receiver's check)."""
    parts = dict(p.split("=", 1) for p in (header or "").split(",") if "=" in p)
    stamp, digest = parts.get("t", ""), parts.get("v1", "")
    if not stamp.isdigit() or not digest:
        return False
    current = int(time.time()) if now is None else now
    if abs(current - int(stamp)) > SIGNATURE_TOLERANCE_SECONDS:
        return False
    expected = sign(secret, int(stamp), body).partition(",v1=")[2]
    return hmac.compare_digest(expected, digest)


class TargetRefused(ValueError):
    """The URL points somewhere a notification may not go."""


def validate_url(url: str) -> str:
    """The normalised URL, or :class:`TargetRefused`. Pure: no name resolution."""
    if len(url) > URL_MAX_CHARS:
        raise TargetRefused("the url is too long")
    parts = urlsplit(url.strip())
    if parts.scheme != "https":
        raise TargetRefused("the url must use https")
    if not parts.hostname or parts.username or parts.password or parts.fragment:
        raise TargetRefused("the url needs a host and no credentials or fragment")
    host = parts.hostname.lower()
    if host == "localhost" or host.endswith(LOCAL_SUFFIXES):
        raise TargetRefused("the url may not point at a local host")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None and not _public(literal):
        raise TargetRefused("the url may not point at a private address")
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", parts.query, ""))


async def check_addresses(url: str) -> None:
    """Refuse unless every address the host resolves to is public."""
    host = urlsplit(url).hostname or ""
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise TargetRefused("the host does not resolve") from exc
    addresses = {str(info[4][0]) for info in infos}
    if not addresses or not all(_public(ipaddress.ip_address(a)) for a in addresses):
        raise TargetRefused("the host resolves to a private address")


def _public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_global and not (address.is_multicast or address.is_reserved)


class PushNotifier(PushNotificationSender):
    """Signed, target-checked delivery of task updates to every registered config."""

    def __init__(
        self, configs: PushNotificationConfigStore, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self._configs = configs
        self._client = client or httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=False)

    async def validate_url(self, url: str) -> bool:
        """The SDK's ``push_url_validator``: refused at registration, not only at delivery."""
        try:
            await check_addresses(validate_url(url))
        except TargetRefused as exc:
            log.warning("A2A push target refused: %s", exc)
            return False
        return True

    async def send_notification(self, task_id: str, event: PushNotificationEvent) -> None:
        body = json.dumps(
            MessageToDict(to_stream_response(event)), separators=(",", ":"), sort_keys=True
        ).encode()
        configs = await self._configs.get_info_for_dispatch(task_id)
        await asyncio.gather(*(self.deliver(c.url, c.token, task_id, body) for c in configs))

    async def deliver(self, url: str, token: str, task_id: str, body: bytes) -> bool:
        """One delivery with retries; ``False`` when refused or never accepted."""
        if not token:
            log.warning("A2A push for %s skipped: the config has no token to sign with", task_id)
            return False
        try:
            target = validate_url(url)
            await check_addresses(target)
        except TargetRefused as exc:
            log.warning("A2A push target refused for %s: %s", task_id, exc)
            return False
        headers = {
            "Content-Type": "application/json",
            EVENT_HEADER: EVENT_NAME,
            DELIVERY_HEADER: f"{task_id}:{time.time_ns()}",
            TOKEN_HEADER: token,
        }
        for attempt in range(1, ATTEMPTS + 1):
            headers[SIGNATURE_HEADER] = sign(token, int(time.time()), body)
            try:
                response = await self._client.post(target, content=body, headers=headers)
                if response.status_code < 300:
                    return True
                failure = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                failure = f"{type(exc).__name__}: {exc}"
            log.warning("A2A push attempt %d for %s failed: %s", attempt, task_id, failure)
            if attempt < ATTEMPTS:
                await asyncio.sleep(0.2 * 2 ** (attempt - 1))
        return False
