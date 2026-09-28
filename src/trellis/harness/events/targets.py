"""Where a webhook may point. The harness performs the request, so a per-run URL is checked
as a server-side request forgery would be: https only, no credentials or fragment, never a
local name or a non-public address, and every address the host resolves to is checked at
delivery time (the same rules the Memory Service applies to its subscriptions)."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Final
from urllib.parse import urlsplit, urlunsplit

URL_MAX_CHARS: Final = 2048
LOCAL_HOSTS: Final = frozenset({"localhost"})
LOCAL_SUFFIXES: Final = (".localhost", ".local", ".internal")


class TargetRefused(ValueError):
    """The URL points somewhere a webhook may not go."""


def _is_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_global and not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def validate_url(url: str, *, allow_local: bool = False) -> str:
    """The normalised URL, or ``TargetRefused``. Pure: no name resolution here."""
    if len(url) > URL_MAX_CHARS:
        raise TargetRefused("webhook url is too long")
    parts = urlsplit(url.strip())
    if parts.scheme not in ("https", "http") or (parts.scheme == "http" and not allow_local):
        raise TargetRefused("webhook url must use https")
    if not parts.hostname or parts.username or parts.password or parts.fragment:
        raise TargetRefused("webhook url needs a host and no credentials or fragment")
    host = parts.hostname.lower()
    if not allow_local:
        if host in LOCAL_HOSTS or host.endswith(LOCAL_SUFFIXES):
            raise TargetRefused("webhook url may not point at a local host")
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None and not _is_public(literal):
            raise TargetRefused("webhook url may not point at a private address")
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", parts.query, ""))


async def resolved_addresses(url: str, *, allow_local: bool = False) -> list[str]:
    """Every address the host resolves to; refused unless all of them are public."""
    host = urlsplit(url).hostname or ""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise TargetRefused("webhook host does not resolve") from exc
    addresses = sorted({str(info[4][0]) for info in infos})
    if not addresses:
        raise TargetRefused("webhook host does not resolve")
    if not allow_local:
        for address in addresses:
            if not _is_public(ipaddress.ip_address(address)):
                raise TargetRefused("webhook host resolves to a private address")
    return addresses
