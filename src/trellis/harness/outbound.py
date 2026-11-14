"""Where a URL that is somebody else's text may point: the check every address the harness is
given to reach out to passes — an A2A caller's push target (``a2a.push``), the webhook a Deep
Agents async sub-agent's server is told to call (``ReAct``) — as a server-side request
forgery would be checked: https, no credentials or fragment, never a local name or a
non-public address. Two halves, both here: what the URL says (:func:`validate_url`, pure, for
a setting read when an agent is built) and where its host leads (:func:`checked`, which
resolves it, for the moment the harness sends to it or hands it on).
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Final
from urllib.parse import urlsplit, urlunsplit

URL_MAX_CHARS: Final = 2048
LOCAL_SUFFIXES: Final = (".localhost", ".local", ".internal")


class TargetRefused(ValueError):
    """The URL points somewhere the harness may not send to."""


def validate_url(url: str) -> str:
    """The normalised URL, or :class:`TargetRefused`. Pure: no name resolution
    (:func:`checked` does that)."""
    if len(url) > URL_MAX_CHARS:
        raise TargetRefused("the url is too long")
    parts = urlsplit(url.strip())
    if parts.scheme != "https":
        raise TargetRefused("the url must use https")
    if not parts.hostname or parts.username or parts.password or parts.fragment:
        raise TargetRefused("the url needs a host and no credentials or fragment")
    host = parts.hostname.lower().rstrip(".")  # a trailing dot names the same host
    if host == "localhost" or host.endswith(LOCAL_SUFFIXES):
        raise TargetRefused("the url may not point at a local host")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None and not public(literal):
        raise TargetRefused("the url may not point at a private address")
    last = host.rsplit(".", 1)[-1]
    if literal is None and (last.isdigit() or last.startswith("0x")):
        # no name ends in a number: a resolver reads 2130706433, 127.1 or 0x7f000001 as an address
        raise TargetRefused("the url's host is a number, not a name or an address in full")
    try:
        port = "" if parts.port is None else f":{parts.port}"
    except ValueError as exc:
        raise TargetRefused("the url's port is not a number") from exc
    netloc = f"[{host}]{port}" if ":" in host else f"{host}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))


async def checked(url: str) -> str:
    """The normalised URL (:func:`validate_url`) when every address its host resolves to is
    public, or :class:`TargetRefused`: a name may say one thing and lead to another."""
    target = validate_url(url)
    host = urlsplit(target).hostname or ""
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise TargetRefused("the host does not resolve") from exc
    addresses = {str(info[4][0]) for info in infos}
    if not addresses or not all(public(ipaddress.ip_address(a)) for a in addresses):
        raise TargetRefused("the host resolves to a private address")
    return target


def public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether ``address`` is a public one (an IPv4-mapped IPv6 address as its IPv4 one)."""
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_global and not (address.is_multicast or address.is_reserved)
