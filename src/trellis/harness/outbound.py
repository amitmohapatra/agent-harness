"""Where a URL that is somebody else's text may point: the check every address the harness is
given to reach out to passes — an A2A caller's push target (``a2a.push``), the webhook a Deep
Agents async sub-agent's server is told to call (``ReAct``) — as a server-side request
forgery would be checked: https, no credentials or fragment, never a local name or a
non-public address. Pure: no name resolution (``a2a.push`` checks every resolved address again
when it delivers).
"""

from __future__ import annotations

import ipaddress
from typing import Final
from urllib.parse import urlsplit, urlunsplit

URL_MAX_CHARS: Final = 2048
LOCAL_SUFFIXES: Final = (".localhost", ".local", ".internal")


class TargetRefused(ValueError):
    """The URL points somewhere the harness may not send to."""


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
    if literal is not None and not public(literal):
        raise TargetRefused("the url may not point at a private address")
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", parts.query, ""))


def public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether ``address`` is a public one (an IPv4-mapped IPv6 address as its IPv4 one)."""
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_global and not (address.is_multicast or address.is_reserved)
