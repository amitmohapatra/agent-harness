"""The live stack a suite runs against (``tests/features``, ``tests/scenarios``,
``tests/live``): what its environment names, and whether a service of it answers."""

from __future__ import annotations

import os

import httpx


def env(name: str) -> str | None:
    """The environment's ``name``, ``None`` when unset or blank."""
    value = os.environ.get(name, "").strip()
    return value or None


def reachable(url: str | None) -> bool:
    """Whether the service at ``url`` answers its liveness check."""
    if url is None:
        return False
    try:
        return httpx.get(url.rstrip("/") + "/health/live", timeout=3.0).status_code < 500
    except httpx.HTTPError:
        return False
