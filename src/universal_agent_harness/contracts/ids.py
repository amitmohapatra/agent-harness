"""Identifier helpers. Stable, deterministic ids are how the harness stays idempotent."""

from __future__ import annotations

import hashlib
import re
import uuid

_SAFE = re.compile(r"[^A-Za-z0-9._:\-]")


def new_id(prefix: str = "") -> str:
    """A fresh opaque id. ``prefix`` keeps ids greppable (``run_``, ``req_``...)."""
    return f"{prefix}{uuid.uuid4().hex}"


def safe_id(value: object, *, max_len: int = 200) -> str:
    """Coerce an arbitrary framework id into the id alphabet shared with the Memory Service."""
    cleaned = _SAFE.sub("-", str(value)).strip("-.")[:max_len]
    return cleaned or "x"


def stable_id(*parts: object, prefix: str = "", size: int = 16) -> str:
    """A deterministic id derived from ``parts``.

    Used wherever a retry must produce the *same* id as the original attempt: agent run ids
    derived from (thread, turn, agent), idempotency keys for observations, artifacts and
    tool calls. ``None`` and ``""`` are distinct from a missing part only by position.
    """
    h = hashlib.blake2b(digest_size=size)
    for part in parts:
        h.update(b"" if part is None else str(part).encode("utf-8"))
        h.update(b"\x1f")
    return f"{prefix}{h.hexdigest()}"
