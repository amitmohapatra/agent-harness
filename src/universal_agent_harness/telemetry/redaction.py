"""What may leave the process (§27). Conservative by default; opt in per tenant.

The rules are deliberately boring: drop anything whose *name* looks like a secret, drop
anything whose value looks like a credential, and replace anything large with a reference
(a size and a content hash) instead of the content. A redactor never raises — a redaction
bug must not break an execution — and unknown types are stringified before truncation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Mapping
from typing import Any

REDACTED = "[redacted]"

#: Words that make an attribute name sensitive. Matched against the *segments* of a key
#: (``api_key`` -> ``api``/``key``), never as raw substrings: substring matching redacts
#: ``gen_ai.usage.input_tokens`` because it contains "token", which is how observability
#: quietly loses its own metrics.
SENSITIVE_KEY_WORDS: frozenset[str] = frozenset(
    {
        "authorization",
        "apikey",
        "secret",
        "secrets",
        "password",
        "passwd",
        "token",
        "credential",
        "credentials",
        "cookie",
        "cookies",
        "bearer",
        "ssn",
        "cvv",
        "pii",
        "passphrase",
    }
)

#: Sensitive names spelled as consecutive segments (``api_key``, ``private-key``...).
SENSITIVE_KEY_PHRASES: frozenset[tuple[str, str]] = frozenset(
    {
        ("api", "key"),
        ("access", "key"),
        ("secret", "key"),
        ("private", "key"),
        ("session", "key"),
        ("card", "number"),
        ("access", "token"),
        ("refresh", "token"),
        ("id", "token"),
    }
)

#: Backwards-compatible alias for callers that passed extra substrings.
SENSITIVE_KEY_PARTS: tuple[str, ...] = tuple(sorted(SENSITIVE_KEY_WORDS))

#: Values that look like credentials even under an innocent key.
_VALUE_PATTERNS = (
    re.compile(r"^(sk|pk|rk)-[A-Za-z0-9_\-]{12,}$"),
    re.compile(r"^Bearer\s+[A-Za-z0-9._\-]{10,}$", re.IGNORECASE),
    re.compile(r"^eyJ[A-Za-z0-9._\-]{20,}$"),  # JWT
    # A long base64-ish blob, but only when it has the character mix of a real key: plain
    # prose of the same length (and a run of one character) must not be flagged.
    re.compile(
        r"^(?=[A-Za-z0-9+/]*[a-z])(?=[A-Za-z0-9+/]*[A-Z])(?=[A-Za-z0-9+/]*\d)"
        r"[A-Za-z0-9+/]{40,}={0,2}$"
    ),
)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


class DefaultRedactor:
    """The default :class:`~universal_agent_contracts.ports.TelemetryRedactor`."""

    def __init__(
        self,
        *,
        max_value_chars: int = 2000,
        extra_sensitive_keys: tuple[str, ...] = (),
        mask_emails: bool = True,
        drop_payloads: bool = False,
    ) -> None:
        self.max_value_chars = max_value_chars
        self.sensitive_words = SENSITIVE_KEY_WORDS | {k.lower() for k in extra_sensitive_keys}
        self.mask_emails = mask_emails
        #: When true (the capture policy said "no raw content"), payloads become references.
        self.drop_payloads = drop_payloads

    # -- ports -------------------------------------------------------------------
    def redact_attributes(self, attributes: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in attributes.items():
            if value is None:
                continue
            # A credential is never a number: exempting numeric values keeps token counts,
            # costs and latencies out of the name-based rules.
            if not isinstance(value, bool | int | float) and self._sensitive_key(key):
                out[key] = REDACTED
                continue
            out[key] = self._scalar(value)
        return out

    def redact_input(self, value: Any) -> Any:
        return self._payload(value)

    def redact_output(self, value: Any) -> Any:
        return self._payload(value)

    # -- internals ---------------------------------------------------------------
    def _sensitive_key(self, key: str) -> bool:
        segments = _segments(key)
        if self.sensitive_words & set(segments):
            return True
        pairs = set(itertools.pairwise(segments))
        return bool(pairs & SENSITIVE_KEY_PHRASES)

    def _scalar(self, value: Any) -> Any:
        if isinstance(value, bool | int | float):
            return value
        if isinstance(value, list | tuple):
            # OpenTelemetry accepts homogeneous scalar sequences; keeping them as sequences
            # (rather than a JSON string) is what lets a backend filter on them.
            return [self._scalar(v) for v in value]
        text = value if isinstance(value, str) else _stringify(value)
        if any(p.match(text) for p in _VALUE_PATTERNS):
            return REDACTED
        if self.mask_emails:
            text = _EMAIL.sub("[email]", text)
        return _truncate(text, self.max_value_chars)

    def _payload(self, value: Any) -> Any:
        if value is None:
            return None
        if self.drop_payloads:
            return reference(value)
        if isinstance(value, Mapping):
            return {
                k: (REDACTED if self._sensitive_key(str(k)) else self._payload(v))
                for k, v in value.items()
            }
        if isinstance(value, list | tuple):
            return [self._payload(v) for v in value]
        return self._scalar(value)


class NoOpRedactor:
    """Passes everything through. Only for tests and explicitly trusted environments."""

    def redact_attributes(self, attributes: Mapping[str, Any]) -> dict[str, Any]:
        return dict(attributes)

    def redact_input(self, value: Any) -> Any:
        return value

    def redact_output(self, value: Any) -> Any:
        return value


def reference(value: Any) -> dict[str, Any]:
    """A content-free stand-in: type, size and hash. Enough to correlate, safe to export."""
    text = _stringify(value)
    raw = text.encode("utf-8", "replace")
    return {
        "redacted": True,
        "type": type(value).__name__,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest()[:32],
    }


_SEGMENT = re.compile(r"[^a-z0-9]+")


def _segments(key: str) -> tuple[str, ...]:
    """Lower-case word segments of an attribute name (``X-Api-Key`` -> ``x``/``api``/``key``)."""
    return tuple(part for part in _SEGMENT.split(key.lower()) if part)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"
