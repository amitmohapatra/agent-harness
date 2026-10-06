"""What may leave the process: span attributes, the run's events (AG-UI, A2A task updates and
push notifications: ``events.py``), and the tool records sent to the memory service.

The rules are deliberately boring: drop anything whose *name* looks like a secret, drop
anything whose value looks like a credential, mask e-mail addresses, and cut long values. Text
that is a JSON object or array (a model's tool-call arguments, a tool result as the model
reads it) is redacted as what it holds, and stays JSON text. A redactor never raises — a
redaction bug must not break a run — and unknown types are stringified before truncation.
"""

from __future__ import annotations

import functools
import itertools
import json
import re
from collections.abc import Mapping
from typing import Any

REDACTED = "[redacted]"
#: Longest string value kept; the rest is cut and says how much was cut.
MAX_VALUE_CHARS = 2000
#: Nesting deeper than this is cut to a marker: a value that contains itself would
#: otherwise recurse until RecursionError, and a redactor must not raise.
MAX_DEPTH = 32
#: What a cut-off nested value becomes.
TOO_DEEP = "[nested too deep]"

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

#: An address starts where a run of its characters starts (the lookbehind): tried from every
#: position inside a long run instead, masking would be quadratic in the value's length.
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+\.[\w.]+")


class Redactor:
    """The contracts ``TelemetryRedactor`` every span, outbound event and memory tool record
    goes through."""

    __slots__ = ()

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
    @staticmethod
    def _sensitive_key(key: str) -> bool:
        return _sensitive(key)

    def _scalar(self, value: Any, depth: int = 0) -> Any:
        if isinstance(value, bool | int | float):
            return value
        if isinstance(value, list | tuple):
            if depth >= MAX_DEPTH:
                return TOO_DEEP
            # OpenTelemetry accepts homogeneous scalar sequences; keeping them as sequences
            # (rather than a JSON string) is what lets a backend filter on them.
            return [self._scalar(v, depth + 1) for v in value]
        text = value if isinstance(value, str) else _stringify(value)
        held = _json(text)
        if held is not None:
            text = _stringify(self._payload(held, depth + 1))
        elif any(p.match(text) for p in _VALUE_PATTERNS):
            return REDACTED
        return _truncate(_EMAIL.sub("[email]", text), MAX_VALUE_CHARS)

    def _payload(self, value: Any, depth: int = 0) -> Any:
        if value is None:
            return None
        if isinstance(value, Mapping | list | tuple) and depth >= MAX_DEPTH:
            return TOO_DEEP
        if isinstance(value, Mapping):
            return {
                k: (REDACTED if self._sensitive_key(str(k)) else self._payload(v, depth + 1))
                for k, v in value.items()
            }
        if isinstance(value, list | tuple):
            return [self._payload(v, depth + 1) for v in value]
        return self._scalar(value, depth)


_SEGMENT = re.compile(r"[^a-z0-9]+")


@functools.lru_cache(maxsize=4096)
def _sensitive(key: str) -> bool:
    """Whether an attribute name is sensitive (names repeat: the answer is cached)."""
    segments = _segments(key)
    if SENSITIVE_KEY_WORDS & set(segments):
        return True
    return bool(set(itertools.pairwise(segments)) & SENSITIVE_KEY_PHRASES)


def _segments(key: str) -> tuple[str, ...]:
    """Lower-case word segments of an attribute name (``X-Api-Key`` -> ``x``/``api``/``key``)."""
    return tuple(part for part in _SEGMENT.split(key.lower()) if part)


def _json(text: str) -> dict[str, Any] | list[Any] | None:
    """The object or array ``text`` is the JSON of, or ``None``."""
    if not text.lstrip().startswith(("{", "[")):
        return None
    try:
        held = json.loads(text)
    except ValueError:
        return None
    return held if isinstance(held, dict | list) else None


def _stringify(value: Any) -> str:
    """A value that is not text, as text (``_scalar`` handles text itself)."""
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


#: The redactor the harness uses. Stateless, so one is shared.
DEFAULT = Redactor()


def redact_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    return DEFAULT.redact_attributes(attributes)
