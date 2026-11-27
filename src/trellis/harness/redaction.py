"""What a trace keeps: span attributes (``telemetry.py``) pass this redactor, and nothing else
does. It masks credentials only; personal data is the domain's to define, so the run's events,
its answer, memory, A2A and the journal carry values as they are, and a team masks what its
users see with an ``on_output`` hook (``trellis.harness.hooks``).

The rules are deliberately boring: drop anything whose *name* looks like a secret, drop
anything whose value looks like a credential, mask a credential inside longer text where it
stands (a token by its provider's documented shape, a private key block, a URL's password,
the value of a name that is a secret's), and cut long values. Text
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

#: The same credentials inside longer text (a command line, an error message): masked where
#: they stand, the rest of the text kept.
_EMBEDDED = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?:sk|pk|rk)-[A-Za-z0-9_\-]{12,}"
    r"|Bearer\s+(?=[A-Za-z0-9._\-]*\d)[A-Za-z0-9._\-]{10,}"
    r"|eyJ[A-Za-z0-9._\-]{20,}"
    r")",
    re.IGNORECASE,
)


#: Tokens whose provider documents their shape, wherever they stand. Case matters here (an
#: AWS key id is upper case; "asiapacific..." is a word), so these are matched apart from
#: the patterns above.
#:
#: * GitHub: ``ghp_`` (personal), ``gho_`` (OAuth), ``ghu_`` / ``ghs_`` (app user / server),
#:   ``ghr_`` (refresh), then at least 36 letters and digits; a fine-grained personal token is
#:   ``github_pat_`` and letters, digits and underscores (GitHub Docs, "About authentication
#:   to GitHub: GitHub's token formats").
#: * AWS: an access key id is ``AKIA`` (long-term) or ``ASIA`` (temporary) and 16 upper-case
#:   letters and digits (IAM User Guide, "IAM identifiers: unique identifiers"). The secret
#:   key has no shape of its own: it is masked by its label (``_LABELLED``).
#: * Slack: ``xoxb-`` (bot), ``xoxp-`` (user), ``xoxa-`` / ``xoxr-`` (configuration and
#:   refresh), ``xoxs-``, and ``xapp-`` (app level) (Slack API, "Token types").
#: * Google: an API key is ``AIza`` and 35 letters, digits, ``-`` or ``_`` (39 in all).
#: * Stripe: ``sk_live_`` / ``sk_test_`` (secret) and ``rk_live_`` / ``rk_test_``
#:   (restricted); ``pk_`` is publishable, and stays (Stripe Docs, "API keys").
_SHAPED = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"gh[pousr]_[A-Za-z0-9]{36,255}"
    r"|github_pat_[A-Za-z0-9_]{22,255}"
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}(?![0-9A-Za-z])"
    r"|xox[abprs]-[A-Za-z0-9\-]{10,255}"
    r"|xapp-[A-Za-z0-9\-]{10,255}"
    r"|AIza[0-9A-Za-z_\-]{35}(?![0-9A-Za-z_\-])"
    r"|[sr]k_(?:live|test)_[A-Za-z0-9]{10,255}"
    r")"
)

#: A private key as files hold it (RFC 7468): the whole block, and one cut off before its end
#: to the end of the text - never shown in part. A certificate or a public key is not one.
_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:[A-Z]{2,12} ){0,2}PRIVATE KEY(?: BLOCK)?-----"
    r"[\s\S]*?(?:-----END (?:[A-Z]{2,12} ){0,2}PRIVATE KEY(?: BLOCK)?-----|\Z)"
)

#: The password of a URL's user (RFC 3986, userinfo): ``scheme://user:password@host``.
_URL_PASSWORD = re.compile(r"(?<=://)([^\s/:@]{1,128}):([^\s/@]{1,512})@(?=[^\s/@])")

#: ``Authorization: Basic <credentials>`` and its kin: the scheme stays, what follows goes
#: (``Bearer`` is masked above, with its word).
_AUTHORIZATION = re.compile(
    r"(?i)(\bauthorization\s{0,3}[:=]\s{0,3}(?:basic|digest|negotiate|token)\s{1,3})[^\s'\",;]{4,2048}"
)

#: ``name=value`` and ``name: value`` inside text - a command line, a query string, a log
#: line, a fragment of JSON. Whether the name is a secret's is the same question as for an
#: attribute's name (``_sensitive``); ``_labelled`` decides what a value must look like.
_LABELLED = re.compile(
    r"(?<![A-Za-z0-9_.\-])(?P<quoted>[\"']?)(?P<name>[A-Za-z][A-Za-z0-9_.\-]{1,63})(?P=quoted)"
    r"(?P<sep>\s{0,3}[=:]\s{0,3})"
    # a quoted value runs to its closing quote; a bare one to the next space or separator
    r"(?:(?P<open>[\"'])(?P<said>[^\"'\n]{1,2048})(?P=open)|(?P<bare>[^\s\"',;&]{1,2048}))"
)
#: A value after a colon is masked only when it reads as a credential, not as a word of a
#: sentence ("Password: must not be empty"): it has a digit or a symbol, or is this long.
_COLON_VALUE_CHARS = 16
#: What may close a sentence, a call or a list right after a bare value.
_CLOSING = ")]}>.,:;!?"
#: A bare value holding one of these is an expression (``settings().key``, ``env["K"]``).
_BRACKETS = frozenset("()[]{}<>")
#: Values that say there is no secret: what a trace is read for when a key was not set.
_NOTHING = frozenset({"none", "null", "nil", "true", "false", "undefined", "unset", "missing"})


class Redactor:
    """The contracts ``TelemetryRedactor`` every span attribute goes through."""

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
        else:
            text = _masked(text)
        return _truncate(text, MAX_VALUE_CHARS)

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


def _masked(text: str) -> str:
    """``text`` with every credential in it masked where it stands, the rest kept: a private
    key block, a token by its shape, a URL's password, an ``Authorization`` header's
    credentials, and the value of a name that is a secret's."""
    if "PRIVATE KEY" in text:
        text = _PRIVATE_KEY.sub(REDACTED, text)
    text = _SHAPED.sub(REDACTED, _EMBEDDED.sub(REDACTED, text))
    if "://" in text:
        text = _URL_PASSWORD.sub(rf"\1:{REDACTED}@", text)
    text = _AUTHORIZATION.sub(rf"\1{REDACTED}", text)
    if "=" in text or ":" in text:
        text = _LABELLED.sub(_labelled, text)
    return text


def _labelled(found: re.Match[str]) -> str:
    """``name=value`` with the value masked when the name is a secret's and the value could
    be one: not a bare number (a count is no credential), not already masked, and after a
    colon not a plain word (``_COLON_VALUE_CHARS``)."""
    if not _sensitive(found["name"]):
        return found[0]
    if found["said"] is not None:
        which, value, after = "said", found["said"], found["open"]
    else:
        # what closes a sentence or a call around the value is not part of it
        which, value = "bare", found["bare"].rstrip(_CLOSING)
        after = found["bare"][len(value) :]
        if _BRACKETS & set(value):  # an expression in code, not a credential
            return found[0]
    if not value or _number(value) or value.lower() in _NOTHING or value.startswith(REDACTED):
        return found[0]
    if value[0] == "=":  # ``secret == other``: a comparison, not an assignment
        return found[0]
    worded = not value[0].isalnum() or (value.isalpha() and len(value) < _COLON_VALUE_CHARS)
    if ":" in found["sep"] and worded:
        return found[0]
    return found[0][: found.start(which) - found.start()] + REDACTED + after


def _number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


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
