from __future__ import annotations

import time

from trellis.contracts import TelemetryRedactor
from trellis.harness.redaction import (
    DEFAULT,
    MAX_DEPTH,
    MAX_VALUE_CHARS,
    REDACTED,
    TOO_DEEP,
    redact_attributes,
)


def test_it_is_the_contracts_redactor() -> None:
    assert isinstance(DEFAULT, TelemetryRedactor)


def test_secret_names_and_credential_values_are_dropped() -> None:
    out = redact_attributes(
        {
            "api_key": "abc",
            "X-Api-Key": "abc",
            "gen_ai.usage.input_tokens": 12,
            "note": "Bearer abcdefghijklmnop",
            "who": "mail me at a@b.co",
        }
    )
    assert out["api_key"] == REDACTED
    assert out["X-Api-Key"] == REDACTED
    assert out["gen_ai.usage.input_tokens"] == 12
    assert out["note"] == REDACTED
    assert out["who"] == "mail me at [email]"


def test_long_values_are_cut_and_payloads_redacted_recursively() -> None:
    assert len(redact_attributes({"text": "x" * 5000})["text"]) < MAX_VALUE_CHARS + 50
    assert DEFAULT.redact_input({"user": {"password": "p", "name": "n"}}) == {
        "user": {"password": REDACTED, "name": "n"}
    }


def test_a_value_of_a_megabyte_is_redacted_in_linear_time() -> None:
    """A tool's output can be large: masking e-mail addresses must not be quadratic in it."""
    started = time.perf_counter()
    out = DEFAULT.redact_output("x" * (1024 * 1024) + " write to ada.l+x@mail.example.com")
    assert time.perf_counter() - started < 1.0
    assert isinstance(out, str) and out.startswith("x" * MAX_VALUE_CHARS + "...[truncated")
    assert DEFAULT.redact_output("x" * 3000 + "@example.com")[:10] == "[email]"


def test_outbound_payloads_are_redacted_like_inbound_ones() -> None:
    payload = {"token": "t", "items": [{"note": "sk-abcdefghijklmnopqrstu"}, ("a@b.co", 3)]}
    assert DEFAULT.redact_output(payload) == {
        "token": REDACTED,
        "items": [{"note": REDACTED}, ["[email]", 3]],
    }
    assert DEFAULT.redact_input(None) is None


def test_attribute_values_of_any_type_are_kept_safe_and_bounded() -> None:
    circular: dict[str, object] = {}
    circular["self"] = circular
    out = redact_attributes(
        {
            "absent": None,
            "input_tokens": 12,
            "password": 7,  # a number is never a credential
            "flag": True,
            "blob": b"\x00\x01\x02",
            "shape": {"user": "a@b.co"},
            "odd": {("a", "b"): 1},  # a key JSON cannot carry
            "loop": circular,
            "tags": ("agent", "react"),
        }
    )
    assert "absent" not in out
    assert out["input_tokens"] == 12 and out["password"] == 7 and out["flag"] is True
    assert out["blob"] == "<3 bytes>"
    assert out["shape"] == '{"user": "[email]"}'
    assert out["odd"] == "{('a', 'b'): 1}"
    assert out["loop"] == str(circular)
    assert out["tags"] == ["agent", "react"]


def test_names_are_sensitive_by_their_words_not_their_substrings() -> None:
    out = redact_attributes(
        {
            "refresh-token": "r",
            "Card.Number": "4111",
            "tokenizer": "bpe",  # contains "token", is not one
            "monkey": "business",  # contains "key", is not one
        }
    )
    assert out == {
        "refresh-token": REDACTED,
        "Card.Number": REDACTED,
        "tokenizer": "bpe",
        "monkey": "business",
    }


def test_a_value_that_contains_itself_is_cut_not_a_crash() -> None:
    """A redactor never raises: a self-referencing list or dict stops at MAX_DEPTH."""
    loop: list[object] = ["x"]
    loop.append(loop)
    cyclic: dict[str, object] = {"name": "run"}
    cyclic["self"] = cyclic

    attributes = redact_attributes({"items": loop})
    payload = DEFAULT.redact_input(cyclic)

    nested = attributes["items"]
    for _ in range(MAX_DEPTH):
        assert nested[0] == "x"
        nested = nested[1]
    assert nested == TOO_DEEP
    inner = payload
    for _ in range(MAX_DEPTH):
        assert inner["name"] == "run"
        inner = inner["self"]
    assert inner == TOO_DEEP


def test_ordinary_nesting_is_untouched_by_the_depth_limit() -> None:
    deep: object = "leaf"
    for _ in range(MAX_DEPTH - 1):
        deep = [deep]
    assert DEFAULT.redact_output({"v": deep}) == {"v": deep}
