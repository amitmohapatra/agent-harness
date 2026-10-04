from __future__ import annotations

from trellis.contracts import TelemetryRedactor
from trellis.harness.redaction import DEFAULT, MAX_VALUE_CHARS, REDACTED, redact_attributes


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
