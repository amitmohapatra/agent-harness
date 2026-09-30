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
