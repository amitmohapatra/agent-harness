"""Redaction defaults (§27): nothing sensitive leaves by accident."""

from __future__ import annotations

from universal_agent_harness.telemetry.redaction import DefaultRedactor, NoOpRedactor, reference


def test_sensitive_keys_are_redacted():
    out = DefaultRedactor().redact_attributes(
        {"authorization": "Bearer abc", "api_key": "x", "cookie": "c", "tool.name": "search"}
    )
    assert out["authorization"] == "[redacted]"
    assert out["api_key"] == "[redacted]"
    assert out["cookie"] == "[redacted]"
    assert out["tool.name"] == "search"


def test_credential_shaped_values_are_redacted_under_innocent_keys():
    out = DefaultRedactor().redact_attributes({"note": "sk-abcdefghijklmnopqrstu"})
    assert out["note"] == "[redacted]"


def test_emails_are_masked_and_long_values_truncated():
    redactor = DefaultRedactor(max_value_chars=20)
    out = redactor.redact_attributes({"who": "contact me at a.b@example.com"})
    assert "example.com" not in out["who"]
    long = redactor.redact_attributes({"text": "x" * 100})["text"]
    assert long.startswith("x" * 20) and "truncated" in long


def test_drop_payloads_replaces_content_with_a_reference():
    redactor = DefaultRedactor(drop_payloads=True)
    out = redactor.redact_input({"prompt": "customer SSN is 123"})
    assert out["redacted"] is True
    assert "123" not in str(out)
    assert out["size_bytes"] > 0 and len(out["sha256"]) == 32


def test_nested_payloads_keep_shape_but_drop_secrets():
    out = DefaultRedactor().redact_input({"args": {"password": "hunter2", "q": "stock"}})
    assert out["args"]["password"] == "[redacted]"
    assert out["args"]["q"] == "stock"


def test_noop_redactor_is_transparent():
    assert NoOpRedactor().redact_attributes({"api_key": "x"}) == {"api_key": "x"}


def test_reference_is_content_free():
    ref = reference("secret document body")
    assert "secret" not in str(ref)
    assert ref["type"] == "str"


def test_metric_attributes_are_not_mistaken_for_secrets():
    """A substring rule redacts ``...input_tokens`` because it contains "token"; the
    word-segment rule must not."""
    out = DefaultRedactor().redact_attributes(
        {
            "gen_ai.usage.input_tokens": 11,
            "gen_ai.usage.total_tokens": 18,
            "tool.args.schema": ["sku", "region"],
            "access_token": "abc123def456",
            "x-api-key": "plain",
            "Authorization": "Basic abc",
        }
    )
    assert out["gen_ai.usage.input_tokens"] == 11
    assert out["gen_ai.usage.total_tokens"] == 18
    assert out["tool.args.schema"] == ["sku", "region"]
    assert out["access_token"] == "[redacted]"
    assert out["x-api-key"] == "[redacted]"
    assert out["Authorization"] == "[redacted]"
