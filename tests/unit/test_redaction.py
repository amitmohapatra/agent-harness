from __future__ import annotations

import json
import time

from trellis.contracts import TelemetryRedactor
from trellis.harness.redaction import (
    DEFAULT,
    MAX_DEPTH,
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
    assert out["who"] == "mail me at a@b.co"  # personal data is the team's to mask


def test_each_secret_word_and_phrase_in_a_name_redacts_it() -> None:
    """A name holding any of the words, or two-word phrases, that mark a secret is redacted,
    whatever separates its parts; a name with none of them is kept."""
    words = [
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
    ]
    phrases = ("api_key", "access-key", "secret_key", "private.key", "session_key", "card_number")
    names = [f"x.{w}" for w in words] + [f"x.{p}" for p in phrases]
    names += ["x.access_token", "x.refresh_token", "x.id_token"]
    out = redact_attributes(dict.fromkeys([*names, "x.cardinal", "x.session"], "v"))
    assert [n for n in names if out[n] != REDACTED] == []
    assert out["x.cardinal"] == out["x.session"] == "v"


def test_long_values_are_cut_and_payloads_redacted_recursively() -> None:
    assert redact_attributes({"text": "x" * 2000})["text"] == "x" * 2000  # 2000: kept whole
    cut = redact_attributes({"text": "x" * 2001})["text"]
    assert cut == "x" * 2000 + "...[truncated 1 chars]"
    assert DEFAULT.redact_input({"user": {"password": "p", "name": "n"}}) == {
        "user": {"password": REDACTED, "name": "n"}
    }


def test_a_value_of_a_megabyte_is_redacted_in_linear_time() -> None:
    """A tool's output can be large: redacting it must not be quadratic in it."""
    started = time.perf_counter()
    out = DEFAULT.redact_output("x" * (1024 * 1024) + " write to ada.l+x@mail.example.com")
    assert time.perf_counter() - started < 1.0
    assert isinstance(out, str) and out.startswith("x" * 2000 + "...[truncated")


def test_outbound_payloads_are_redacted_like_inbound_ones() -> None:
    payload = {"token": "t", "items": [{"note": "sk-abcdefghijklmnopqrstu"}, ("a@b.co", 3)]}
    assert DEFAULT.redact_output(payload) == {
        "token": REDACTED,
        "items": [{"note": REDACTED}, ["a@b.co", 3]],
    }
    assert DEFAULT.redact_input(None) is None


def test_json_text_is_redacted_as_what_it_holds_and_stays_json() -> None:
    """A model's tool-call arguments and a tool result as the model reads it are JSON text."""
    arguments = json.dumps({"email": "ada@example.com", "api_key": "sk-live0123456789abcdef"})
    shown = DEFAULT.redact_input({"function": {"name": "notify", "arguments": arguments}})
    assert json.loads(shown["function"]["arguments"]) == {
        "email": "ada@example.com",
        "api_key": REDACTED,
    }
    result = json.dumps([{"receipt": "sk-live0123456789abcdef"}, "Bearer abcdefghijklmnop"])
    assert json.loads(redact_attributes({"output": result})["output"]) == [
        {"receipt": REDACTED},
        REDACTED,
    ]
    assert DEFAULT.redact_output("[not json") == "[not json"


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
    assert out["shape"] == '{"user": "a@b.co"}'
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


def test_a_credential_inside_longer_text_is_masked_where_it_stands() -> None:
    """A command line, an error message, a tool's arguments: the credential in it goes, the
    rest stays (a span's status message and tool arguments carry such text)."""
    key = "sk-live-0123456789abcdef0123456789abcdef"
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJlMTIz"
    out = redact_attributes(
        {
            "command": f"curl -H 'Authorization: Bearer abc123def456ghi' --key {key} x",
            "error": f"the vendor refused {jwt}: try again",
            "args": json.dumps({"command": f"deploy --token={key}"}),
            "prose": "a task-management board, Bearer authentication, the ask-for-help desk",
        }
    )
    assert out["command"] == "curl -H 'Authorization: [redacted]' --key [redacted] x"
    assert out["error"] == "the vendor refused [redacted]: try again"
    assert (
        key not in out["args"] and json.loads(out["args"])["command"] == "deploy --token=[redacted]"
    )
    assert out["prose"] == "a task-management board, Bearer authentication, the ask-for-help desk"


def _masked(text: str) -> str:
    return redact_attributes({"note": text})["note"]


def test_a_providers_token_is_masked_by_its_documented_shape() -> None:
    """Tokens whose format their provider documents, inside longer text and alone: the
    text around them stays, so the trace still says what happened."""
    secrets = [
        "ghp_" + "a1B2c3D4e5F6g7H8i9J0" * 2,  # GitHub personal access token (classic)
        "gho_" + "Z9y8X7w6V5u4T3s2R1q0" * 2,  # GitHub OAuth
        "ghs_" + "0123456789abcdefghijABCDEFGHIJ012345",  # GitHub server-to-server
        "github_pat_" + "11ABCDEFG0aBcDeFgHiJkL_" + "mNoPqRsTuVwXyZ0123456789" * 2,
        "AKIAIOSFODNN7EXAMPLE",  # AWS access key id (AWS's own documentation example)
        "ASIAY34FZKBOKMUTVV7A",  # AWS temporary access key id
        "xoxb-2048-3520-AbCdEfGhIjKlMnOpQrStUvWx",  # Slack bot token
        "xapp-1-A0123BCDEF-456-abcdef0123456789",  # Slack app-level token
        "AIzaSyA-1234567890abcdefghijklmnopqrstu",  # Google API key (39 characters)
        "sk_live_51H8abcDEFghiJKLmnoPQR",  # Stripe secret key
        "rk_test_51H8abcDEFghiJKLmnoPQR",  # Stripe restricted key
    ]
    for secret in secrets:
        assert _masked(secret) == REDACTED, secret
        said = _masked(f"the call failed with {secret} in its header, retrying")
        assert secret not in said and said.startswith("the call failed with [redacted] in")


def test_a_labelled_secret_inside_text_is_masked_and_its_label_kept() -> None:
    """``password=...`` in a command line, a query string, a log line or a header: the value
    goes, the name stays (it says what was there)."""
    cases = {
        "psql --host db password=hunter2 --port 5432": "psql --host db password=[redacted] --port 5432",
        "GET /v1/items?api_key=abc123def&limit=10": "GET /v1/items?api_key=[redacted]&limit=10",
        "export CLIENT_SECRET='s3cr3t-value'": "export CLIENT_SECRET='[redacted]'",
        "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY": (
            "aws_secret_access_key = [redacted]"
        ),
        'config: {"passwd": "tr0ub4dor&3", "user": "ada"} loaded': (
            'config: {"passwd": "[redacted]", "user": "ada"} loaded'
        ),
        "password: hunter2": "password: [redacted]",
        "it failed (password=hunter2).": "it failed (password=[redacted]).",
        "token=abc123, then retried": "token=[redacted], then retried",
        "secret: correcthorsebatterystaple": "secret: [redacted]",
        "Authorization: Basic dXNlcjpwYXNzd29yZA==": "Authorization: Basic [redacted]",
        "postgresql://ada:s3cr3t@db.internal:5432/app": (
            "postgresql://ada:[redacted]@db.internal:5432/app"
        ),
        "clone https://x-access-token:abcDEF123456@github.com/acme/app.git": (
            "clone https://x-access-token:[redacted]@github.com/acme/app.git"
        ),
    }
    for text, expected in cases.items():
        assert _masked(text) == expected, text


def test_a_private_key_block_is_masked_whole() -> None:
    body = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7\n" * 3
    for kind in ("", "RSA ", "EC ", "OPENSSH ", "ENCRYPTED "):
        block = f"-----BEGIN {kind}PRIVATE KEY-----\n{body}-----END {kind}PRIVATE KEY-----"
        assert _masked(f"key file:\n{block}\nloaded") == "key file:\n[redacted]\nloaded"
    # cut off before its end (a truncated log): masked to the end, never shown in part
    cut = f"-----BEGIN PRIVATE KEY-----\n{body}"
    assert _masked(f"read {cut}") == "read [redacted]"


def test_what_is_no_secret_is_left_as_it_is() -> None:
    """Traces are read to debug runs: a count, an identifier, a hash or a sentence about
    secrets is not a secret, and a public certificate is not a private key."""
    kept = [
        "token_count=512 max_tokens=256 input_tokens: 12",
        "the password policy is: at least twelve characters",
        "Password: must not be empty",
        "token: the unit a model reads text in",
        "password=",
        "password=''",
        "run_id=run_68dc948dd8a84a0ba7e9ca920837962b thread=thr_01M8FHPFHM580TAQJADANY8R7S",
        "commit 39c26a88f0a5dc114f97e51c855dffae1bc29958 on main",
        "id 123e4567-e89b-12d3-a456-426614174000",
        "the region is asiapacificregional1 and the stage is akiaproductionstage1",
        "https://docs.example.com/guide:intro@section",
        "mailto:ada@example.com and http://example.com:8080/path",
        "-----BEGIN CERTIFICATE-----\nMIIDdzCCAl+gAwIBAgIE\n-----END CERTIFICATE-----",
        "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkq\n-----END PUBLIC KEY-----",
        "ghp_short and gho_ and github_pat_ are prefixes GitHub documents",
        "key=value pairs: retries=3, timeout=30s, mode=fast",
        # code and its debugging output: an expression, a count, a format, "it was not set"
        "await ctx.context(question, token_budget=8000)",
        '"input_token_details": {"cache_read": 70, "cache_creation": 20},',
        'token = value.split(" ", 1)[-1].strip()',
        'new_secret = rotated.json()["secret"]',
        "llm_api_key=None, api_key=null, password=false",
        'f"input tokens {tokens:,.0f} + {token_margin:.0%}"',
        "api_key=os.environ[provider.key_variable]",
        "assert secret == expected and token != other",
        "sk_live_ is Stripe's prefix; pk_live_51H8abcDEFghiJKLmnoPQR is publishable",
    ]
    for text in kept:
        assert _masked(text) == text, text


def test_a_secret_that_is_a_number_is_still_a_count_and_one_already_masked_stays_once() -> None:
    assert _masked("token=1234") == "token=1234", "a credential is never a bare number"
    assert _masked("password=[redacted]") == "password=[redacted]"
    assert _masked("Authorization: Bearer abcdefghij1234567890") == "Authorization: [redacted]"


def test_masking_stays_linear_on_text_built_to_slow_it() -> None:
    hostile = [
        "password=" + "a" * 50_000,
        "a=" * 30_000,
        "-----BEGIN PRIVATE KEY-----" + "A" * 60_000,
        "://" + "u:" * 20_000 + "@",
        ("ghp_" + "a" * 30 + " ") * 2_000,
        "x" * 100_000,
    ]
    started = time.perf_counter()
    for text in hostile:
        redact_attributes({"note": text})
    assert time.perf_counter() - started < 2.0
