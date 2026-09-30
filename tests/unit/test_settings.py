from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from trellis import Settings

ROOT = Path(__file__).resolve().parents[2]


def test_every_variable_is_read_from_the_environment() -> None:
    settings = Settings.from_env(
        {
            "TRELLIS_TENANT": "acme",
            "BIFROST_URL": "http://gw/v1",
            "BIFROST_VIRTUAL_KEY": "vk",
            "MEMORY_URL": "http://mem",
            "MEMORY_API_KEY": "mk",
            "TRELLIS_MEMORY_MODEL_KEY": "sk-model",
            "RUNS_URL": "http://runs",
            "RUNS_API_KEY": "rk",
            "TRELLIS_EVAL_SAMPLE": "0.5",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel",
            "LANGFUSE_PUBLIC_KEY": "pk",
            "LANGFUSE_SECRET_KEY": "sk",
            "LANGFUSE_HOST": "http://lf",
        }
    )
    assert settings.tenant == "acme"
    assert settings.eval_sample == 0.5
    assert settings.runs_url == "http://runs"
    assert settings.langfuse_host == "http://lf"


def test_nothing_set_means_nothing_configured() -> None:
    settings = Settings.from_env({"MEMORY_URL": "  "})
    assert settings.tenant == "default"
    assert settings.memory_url is None
    assert settings.eval_sample == 0.1


def test_a_sample_rate_outside_zero_to_one_is_refused() -> None:
    with pytest.raises(ValidationError):
        Settings.from_env({"TRELLIS_EVAL_SAMPLE": "2"})


def test_env_example_documents_exactly_what_is_read() -> None:
    documented = set(
        re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", (ROOT / ".env.example").read_text(), re.M)
    )
    source = (ROOT / "src/trellis/harness/settings.py").read_text()
    read = set(re.findall(r'get\("([A-Z0-9_]+)"\)', source))
    assert read == documented
