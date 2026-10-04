from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from trellis import Settings
from trellis.harness.settings import parse_headers

ROOT = Path(__file__).resolve().parents[2]


def test_every_variable_is_read_from_the_environment() -> None:
    settings = Settings.from_env(
        {
            "BIFROST_URL": "http://gw/v1",
            "BIFROST_VIRTUAL_KEY": "vk",
            "TRELLIS_API_KEY": "tk",
            "MEMORY_URL": "http://mem",
            "RUNS_URL": "http://runs",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel",
            "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Basic%20cGs6c2s=,x-langfuse-host=http://lf",
            "TRELLIS_SPOOL_DIR": "/var/spool/trellis",
            "TRELLIS_WORKER_CONCURRENCY": "6",
            "TRELLIS_GROUNDING_SAMPLE": "0.25",
            "TRELLIS_JUDGE_MODEL": "judges/strong",
            "TRELLIS_JUDGE_VIRTUAL_KEY": "eval-vk",
            "TRELLIS_JUDGE_SAMPLE": "0.5",
        }
    )
    assert settings.grounding_sample == 0.25
    assert (settings.judge_model, settings.judge_virtual_key) == ("judges/strong", "eval-vk")
    assert settings.judge_sample == 0.5
    assert settings.api_key == "tk"
    assert settings.spool_dir == "/var/spool/trellis" and settings.worker_concurrency == 6
    assert settings.runs_url == "http://runs"
    assert settings.otlp_headers == {
        "authorization": "Basic cGs6c2s=",
        "x-langfuse-host": "http://lf",
    }


def test_nothing_set_means_nothing_configured() -> None:
    settings = Settings.from_env({"MEMORY_URL": "  "})
    assert settings == Settings()


def test_the_grounding_sample_is_a_share_of_runs() -> None:
    assert Settings.from_env({}).grounding_sample == 0.1
    assert Settings.from_env({"TRELLIS_GROUNDING_SAMPLE": "0"}).grounding_sample == 0.0
    for bad in ("1.5", "-0.1", "often"):
        with pytest.raises(ValidationError, match="grounding_sample"):
            Settings.from_env({"TRELLIS_GROUNDING_SAMPLE": bad})
        with pytest.raises(ValidationError, match="judge_sample"):
            Settings.from_env({"TRELLIS_JUDGE_SAMPLE": bad})
    assert Settings.from_env({}).judge_sample is None  # the harness decides: 0.1 with judges


def test_otlp_headers_parse_like_the_otel_spec() -> None:
    assert parse_headers("") == {}
    assert parse_headers(" A = b%3Dc , broken, =x ") == {"a": "b=c"}


def test_env_example_documents_exactly_what_is_read() -> None:
    documented = set(
        re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", (ROOT / ".env.example").read_text(), re.M)
    )
    source = (ROOT / "src/trellis/harness/settings.py").read_text()
    read = set(re.findall(r'get\("([A-Z0-9_]+)"\)', source))
    assert read == documented
