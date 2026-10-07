from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from trellis import Settings
from trellis.harness.prompts import LangfusePrompts, PromptSources
from trellis.harness.settings import parse_headers
from trellis.harness.telemetry import Langfuse

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
            "TRELLIS_AGENT_VERSION": "2026.10.5",
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
    assert settings.runs_url == "http://runs" and settings.agent_version == "2026.10.5"
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


LANGFUSE_KEYS = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}
#: ``Basic base64("pk:sk")``, as Langfuse's SDK sends its key pair.
BASIC = "Basic cGs6c2s="


def test_langfuse_keys_alone_export_traces_and_scores_to_langfuse() -> None:
    settings = Settings.from_env(LANGFUSE_KEYS)
    assert settings.otlp_endpoint == "https://cloud.langfuse.com/api/public/otel"
    assert settings.otlp_headers == {
        "authorization": BASIC,
        "x-langfuse-ingestion-version": "4",
    }
    scores = Langfuse.of(settings)
    assert scores is not None and scores.host == "https://cloud.langfuse.com"
    (prompts,) = PromptSources.of(settings).sources
    assert isinstance(prompts, LangfusePrompts) and prompts.host == scores.host

    hosted = Settings.from_env({**LANGFUSE_KEYS, "LANGFUSE_HOST": "https://lf.example/"})
    assert hosted.otlp_endpoint == "https://lf.example/api/public/otel"
    assert (scores := Langfuse.of(hosted)) is not None and scores.host == "https://lf.example"


def test_explicit_otlp_settings_win_over_the_langfuse_keys() -> None:
    collector = Settings.from_env({**LANGFUSE_KEYS, "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel"})
    assert collector.otlp_endpoint == "http://otel" and collector.otlp_headers == {}
    assert Langfuse.of(collector) is None  # a collector is not told Langfuse's credentials
    assert collector.langfuse_public_key == "pk"  # prompts still read from Langfuse

    headed = Settings.from_env(
        {**LANGFUSE_KEYS, "OTEL_EXPORTER_OTLP_HEADERS": "x-langfuse-ingestion-version=3,a=b"}
    )
    assert headed.otlp_endpoint == "https://cloud.langfuse.com/api/public/otel"
    assert headed.otlp_headers == {
        "authorization": BASIC,
        "x-langfuse-ingestion-version": "3",
        "a": "b",
    }


@pytest.mark.parametrize(
    "env",
    [{"LANGFUSE_PUBLIC_KEY": "pk"}, {"LANGFUSE_SECRET_KEY": "sk"}, {"LANGFUSE_HOST": "http://lf"}],
    ids=["no-secret", "no-public", "host-only"],
)
def test_without_both_langfuse_keys_nothing_is_derived(env: dict[str, str]) -> None:
    settings = Settings.from_env(env)
    assert settings.otlp_endpoint is None and settings.otlp_headers == {}


def test_the_otel_form_still_reaches_langfuse_as_the_keys_do() -> None:
    otel = Settings.from_env(
        {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "https://cloud.langfuse.com/api/public/otel",
            "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Basic%20cGs6c2s=",
        }
    )
    via_otel, via_keys = Langfuse.of(otel), Langfuse.of(Settings.from_env(LANGFUSE_KEYS))
    assert via_otel is not None and via_keys is not None
    assert via_otel.host == via_keys.host
    assert otel.otlp_headers["authorization"] == BASIC


def test_env_example_documents_exactly_what_is_read() -> None:
    documented = set(
        re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", (ROOT / ".env.example").read_text(), re.M)
    )
    source = (ROOT / "src/trellis/harness/settings.py").read_text()
    read = set(re.findall(r'get\("([A-Z0-9_]+)"\)', source))
    assert read == documented
