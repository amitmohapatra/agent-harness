from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from trellis import Settings
from trellis.harness import settings as settings_module
from trellis.harness.prompts import LangfusePrompts, PromptSources
from trellis.harness.settings import parse_headers, sourced
from trellis.harness.telemetry import Langfuse

ROOT = Path(__file__).resolve().parents[2]
#: The spool's detection, which the suite's fixture replaces for every other test.
DEFAULT_SPOOL = settings_module.default_spool


def test_every_variable_is_read_from_the_environment() -> None:
    settings = Settings.from_env(
        {
            "BIFROST_URL": "http://gw/v1",
            "BIFROST_VIRTUAL_KEY": "vk",
            "TRELLIS_API_KEY": "tk",
            "MEMORY_URL": "http://mem",
            "RUNS_URL": "http://runs",
            "TRELLIS_SPOOL_DIR": "/var/spool/trellis",
            "OTEL_RESOURCE_ATTRIBUTES": "service.version=2026.10.5,deployment.environment=prod",
            "TRELLIS_EVAL_SAMPLE": "0.25",
            "TRELLIS_JUDGE_VIRTUAL_KEY": "eval-vk",
            "SANDBOX": "docker",
        }
    )
    assert settings.eval_sample == 0.25 and settings.judge_virtual_key == "eval-vk"
    assert settings.api_key == "tk" and settings.sandbox == "docker"
    assert settings.spool_dir == "/var/spool/trellis"
    assert settings.runs_url == "http://runs" and settings.agent_version == "2026.10.5"


def test_nothing_set_means_nothing_configured(tmp_path: Path) -> None:
    settings = Settings.from_env({"MEMORY_URL": "  ", "K_SERVICE": "svc"})  # no spool here
    assert settings == Settings()


def test_the_eval_sample_is_a_share_of_runs() -> None:
    assert Settings.from_env({}).eval_sample == 0.1
    assert Settings.from_env({"TRELLIS_EVAL_SAMPLE": "0"}).eval_sample == 0.0
    for bad in ("1.5", "-0.1", "often"):
        with pytest.raises(ValidationError, match="eval_sample"):
            Settings.from_env({"TRELLIS_EVAL_SAMPLE": bad})


@pytest.mark.parametrize(
    ("name", "value", "field", "expected"),
    [
        ("TRELLIS_GROUNDING_SAMPLE", "0.3", "eval_sample", 0.3),
        ("TRELLIS_JUDGE_SAMPLE", "0.4", "eval_sample", 0.4),
        ("TRELLIS_AGENT_VERSION", "v9", "agent_version", "v9"),
        ("TRELLIS_WORKER_CONCURRENCY", "6", "worker_concurrency", 6),
        ("TRELLIS_JUDGE_MODEL", "judges/strong", "judge_model", "judges/strong"),
        ("SANDBOX_IMAGE", "img:1", "sandbox_image", "img:1"),
        ("PROMPTS_DIR", "./prompts", "prompts_dir", "./prompts"),
        ("SKILLS_DIR", "./skills", "skills_dir", "./skills"),
        ("LANGFUSE_HOST", "https://lf", "langfuse_host", "https://lf"),
    ],
)
def test_a_retired_variable_still_works_for_a_release_and_says_its_replacement(
    caplog: pytest.LogCaptureFixture, name: str, value: str, field: str, expected: object
) -> None:
    settings_module._deprecated.cache_clear()
    with caplog.at_level("WARNING", logger="trellis.settings"):
        assert getattr(Settings.from_env({name: value}), field) == expected
        Settings.from_env({name: value})
    said = [r.message for r in caplog.records if r.message.startswith(name)]
    assert len(said) == 1 and settings_module.DEPRECATED[name] in said[0]


def test_the_retired_sample_fields_are_read_as_the_eval_sample(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="trellis.settings"):
        assert Settings(judge_sample=0.5).eval_sample == 0.5  # type: ignore[call-arg]
        assert Settings(grounding_sample=None).eval_sample == 0.1  # type: ignore[call-arg]
    assert "Settings(judge_sample=) is deprecated" in caplog.text


def test_the_new_names_win_over_the_retired_ones() -> None:
    both = Settings.from_env(
        {
            "TRELLIS_EVAL_SAMPLE": "0.2",
            "TRELLIS_JUDGE_SAMPLE": "0.9",
            "LANGFUSE_BASE_URL": "https://new",
            "LANGFUSE_HOST": "https://old",
            "OTEL_RESOURCE_ATTRIBUTES": "service.version=from-otel",
            "TRELLIS_AGENT_VERSION": "old",
        }
    )
    assert (both.eval_sample, both.langfuse_host, both.agent_version) == (
        0.2,
        "https://new",
        "from-otel",
    )


def test_the_spool_defaults_to_the_xdg_state_folder_where_it_lasts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_module._no_spool.cache_clear()
    monkeypatch.setattr(settings_module, "default_spool", DEFAULT_SPOOL)  # the real one
    monkeypatch.setattr(settings_module, "_container", lambda: False)
    lasting = Settings.from_env({"XDG_STATE_HOME": str(tmp_path)})
    assert lasting.spool_dir == str(tmp_path / "trellis")
    assert (
        Settings.from_env({"TRELLIS_SPOOL_DIR": "/x", "XDG_STATE_HOME": str(tmp_path)}).spool_dir
        == "/x"
    )
    with caplog.at_level("WARNING", logger="trellis.settings"):
        for marker in settings_module.EPHEMERAL_MARKERS:
            assert (
                Settings.from_env({marker: "1", "XDG_STATE_HOME": str(tmp_path)}).spool_dir is None
            )
        blocked = tmp_path / "file"
        blocked.write_text("not a folder")
        assert Settings.from_env({"XDG_STATE_HOME": str(blocked)}).spool_dir is None
    assert "will not survive a restart" in caplog.text
    monkeypatch.setattr(settings_module, "_container", lambda: True)
    assert Settings.from_env({"XDG_STATE_HOME": str(tmp_path)}).spool_dir is None


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

    hosted = Settings.from_env({**LANGFUSE_KEYS, "LANGFUSE_BASE_URL": "https://lf.example/"})
    assert hosted.otlp_endpoint == "https://lf.example/api/public/otel"
    assert (scores := Langfuse.of(hosted)) is not None and scores.host == "https://lf.example"


def test_an_otel_exporter_variable_wins_over_the_langfuse_keys() -> None:
    for name in settings_module.OTEL_EXPORTER_ENV:
        collector = Settings.from_env({**LANGFUSE_KEYS, name: "http://otel"})
        assert collector.otlp_endpoint is None and collector.otlp_headers == {}  # the SDK's
        assert Langfuse.of(collector) is not None  # scores and prompts: Langfuse's own names


@pytest.mark.parametrize(
    "env",
    [
        {"LANGFUSE_PUBLIC_KEY": "pk"},
        {"LANGFUSE_SECRET_KEY": "sk"},
        {"LANGFUSE_BASE_URL": "http://lf"},
    ],
    ids=["no-secret", "no-public", "host-only"],
)
def test_without_both_langfuse_keys_nothing_is_derived(env: dict[str, str]) -> None:
    settings = Settings.from_env(env)
    assert settings.otlp_endpoint is None and settings.otlp_headers == {}


def test_env_example_documents_exactly_what_is_read() -> None:
    text = (ROOT / ".env.example").read_text()
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", text, re.M))
    source = (ROOT / "src/trellis/harness/settings.py").read_text()
    read = set(re.findall(r'"([A-Z][A-Z0-9_]+)"', source)) - set(settings_module.OTEL_EXPORTER_ENV)
    read -= {*settings_module.EPHEMERAL_MARKERS, "XDG_STATE_HOME", "OTEL_RESOURCE_ATTRIBUTES"}
    assert read == documented


def test_the_retired_sample_rates_are_read_as_the_larger_of_them() -> None:
    """A retired rate of 0 does not turn the other part off: ``without=`` does that now."""
    env = {"TRELLIS_GROUNDING_SAMPLE": "0", "TRELLIS_JUDGE_SAMPLE": "0.5"}
    assert Settings.from_env(env).eval_sample == 0.5
    found = sourced(env)["eval_sample"]
    assert found == (0.5, "TRELLIS_GROUNDING_SAMPLE and TRELLIS_JUDGE_SAMPLE")
    assert Settings(grounding_sample=0.0, judge_sample=0.3).eval_sample == 0.3  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        Settings.from_env({"TRELLIS_JUDGE_SAMPLE": "lots"})
