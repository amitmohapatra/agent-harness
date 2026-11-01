from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from trellis import Settings
from trellis.harness import settings as settings_module
from trellis.harness.settings import langfuse_keys, langfuse_otlp, parse_headers, sourced

ROOT = Path(__file__).resolve().parents[2]
#: The spool's detection, which the suite's fixture replaces for every other test.
DEFAULT_SPOOL = settings_module.default_spool


def test_every_variable_is_read_from_the_environment() -> None:
    env = {
        "LLM_BASE_URL": "http://gw/v1",
        "LLM_API_KEY": "vk",
        "LLM_GATEWAY": "bifrost",
        "LLM_AUTH": "bearer",
        "TRELLIS_API_KEY": "tk",
        "MEMORY_URL": "http://mem",
        "RUNS_URL": "http://runs",
        "TRELLIS_SPOOL_DIR": "/var/spool/trellis",
        "TRELLIS_JUDGE_VIRTUAL_KEY": "eval-vk",
        "TRELLIS_FILES_URL": "gs://acme/files",
    }
    settings = Settings.from_env(env)
    assert settings.model_dump() == {
        "llm_base_url": "http://gw/v1",
        "llm_api_key": "vk",
        "llm_gateway": "bifrost",
        "llm_auth": "bearer",
        "api_key": "tk",
        "memory_url": "http://mem",
        "runs_url": "http://runs",
        "spool_dir": "/var/spool/trellis",
        "judge_virtual_key": "eval-vk",
        "files_url": "gs://acme/files",
    }
    assert set(env) == set(settings_module.VARIABLES.values())  # the nine, and only them
    assert sourced(env)["api_key"] == ("tk", "TRELLIS_API_KEY")


def test_nothing_set_means_nothing_configured(tmp_path: Path) -> None:
    settings = Settings.from_env({"MEMORY_URL": "  ", "K_SERVICE": "svc"})  # no spool here
    assert settings == Settings()


@pytest.mark.parametrize(
    "retired",
    ["TRELLIS_EVAL_SAMPLE", "TRELLIS_JUDGE_MODEL", "SANDBOX", "PROMPTS_DIR", "LANGFUSE_HOST"],
)
def test_a_retired_variable_is_not_read(retired: str) -> None:
    assert Settings.from_env({retired: "x", "K_SERVICE": "svc"}) == Settings()


def test_a_setting_that_is_not_one_is_refused() -> None:
    with pytest.raises(ValidationError, match="eval_sample"):
        Settings(eval_sample=1.0)  # type: ignore[call-arg]


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


def test_a_container_is_said_by_its_dockerenv() -> None:
    assert settings_module._container() is Path("/.dockerenv").exists()


def test_otlp_headers_parse_like_the_otel_spec() -> None:
    assert parse_headers("") == {}
    assert parse_headers(" A = b%3Dc , broken, =x ") == {"a": "b=c"}


def test_langfuses_own_variables_name_its_host_and_keys() -> None:
    keys = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}
    assert langfuse_keys(keys) == ("https://cloud.langfuse.com", "pk", "sk")
    hosted = {**keys, "LANGFUSE_BASE_URL": "https://lf.example/"}
    assert langfuse_keys(hosted) == ("https://lf.example", "pk", "sk")
    for partial in ({"LANGFUSE_PUBLIC_KEY": "pk"}, {"LANGFUSE_BASE_URL": "http://lf"}):
        assert langfuse_keys(partial) is None
    assert langfuse_otlp(None, "pk", "sk") == (
        "https://cloud.langfuse.com/api/public/otel",
        {"authorization": "Basic cGs6c2s=", "x-langfuse-ingestion-version": "4"},
    )


def test_env_example_documents_exactly_what_is_read() -> None:
    text = (ROOT / ".env.example").read_text()
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", text, re.M))
    source = (ROOT / "src/trellis/harness/settings.py").read_text()
    read = set(re.findall(r'"([A-Z][A-Z0-9_]+)"', source)) - set(settings_module.OTEL_EXPORTER_ENV)
    read -= {
        *settings_module.EPHEMERAL_MARKERS,
        "XDG_STATE_HOME",
        "XDG_DATA_HOME",
        "OTEL_RESOURCE_ATTRIBUTES",
    }
    assert read == documented
