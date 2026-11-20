from __future__ import annotations

import re
from pathlib import Path
from typing import Any

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
        "TRELLIS_SCHEDULE_MIN_INTERVAL": "600",
        "TRELLIS_SCHEDULE_MAX_PER_USER": "5",
        "TRELLIS_SCHEDULE_DEFAULT_TIME": "08:30",
        "TRELLIS_SCHEDULE_RUN_TIMEOUT": "300",
        "TRELLIS_SCHEDULE_RUN_MAX_TOOL_CALLS": "20",
        "TRELLIS_SCHEDULE_PAUSE_AFTER_FAILURES": "5",
        "TRELLIS_SCHEDULE_EXPIRE_DAYS": "30",
        "TRELLIS_HOST_EXECUTION": "allow",
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
        "schedule_min_interval": 600.0,
        "schedule_max_per_user": 5,
        "schedule_default_time": "08:30",
        "schedule_run_timeout": 300.0,
        "schedule_run_max_tool_calls": 20,
        "schedule_pause_after_failures": 5,
        "schedule_expire_days": 30,
        "host_execution": "allow",
    }
    assert set(env) == set(settings_module.VARIABLES.values())  # every one, and only them
    assert sourced(env)["api_key"] == ("tk", "TRELLIS_API_KEY")


def test_nothing_set_means_nothing_configured(tmp_path: Path) -> None:
    settings = Settings.from_env({"MEMORY_URL": "  ", "K_SERVICE": "svc"})  # no spool here
    assert settings == Settings()


@pytest.mark.parametrize(
    "retired",
    [
        "TRELLIS_EVAL_SAMPLE",
        "TRELLIS_JUDGE_MODEL",
        "SANDBOX",
        "PROMPTS_DIR",
        "LANGFUSE_HOST",
        "TRELLIS_REQUIRE_SANDBOX",
    ],
)
def test_a_retired_variable_is_not_read(retired: str) -> None:
    assert Settings.from_env({retired: "x", "K_SERVICE": "svc"}) == Settings()


def test_the_schedule_settings_default_to_the_handoffs_and_are_checked() -> None:
    """The handoff's phase 3 defaults: at most hourly, 5 a user, each fired run 10 minutes and
    30 tool calls, paused after 3 failed runs in a row, asked to be kept after 90 days."""
    settings = Settings()
    assert (settings.schedule_min_interval, settings.schedule_max_per_user) == (3600.0, 5)
    assert (settings.schedule_run_timeout, settings.schedule_run_max_tool_calls) == (600.0, 30)
    assert (settings.schedule_pause_after_failures, settings.schedule_expire_days) == (3, 90)
    assert Settings(schedule_max_per_user=0).schedule_max_per_user == 0  # none may be set
    assert Settings().schedule_default_time == "09:00"
    for wrong in (
        {"schedule_min_interval": 0},
        {"schedule_max_per_user": -1},
        {"schedule_default_time": "9am"},
        {"schedule_default_time": "24:00"},
        {"schedule_run_timeout": 0},
        {"schedule_run_max_tool_calls": 0},
        {"schedule_pause_after_failures": 0},
        {"schedule_expire_days": 0},
    ):
        with pytest.raises(ValidationError):
            Settings(**wrong)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="frozen"):
        settings.memory_url = "http://mem"  # type: ignore[misc]


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
        for marker in ("K_SERVICE", "KUBERNETES_SERVICE_HOST", "AWS_LAMBDA_FUNCTION_NAME"):  # docs
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
    for partial in (
        {"LANGFUSE_PUBLIC_KEY": "pk"},
        {"LANGFUSE_BASE_URL": "http://lf"},
        {"LANGFUSE_PUBLIC_KEY": " ", "LANGFUSE_SECRET_KEY": "sk"},
    ):
        assert langfuse_keys(partial) is None
    spaced = {**keys, "LANGFUSE_BASE_URL": " "}  # blank: its cloud
    assert langfuse_keys(spaced) == ("https://cloud.langfuse.com", "pk", "sk")
    assert langfuse_otlp("https://lf.example/", "pk", "sk")[0] == (
        "https://lf.example/api/public/otel"
    )
    assert langfuse_otlp(None, "pk", "sk") == (
        "https://cloud.langfuse.com/api/public/otel",
        {"authorization": "Basic cGs6c2s=", "x-langfuse-ingestion-version": "4"},
    )


def test_the_agents_version_is_service_version_in_the_otel_resource_attributes() -> None:
    from trellis.harness.settings import service_version

    attributes = "deployment.environment=prod,service.version=1.4%2B2"
    assert service_version({"OTEL_RESOURCE_ATTRIBUTES": attributes}) == "1.4+2"
    assert service_version({"OTEL_RESOURCE_ATTRIBUTES": "service.version="}) is None
    assert service_version({}) is None


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


def test_a_deployment_that_denies_wraps_no_shell_agent_without_a_sandbox() -> None:
    """Where the deployment says ``deny``, Claude Code without its sandbox enabled (or with
    commands let out of it: Claude's own default lets them out) and an ACP agent without one are
    refused at ``h.wrap``, naming ``TRELLIS_HOST_EXECUTION``; with one they wrap, and a
    framework that runs no shell of its own is not checked. ``allow``, the default, wraps them
    all."""
    from claude_agent_sdk import ClaudeAgentOptions

    from trellis import Harness
    from trellis.contracts import ConfigurationError
    from trellis.harness.adapters.acp import ACPAgent

    async def plain(input: str) -> str:
        return input

    refused = [
        ClaudeAgentOptions(),
        ClaudeAgentOptions(sandbox={"enabled": True}),
        ClaudeAgentOptions(sandbox={"enabled": True, "allowUnsandboxedCommands": True}),
        ACPAgent(command=("kimi", "acp")),
    ]
    h = Harness(config=Settings(host_execution="deny"), memory=False)
    for n, target in enumerate(refused):
        with pytest.raises(ConfigurationError, match="TRELLIS_HOST_EXECUTION=deny"):
            h.wrap(target, id=f"refused-{n}")
    boxed: Any = {"enabled": True, "allowUnsandboxedCommands": False}
    h.wrap(ClaudeAgentOptions(sandbox=boxed), id="claude-boxed")
    h.wrap(ACPAgent(command=("kimi", "acp"), sandbox=object()), id="kimi-boxed")
    h.wrap(plain, id="no-shell")
    assert set(h.agents) == {"claude-boxed", "kimi-boxed", "no-shell"}
    allowed = Harness(config=Settings(host_execution="allow"), memory=False)
    for n, target in enumerate(refused):
        allowed.wrap(target, id=f"allowed-{n}")
    assert len(allowed.agents) == len(refused)
