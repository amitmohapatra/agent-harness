"""Configuration loading and validation (§66, §67)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from trellis.harness import HarnessConfig
from trellis.harness.config.settings import RunsEngine, env_overrides


def test_defaults_are_safe():
    cfg = HarnessConfig.load(env=False)
    assert cfg.memory.enabled and cfg.memory.retrieve_before
    assert cfg.telemetry.enabled
    assert cfg.observability.langfuse.enabled is False
    assert cfg.retries.enabled is False  # retries are opt-in
    assert cfg.telemetry.capture.inputs is False  # no payloads by default
    assert cfg.telemetry.capture.outputs is False
    assert cfg.telemetry.capture.memory_content is False
    assert cfg.telemetry.capture.user_id is False
    assert cfg.memory.failure_mode == "non_blocking"
    assert cfg.observability.failure_mode == "non_blocking"


def test_capture_and_sampling_are_defined_once():
    """Langfuse must not carry a second copy of the capture policy or the sample rate."""
    lf = HarnessConfig.load(env=False).observability.langfuse
    assert not hasattr(lf, "capture")
    assert not hasattr(lf, "sampling")


def test_no_setting_exists_only_to_agree_with_a_provider():
    """A provider is enabled by supplying what it needs, never by a flag repeating it."""
    cfg = HarnessConfig.load(env=False)
    assert not hasattr(cfg, "policy")
    assert not hasattr(cfg, "frameworks")
    # ``registry`` carries the address and credential a client is built from, the same shape
    # as ``observability.langfuse``. What it must never carry is an ``enabled`` flag: that
    # would let configuration claim a registry the deployment has no way to reach, and the
    # first symptom would be a startup failure that reads as an outage.
    assert not hasattr(cfg.registry, "enabled")
    assert cfg.registry.configured is False


def test_document_form_with_a_harness_key(tmp_path):
    path = tmp_path / "harness.yaml"
    path.write_text(
        """
harness:
  memory:
    enabled: false
  timeouts:
    default_seconds: 5
  telemetry:
    sampling:
      sample_rate: 0.1
  observability:
    langfuse:
      enabled: true
      public_key: pk
      secret_key: sk
"""
    )
    cfg = HarnessConfig.load(path, env=False)
    assert cfg.memory.enabled is False
    assert cfg.timeouts.default_seconds == 5
    assert cfg.telemetry.sampling.sample_rate == 0.1
    assert cfg.observability.langfuse.enabled is True


def test_langfuse_without_keys_fails_at_startup():
    with pytest.raises(ValidationError, match="public_key"):
        HarnessConfig.load({"observability": {"langfuse": {"enabled": True}}}, env=False)


def test_langfuse_requires_telemetry():
    with pytest.raises(ValidationError, match=r"telemetry\.enabled"):
        HarnessConfig.load(
            {
                "telemetry": {"enabled": False},
                "observability": {
                    "langfuse": {"enabled": True, "public_key": "p", "secret_key": "s"}
                },
            },
            env=False,
        )


def test_unknown_keys_are_rejected_rather_than_silently_ignored():
    with pytest.raises(ValidationError):
        HarnessConfig.load({"memory": {"retrieve_bfore": True}}, env=False)


def test_documented_environment_variables():
    env = {
        "UAH_MEMORY_ENABLED": "false",
        "UAH_OTEL_ENABLED": "1",
        "UAH_LANGFUSE_ENABLED": "true",
        "LANGFUSE_PUBLIC_KEY": "pk",
        "LANGFUSE_SECRET_KEY": "sk",
        "LANGFUSE_BASE_URL": "https://lf.internal",
        "UAH_DEFAULT_TIMEOUT": "12.5",
        "UAH_SAMPLE_RATE": "0.25",
        "UAH_JUDGE_ENABLED": "true",
        "UAH_JUDGE_SAMPLE_RATE": "0.4",
        "UAH_JUDGE_MAX_USD_PER_HOUR": "0.25",
        "UAH_JUDGE_RUBRIC_PROMPT_ID": "prompt_judge_v2",
        "UAH_RUNS_ENGINE": "temporal",
        "UAH_TEMPORAL_TARGET": "temporal.internal:7233",
        "UAH_TEMPORAL_TASK_QUEUE": "runs",
    }
    cfg = HarnessConfig.model_validate(env_overrides(env))
    assert cfg.memory.enabled is False
    assert cfg.timeouts.default_seconds == 12.5
    lf = cfg.observability.langfuse
    assert lf.enabled and lf.base_url == "https://lf.internal"
    assert cfg.telemetry.sampling.sample_rate == 0.25
    assert cfg.judge.enabled and cfg.judge.sample_rate == 0.4
    assert cfg.judge.max_usd_per_hour == 0.25
    assert cfg.judge.rubric_prompt_id == "prompt_judge_v2"
    assert cfg.runs.engine is RunsEngine.TEMPORAL and cfg.runs.configured
    assert cfg.runs.temporal.target == "temporal.internal:7233"


def test_the_shipped_example_configuration_is_loadable():
    """``harness.example.yaml`` is documented as the list of every setting. A setting that has
    been renamed or removed makes the whole file unloadable, and an example nobody can load is
    worse than no example — so it is validated rather than believed."""
    import yaml

    root = Path(__file__).resolve().parents[2]
    cfg = HarnessConfig.load(yaml.safe_load((root / "harness.example.yaml").read_text()))
    assert cfg.judge.enabled is False, "the example must not switch spending on"
    assert cfg.runs.engine is RunsEngine.AGENT_RUNS
    assert cfg.runs.temporal.task_queue == "trellis-runs"


def test_per_agent_judge_limits_fall_back_to_the_defaults():
    cfg = HarnessConfig.load(
        {
            "judge": {
                "enabled": True,
                "sample_rate": 0.1,
                "max_per_hour": 60,
                "agents": {"noisy": {"sample_rate": 1.0}, "off": {"enabled": False}},
            }
        },
        env=False,
    )
    assert cfg.judge.limits_for("noisy").sample_rate == 1.0
    assert cfg.judge.limits_for("noisy").max_per_hour == 60, "unset means the default"
    assert cfg.judge.limits_for("off").enabled is False
    assert cfg.judge.limits_for("unknown").sample_rate == 0.1


def test_invalid_environment_value_is_reported_with_the_variable_name():
    with pytest.raises(ValueError, match="UAH_DEFAULT_TIMEOUT"):
        env_overrides({"UAH_DEFAULT_TIMEOUT": "soon"})


def test_overrides_beat_file_and_env():
    cfg = HarnessConfig.load(
        {"timeouts": {"default_seconds": 1}},
        env=False,
        overrides={"timeouts": {"default_seconds": 9}},
    )
    assert cfg.timeouts.default_seconds == 9
