"""Configuration loading and validation (§66, §67)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from universal_agent_harness import HarnessConfig
from universal_agent_harness.config.settings import env_overrides


def test_defaults_are_safe():
    cfg = HarnessConfig.load(env=False)
    assert cfg.memory.enabled and cfg.memory.retrieve_before
    assert cfg.telemetry.enabled and cfg.telemetry.provider == "opentelemetry"
    assert cfg.observability.langfuse.enabled is False
    assert cfg.retries.enabled is False  # retries are opt-in (§40)
    assert cfg.telemetry.capture.raw_prompts is False  # no payloads by default (§26)
    assert cfg.telemetry.capture.user_id is False
    assert cfg.memory.failure_mode == "non_blocking"


def test_document_form_with_harness_and_frameworks_keys(tmp_path):
    path = tmp_path / "harness.yaml"
    path.write_text(
        """
harness:
  memory:
    enabled: false
  timeouts:
    default_seconds: 5
  observability:
    langfuse:
      enabled: true
      public_key: pk
      secret_key: sk
      sampling:
        sample_rate: 0.1
frameworks:
  langgraph: true
  crewai: false
"""
    )
    cfg = HarnessConfig.load(path, env=False)
    assert cfg.memory.enabled is False
    assert cfg.timeouts.default_seconds == 5
    assert cfg.observability.langfuse.sampling.sample_rate == 0.1
    assert cfg.frameworks.langgraph is True


def test_langfuse_without_keys_fails_at_startup():
    with pytest.raises(ValidationError, match="public_key"):
        HarnessConfig.load({"observability": {"langfuse": {"enabled": True}}}, env=False)


def test_langfuse_requires_telemetry():
    with pytest.raises(ValidationError, match=r"telemetry\.enabled"):
        HarnessConfig.load(
            {
                "telemetry": {"enabled": False},
                "observability": {"langfuse": {"enabled": True, "public_key": "p", "secret_key": "s"}},
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
        "UAH_LANGFUSE_SAMPLE_RATE": "0.25",
    }
    cfg = HarnessConfig.model_validate(env_overrides(env))
    assert cfg.memory.enabled is False
    assert cfg.timeouts.default_seconds == 12.5
    lf = cfg.observability.langfuse
    assert lf.enabled and lf.base_url == "https://lf.internal" and lf.sampling.sample_rate == 0.25


def test_invalid_environment_value_is_reported_with_the_variable_name():
    with pytest.raises(ValueError, match="UAH_DEFAULT_TIMEOUT"):
        env_overrides({"UAH_DEFAULT_TIMEOUT": "soon"})


def test_overrides_beat_file_and_env():
    cfg = HarnessConfig.load(
        {"timeouts": {"default_seconds": 1}}, env=False, overrides={"timeouts": {"default_seconds": 9}}
    )
    assert cfg.timeouts.default_seconds == 9
