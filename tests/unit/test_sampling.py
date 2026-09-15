"""Head sampling (§28)."""

from __future__ import annotations

from universal_agent_harness.config.settings import SamplingConfig
from universal_agent_harness.telemetry.sampling import Sampler, roll


def test_full_and_zero_rates():
    assert roll("run", 1.0).sampled is True
    assert roll("run", 0.0).sampled is False


def test_decision_is_deterministic_in_run_id():
    first = roll("run-abc", 0.5)
    assert all(roll("run-abc", 0.5).sampled == first.sampled for _ in range(20))


def test_rate_is_approximately_honoured():
    kept = sum(roll(f"run-{i}", 0.1).sampled for i in range(2000))
    assert 120 < kept < 280  # ~10% with sampling noise


def test_errors_and_critical_agents_bypass_the_low_rate():
    sampler = Sampler(
        SamplingConfig(sample_rate=0.0, error_sample_rate=1.0, critical_agent_sample_rate=1.0,
                       critical_agents=("billing-agent",))
    )
    assert sampler.decide(agent_id="x", run_id="r").sampled is False
    assert sampler.decide(agent_id="x", run_id="r", is_error=True).sampled is True
    assert sampler.decide(agent_id="billing-agent", run_id="r").sampled is True
