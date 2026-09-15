"""Head sampling (§28). Errors and critical agents are kept even at a low sample rate.

The decision is taken once per execution, in :class:`Sampler.decide`, and carried on the
runtime, so every span of one agent run shares the same fate — a sampled-out run must not
produce a half tree. Sampling is deterministic in the run id, which makes it reproducible
and keeps a retried run on the same side of the decision.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from universal_agent_harness.config.settings import SamplingConfig


@dataclass(frozen=True, slots=True)
class SamplingDecision:
    sampled: bool
    rate: float
    reason: str

    def __bool__(self) -> bool:
        return self.sampled


class Sampler:
    """Deterministic head sampler. ``decide`` is O(1) and allocation-light."""

    def __init__(self, config: SamplingConfig | None = None) -> None:
        self.config = config or SamplingConfig()
        self._critical = frozenset(self.config.critical_agents)

    def decide(self, *, agent_id: str, run_id: str, is_error: bool = False) -> SamplingDecision:
        cfg = self.config
        if is_error:
            return self._roll(run_id, cfg.error_sample_rate, "error")
        if agent_id in self._critical:
            return self._roll(run_id, cfg.critical_agent_sample_rate, "critical_agent")
        return self._roll(run_id, cfg.sample_rate, "default")

    @staticmethod
    def _roll(run_id: str, rate: float, reason: str) -> SamplingDecision:
        return roll(run_id, rate, reason)


def roll(run_id: str, rate: float, reason: str = "default") -> SamplingDecision:
    """Deterministic in ``run_id``: the same run always lands on the same side."""
    if rate >= 1.0:
        return SamplingDecision(True, rate, reason)
    if rate <= 0.0:
        return SamplingDecision(False, rate, reason)
    digest = hashlib.blake2b(run_id.encode(), digest_size=8).digest()
    position = int.from_bytes(digest, "big") / float(1 << 64)
    return SamplingDecision(position < rate, rate, reason)


ALWAYS = Sampler(SamplingConfig(sample_rate=1.0))
