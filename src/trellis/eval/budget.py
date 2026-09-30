"""What the judge may spend, per agent, all checked before a model is called.

* a **rate** — the fraction of runs judged (``TRELLIS_EVAL_SAMPLE``), rolled on the run id so
  a retried run lands on the same side;
* a **count** — judgements per rolling hour, so a traffic spike is not a judging spike;
* a **spend** — dollars per rolling hour, for a gateway that quotes prices. The hard cap
  belongs on the Bifrost virtual key, where the gateway refuses the call.

:meth:`JudgeBudget.reserve` admits a run *and* takes its slot with no ``await`` in between,
which is what makes the ceiling hold under concurrency.
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

#: One rolling window.
HOUR_SECONDS: Final = 3600.0
#: Judgements per agent per rolling hour.
MAX_PER_HOUR: Final = 60
#: Dollars per agent per rolling hour.
MAX_USD_PER_HOUR: Final = 1.0


@dataclass(frozen=True, slots=True)
class Admission:
    admitted: bool
    reason: str

    def __bool__(self) -> bool:
        return self.admitted


@dataclass
class _Window:
    at: deque[float] = field(default_factory=deque)
    usd: deque[tuple[float, float]] = field(default_factory=deque)

    def trim(self, now: float) -> None:
        cutoff = now - HOUR_SECONDS
        while self.at and self.at[0] < cutoff:
            self.at.popleft()
        while self.usd and self.usd[0][0] < cutoff:
            self.usd.popleft()

    @property
    def spent(self) -> float:
        return sum(amount for _, amount in self.usd)


def sampled(run_id: str, rate: float) -> bool:
    """Deterministic in ``run_id``: the same run always lands on the same side."""
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.blake2b(run_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64) < rate


class JudgeBudget:
    """The three ceilings, per agent. Monotonic clock: a wall-clock jump is not a refill."""

    __slots__ = ("_clock", "_windows", "max_per_hour", "max_usd_per_hour", "sample")

    def __init__(
        self,
        sample: float,
        *,
        max_per_hour: int = MAX_PER_HOUR,
        max_usd_per_hour: float = MAX_USD_PER_HOUR,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.sample = sample
        self.max_per_hour = max_per_hour
        self.max_usd_per_hour = max_usd_per_hour
        self._windows: dict[str, _Window] = {}
        self._clock = clock

    def admit(self, agent_id: str, run_id: str) -> Admission:
        """Would this run be judged? Takes nothing."""
        if not sampled(run_id, self.sample):
            return Admission(False, "unsampled")
        window = self._window(agent_id)
        if len(window.at) >= self.max_per_hour:
            return Admission(False, "over_max_per_hour")
        if window.spent >= self.max_usd_per_hour:
            return Admission(False, "over_budget")
        return Admission(True, "sampled")

    def reserve(self, agent_id: str, run_id: str) -> Admission:
        """Admit this run and take its slot, atomically on the event loop. A judgement that
        then abstains still used its slot: the count bounds how often the judge runs."""
        admission = self.admit(agent_id, run_id)
        if admission:
            self._window(agent_id).at.append(self._clock())
        return admission

    def spend(self, agent_id: str, usd: float) -> None:
        """What a judgement cost. Only ever adds: a negative quote is not a refund."""
        if usd > 0:
            self._window(agent_id).usd.append((self._clock(), usd))

    def judged(self, agent_id: str) -> int:
        return len(self._window(agent_id).at)

    def _window(self, agent_id: str) -> _Window:
        window = self._windows.setdefault(agent_id, _Window())
        window.trim(self._clock())
        return window
