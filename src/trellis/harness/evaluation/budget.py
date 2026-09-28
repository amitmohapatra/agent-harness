"""What the online judge is allowed to spend (design §11).

Three ceilings, per agent, all of them cheap to check and none of them a target:

* a **rate** — the fraction of runs judged, rolled on the run id, so the same run is judged
  the same way however many times it is retried;
* a **count** — judgements per rolling hour, which is what stops a traffic spike turning into
  a judging spike;
* a **spend** — dollars per rolling hour, which is the one that matters when a rubric turns
  out to be longer than anyone measured.

All three are enforced *before* the model is called, and the spend is recorded after, so a
judge that is over budget abstains rather than failing.

The spend ceiling is a *brake, not the cap*: it accumulates the price each response quotes, and
a gateway that quotes none (Bifrost in front of OpenRouter, today) leaves it at zero, which
makes ``max_per_hour`` the limit that actually binds. The hard cap belongs on the virtual key,
where the gateway refuses the call — enforcement rather than discipline. ``docs/evaluation.md``
says so where an operator will read it.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from trellis.harness.config.settings import JudgeConfig, JudgeLimits
from trellis.harness.telemetry.sampling import roll

#: One rolling window.
HOUR_SECONDS = 3600.0


@dataclass(frozen=True, slots=True)
class Admission:
    """Whether this run may be judged, and why not when it may not."""

    admitted: bool
    reason: str

    def __bool__(self) -> bool:
        return self.admitted


@dataclass
class _Window:
    """One agent's rolling hour: when it judged, and what it spent."""

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


class JudgeBudget:
    """The three ceilings, per agent. Monotonic clock: a wall-clock jump is not a refill."""

    __slots__ = ("_clock", "_windows", "config")

    def __init__(self, config: JudgeConfig, *, clock: Callable[[], float] | None = None) -> None:
        self.config = config
        self._windows: dict[str, _Window] = {}
        self._clock: Callable[[], float] = clock or time.monotonic

    def limits(self, agent_id: str) -> JudgeLimits:
        return self.config.limits_for(agent_id)

    def admit(self, agent_id: str, run_id: str) -> Admission:
        """Would this run be judged? A question, and nothing more.

        For the caller that wants to skip work early — the interceptor, before it binds
        anything. It takes no slot, so two of these can both say yes; :meth:`reserve` is what
        decides.
        """
        limits = self.limits(agent_id)
        if not limits.enabled:
            return Admission(False, "disabled")
        rate = limits.sample_rate if limits.sample_rate is not None else 0.0
        if not roll(run_id, rate, "judge").sampled:
            return Admission(False, "unsampled")
        window = self._window(agent_id)
        if limits.max_per_hour is not None and len(window.at) >= limits.max_per_hour:
            return Admission(False, "over_max_per_hour")
        if limits.max_usd_per_hour is not None and window.spent >= limits.max_usd_per_hour:
            return Admission(False, "over_budget")
        return Admission(True, "sampled")

    def reserve(self, agent_id: str, run_id: str) -> Admission:
        """Admit this run **and take its slot**, in one go.

        Asking and then recording an hour later is a ceiling that does not hold: judging is
        asynchronous, so between a judge's own check and the verdict it records there are
        awaits, and every concurrent judgement in that gap sees the same free count. With a
        queue of 256 and a ceiling of 60, all 256 pass. Taking the slot at the decision leaves
        no gap to interleave in — there is no ``await`` between the check and the append, which
        is what makes it atomic on an event loop.

        The slot is taken even when the judgement then abstains. That is the honest accounting:
        ``max_per_hour`` bounds *how often the judge runs*, and a judge that ran and could not
        decide still ran.
        """
        admission = self.admit(agent_id, run_id)
        if admission:
            self._window(agent_id).at.append(self._clock())
        return admission

    def spend(self, agent_id: str, usd: float) -> None:
        """What a judgement cost, once it is known.

        Negative is not a discount: a gateway that quotes one would make the spend ceiling
        recede instead of approach, so the accumulator only ever moves one way.
        """
        if usd > 0:
            self._window(agent_id).usd.append((self._clock(), usd))

    def spent(self, agent_id: str) -> float:
        return self._window(agent_id).spent

    def judged(self, agent_id: str) -> int:
        """Judgements in this agent's rolling hour: the number behind ``over_max_per_hour``."""
        return len(self._window(agent_id).at)

    def _window(self, agent_id: str) -> _Window:
        window = self._windows.get(agent_id)
        if window is None:
            window = self._windows[agent_id] = _Window()
        window.trim(self._clock())
        return window


__all__ = ["HOUR_SECONDS", "Admission", "JudgeBudget"]
