"""What a process used, read from Linux's ``/proc`` (no dependency): its CPU seconds (user +
system), its resident memory now and at its peak. ``None`` where ``/proc`` cannot say (another
system, a process gone)."""

from __future__ import annotations

import os
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Final

#: Clock ticks per second, the unit of ``/proc/<pid>/stat``'s times.
TICKS: Final = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def cpu_seconds(pid: int | None = None) -> float | None:
    """The user and system CPU seconds process ``pid`` (this one by default) has used."""
    try:
        text = Path(f"/proc/{pid or os.getpid()}/stat").read_text()
    except OSError:
        return None
    fields = text.rsplit(")", 1)[1].split()  # after the command name, which may hold spaces
    return (int(fields[11]) + int(fields[12])) / TICKS


def _status(pid: int | None, key: str) -> float | None:
    try:
        lines = Path(f"/proc/{pid or os.getpid()}/status").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith(f"{key}:"):
            return int(line.split()[1]) / 1024  # kB to MB
    return None


def rss_mb(pid: int | None = None) -> float | None:
    """Process ``pid``'s resident memory now, in MB."""
    return _status(pid, "VmRSS")


def peak_rss_mb(pid: int | None = None) -> float | None:
    """Process ``pid``'s highest resident memory so far, in MB."""
    return _status(pid, "VmHWM")


def percentile(values: Sequence[float], q: float) -> float | None:
    """The ``q``-th percentile (0-100) of ``values``, interpolated; ``None`` for none."""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return statistics.quantiles(values, n=100, method="inclusive")[round(q) - 1]
