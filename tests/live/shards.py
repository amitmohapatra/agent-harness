"""The live suite in shards: ``LIVE_SHARD=i/n`` runs the i-th of n, each a job that finishes
under the job limit on a CPU model (``make live-shard``).

The tests are packed by how long each took when last measured (``durations.json``, written from
a run's JUnit reports: ``python -m tests.live.shards build/live-*.xml``), the longest first, each
into the shard with the least so far — so the shards take about as long, and every test is in
exactly one. A test with no measurement counts :data:`UNMEASURED` seconds; the packing is
stable for a given set of tests and measurements, so every job agrees on it.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final

#: Where the measured durations are kept, by test id (seconds).
DURATIONS: Final = Path(__file__).with_name("durations.json")
#: The seconds a test with no measurement counts.
UNMEASURED: Final = 120.0


def parse(chosen: str) -> tuple[int, int]:
    """``i/n`` as ``(i, n)``, with ``1 <= i <= n``."""
    index, _, total = chosen.partition("/")
    try:
        which, shards = int(index), int(total)
    except ValueError:
        which, shards = 0, 0
    if not 1 <= which <= shards:
        raise ValueError(f"LIVE_SHARD={chosen}: i/n with 1 <= i <= n")
    return which, shards


def assign(tests: Iterable[str], shards: int, durations: Mapping[str, float]) -> dict[str, int]:
    """Each test's shard (1..``shards``): the longest first, into the shard with the least
    measured time so far (the lowest-numbered on a tie)."""
    totals = [0.0] * shards
    placed: dict[str, int] = {}
    ordered = sorted(set(tests), key=lambda t: (-durations.get(t, UNMEASURED), t))
    for test in ordered:
        least = min(range(shards), key=lambda n: (totals[n], n))
        totals[least] += durations.get(test, UNMEASURED)
        placed[test] = least + 1
    return placed


def load(path: Path | None = None) -> dict[str, float]:
    """The measured durations (:data:`DURATIONS`; none when the file is missing)."""
    path = path or DURATIONS
    if not path.exists():
        return {}
    return {str(k): float(v) for k, v in json.loads(path.read_text()).items()}


def measured(reports: Sequence[Path]) -> dict[str, float]:
    """Each test's seconds in JUnit ``reports`` (pytest's ``--junitxml``), by its test id."""
    found: dict[str, float] = {}
    for report in reports:
        for case in ET.parse(report).getroot().iter("testcase"):
            module = (case.get("classname") or "").replace(".", "/")
            found[f"{module}.py::{case.get('name')}"] = round(float(case.get("time") or 0), 1)
    return found


def main(argv: Sequence[str] | None = None) -> int:
    reports = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    durations = {**load(), **measured(reports)}
    DURATIONS.write_text(json.dumps(dict(sorted(durations.items())), indent=1) + "\n")
    print(f"{len(durations)} tests, {sum(durations.values()) / 3600:.2f} h measured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
