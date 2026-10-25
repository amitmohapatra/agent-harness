"""The live suite's shards (``tests/live/shards.py``): each test in exactly one, the longest
first into the shard with the least so far, the same packing for the same tests and
measurements, measurements read from JUnit reports and kept."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.live import shards


def test_every_test_is_in_one_shard_and_the_shards_are_balanced() -> None:
    durations = {"a": 100.0, "b": 60.0, "c": 50.0, "d": 40.0}
    placed = shards.assign(["d", "a", "b", "c", "a"], 2, durations)
    assert placed == {"a": 1, "b": 2, "c": 2, "d": 1}  # 140 s and 110 s
    assert shards.assign(["c", "b", "a", "d"], 2, durations) == placed  # order-free
    unknown = shards.assign(["x", "y", "z"], 3, {})
    assert sorted(unknown.values()) == [1, 2, 3]  # unmeasured: the same weight each
    assert shards.assign([], 4, {}) == {}


@pytest.mark.parametrize("chosen", ["0/2", "3/2", "a/b", "2", ""])
def test_a_shard_is_i_of_n(chosen: str) -> None:
    with pytest.raises(ValueError, match="i/n with 1 <= i <= n"):
        shards.parse(chosen)
    assert shards.parse("2/3") == (2, 3)


def test_durations_are_read_from_junit_reports_and_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "live-1.xml"
    report.write_text(
        '<testsuites><testsuite><testcase classname="tests.live.test_x" name="test_a[1]" '
        'time="12.34"/><testcase classname="tests.live.test_y" name="test_b"/>'
        "</testsuite></testsuites>"
    )
    kept = tmp_path / "durations.json"
    kept.write_text(json.dumps({"tests/live/test_z.py::test_c": 5.0}))
    monkeypatch.setattr(shards, "DURATIONS", kept)
    assert shards.main([str(report)]) == 0
    assert json.loads(kept.read_text()) == {
        "tests/live/test_x.py::test_a[1]": 12.3,
        "tests/live/test_y.py::test_b": 0.0,
        "tests/live/test_z.py::test_c": 5.0,
    }
    assert shards.load(tmp_path / "missing.json") == {}
