"""What a stored prompt's reference is, and the run's pin outside a run."""

from __future__ import annotations

import pytest

from trellis.contracts import ConfigurationError
from trellis.harness.repository import journaled, named, pinned


@pytest.mark.parametrize(
    ("ref", "expected"),
    [("triage", ("triage", None)), ("triage@3", ("triage", "3")), ("a@b@c", ("a", "b@c"))],
)
def test_a_reference_is_a_name_and_maybe_a_version(
    ref: str, expected: tuple[str, str | None]
) -> None:
    assert pinned(ref) == expected
    assert named(*expected) == ref


@pytest.mark.parametrize("ref", ["", "@3", "triage@"])
def test_a_reference_without_a_name_or_after_its_at_is_refused(ref: str) -> None:
    with pytest.raises(ConfigurationError, match="is not a name"):
        pinned(ref)


async def test_outside_a_run_nothing_is_journaled() -> None:
    async def read() -> int:
        return 7

    assert await journaled(None, "k", read, dump=str, load=int) == 7
