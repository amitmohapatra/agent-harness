"""What a model's instructions read under the harness: the day it is, after them."""

from __future__ import annotations

from trellis.harness import snapshots
from trellis.harness.middleware import KEPT_SAID


def dated(instructions: str = "") -> str:
    """``instructions`` as the model is sent them: the day it is after them
    (``snapshots.today``), in UTC — a test's run has no app that says its user's zone."""
    day = snapshots.said(snapshots.now(), None)
    return "\n\n".join(part for part in (instructions, day) if part)


def kept_dated(instructions: str = "") -> str:
    """``instructions`` as a run whose results the harness keeps sends them, its tools offered:
    what it is told of a large result (``middleware.KEPT_SAID``), then the day."""
    return dated("\n\n".join(part for part in (instructions, KEPT_SAID) if part))
