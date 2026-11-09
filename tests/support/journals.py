"""What a run's checkpoint holds, read back as the harness reads it (its segments applied)."""

from __future__ import annotations

from typing import Any, Final

from trellis.contracts import RunRecord
from trellis.harness.journal import Journal

#: The most a checkpoint may be (agent-runs' ``RUNS__SERVICE__MAX_PAYLOAD_BYTES``): what a test's
#: store refuses past, as agent-runs does (413).
CHECKPOINT_BOUND: Final = 1024 * 1024


async def journal_of(
    runs: Any, record: RunRecord | dict[str, Any] | None, *, tenant: str | None = None
) -> Journal:
    """The journal a run's record (or a checkpoint, of ``tenant``) holds or names."""
    assert record is not None
    if isinstance(record, dict):
        assert tenant is not None
        return await Journal.read(record, runs.artifacts, tenant=tenant)
    return await Journal.read(record.checkpoint, runs.artifacts, tenant=record.tenant_id)
