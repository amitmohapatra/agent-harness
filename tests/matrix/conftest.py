"""``MATRIX_SHARD=i/n`` runs the i-th of n shards of the matrix (by a stable hash of each cell's
id), so CI can split it across jobs; every cell is in exactly one shard."""

from __future__ import annotations

import os
import zlib

import pytest


def shard_of(node_id: str, shards: int) -> int:
    return zlib.crc32(node_id.encode()) % shards + 1


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    chosen = os.environ.get("MATRIX_SHARD", "").strip()
    if not chosen:
        return
    index, _, total = chosen.partition("/")
    which, shards = int(index), int(total)
    if not 1 <= which <= shards:
        raise pytest.UsageError(f"MATRIX_SHARD={chosen}: i/n with 1 <= i <= n")
    kept, dropped = [], []
    for item in items:
        if "matrix" in item.keywords and shard_of(item.nodeid, shards) != which:
            dropped.append(item)
        else:
            kept.append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept
