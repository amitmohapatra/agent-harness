"""Who owns a conversation's history (``history``): one policy, a registry of RAM-only stores."""

from __future__ import annotations

import pytest
from agents import SQLiteSession
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver

from trellis.harness.history import Owner, in_ram, owner


def test_no_store_leaves_the_history_to_memory() -> None:
    assert owner(None, agent_id="a") is Owner.MEMORY


@pytest.mark.parametrize(
    "store", [InMemorySaver(), MemorySaver(), SQLiteSession("s")], ids=["saver", "alias", "sqlite"]
)
def test_a_store_kept_in_ram_leaves_the_history_to_memory_said_once(
    store: object, caplog: pytest.LogCaptureFixture
) -> None:
    assert in_ram(store)
    with caplog.at_level("WARNING", logger="trellis.harness"):
        assert owner(store, agent_id=f"ram_{id(store)}") is Owner.MEMORY
        assert owner(store, agent_id=f"ram_{id(store)}") is Owner.MEMORY
    said = [r for r in caplog.records if "memory only" in r.getMessage()]
    assert len(said) == 1 and said[0].store == type(store).__name__  # type: ignore[attr-defined]


def test_a_durable_store_owns_the_history(tmp_path: pytest.TempPathFactory) -> None:
    on_disk = SQLiteSession("s", f"{tmp_path}/s.db")
    assert not in_ram(on_disk) and owner(on_disk, agent_id="a") is Owner.FRAMEWORK
    assert owner(object(), agent_id="a") is Owner.FRAMEWORK  # unknown: taken as durable
    on_disk.close()
