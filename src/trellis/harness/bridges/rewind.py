"""A run's checkpoint history read after the run, and a run rewound: LangGraph's time travel
(``get_state_history``, ``update_state``, a thread continued from an earlier checkpoint) on a
graph that keeps its checkpoints in its run (``middleware.RunCheckpointer``: ``ReAct``, a
``middleware()`` graph), for ``Agent.checkpoints`` and ``Agent.rewind``.

Outside a run's attempt the checkpointer is given the run's checkpoints as the run store keeps
them (``RunCheckpointer.bound``) and LangGraph's own API does the reading and the updating:
nothing of a checkpoint's format is known here."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Final

from trellis.harness.journal import Journal

#: How many checkpoints of a run's history are read back at once: a long run's history is
#: listed a page at a time, each page's checkpoints the only ones deserialized.
CHECKPOINTS_PAGE: Final = 20


async def checkpoints(target: Any, run_id: str, journal: Journal) -> AsyncIterator[Any]:
    """The checkpoints ``journal`` keeps of the run ``run_id`` of the graph ``target`` (one
    that checkpoints into its run: ``adapters.langgraph.run_scoped``), the
    newest first, as LangGraph's ``StateSnapshot``s (``aget_state_history``), read
    :data:`CHECKPOINTS_PAGE` at a time."""
    saver, graph = target.checkpointer, journal.graph or {}
    config: Any = {"configurable": {"thread_id": run_id}}
    before = None
    while True:
        with saver.bound(run_id, graph):
            found = target.aget_state_history(config, before=before, limit=CHECKPOINTS_PAGE)
            page = [state async for state in found]
        for state in page:
            yield state
        if len(page) < CHECKPOINTS_PAGE:
            return
        before = page[-1].config


async def rewound(
    target: Any, journal: Journal, run_id: str, checkpoint_id: str, values: Any
) -> Journal | None:
    """The journal the run ``run_id`` starts with to continue the graph ``target`` from the
    checkpoint ``checkpoint_id`` of the run ``journal`` is of (``Journal.rewound``), ``values``
    applied to that state as LangGraph's ``update_state`` applies them; ``None``: no
    checkpoint of that run."""
    saver = target.checkpointer
    graph = saver.rewound(journal.graph or {}, checkpoint_id)
    if graph is None:
        return None
    if values is not None:
        with saver.bound(run_id, graph):
            await target.aupdate_state({"configurable": {"thread_id": run_id}}, values)
    return journal.rewound(graph, checkpoint_id)
