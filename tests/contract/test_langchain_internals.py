"""Contract: the few non-public pieces of LangChain and Deep Agents the harness reads, checked
against the installed releases, so a release that changes one fails here — not silently in a
deployment (the harness then warns and runs the graph un-narrowed, ``middleware.narrowing``):

* ``create_agent``'s async model node — its qualified name (``adapters.langgraph.MODEL_NODE``)
  and the composed ``awrap_model_call`` handler it closes over (``middleware.HANDLER``), which
  the per-call tool narrowing re-composes with ``_chain_async_model_call_handlers``;
* Deep Agents' file-state reducer (``automatic.FilesState``);
* ``ContextEditingMiddleware._resolve_token_counter`` (``adaptive.ClearOldResults``'s counter).
"""

from __future__ import annotations

import inspect
from typing import Any

from deepagents.middleware.filesystem import _file_data_reducer
from langchain.agents import create_agent
from langchain.agents.factory import _chain_async_model_call_handlers
from langchain.agents.middleware import ContextEditingMiddleware
from langchain_core.messages.utils import count_tokens_approximately

from tests.support.chat_model import ScriptedChatModel
from trellis.harness.adapters.langgraph import MODEL_NODE, _narrowed
from trellis.harness.middleware import HANDLER


def test_create_agents_model_node_is_of_the_shape_the_narrowing_reads() -> None:
    graph = create_agent(ScriptedChatModel(turns=[]), tools=[])
    node = graph.nodes["model"].bound.afunc  # pyright: ignore[reportAttributeAccessIssue]
    assert node.__qualname__ == MODEL_NODE
    assert HANDLER in node.__code__.co_freevars
    assert _narrowed(graph) is not None
    assert callable(_chain_async_model_call_handlers)


def test_deep_agents_merges_file_states_with_its_reducer() -> None:
    left: Any = {"/a": {"content": ["x"]}}
    right: Any = {"/b": {"content": ["y"]}}
    merged = _file_data_reducer(left, right)
    assert set(merged) == {"/a", "/b"}


def test_context_editing_resolves_its_token_counter_per_request() -> None:
    assert (
        "request" in inspect.signature(ContextEditingMiddleware._resolve_token_counter).parameters
    )
    params = inspect.signature(count_tokens_approximately).parameters
    assert {"use_usage_metadata_scaling", "tools"} <= set(params)
