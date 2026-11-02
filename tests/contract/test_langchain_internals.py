"""Contract: the few non-public pieces of LangChain and Deep Agents the harness reads, checked
against the installed releases, so a release that changes one fails here — not silently in a
deployment (the harness then warns and runs the graph un-narrowed, ``middleware.narrowing``):

* ``create_agent``'s async model node — its qualified name (``adapters.langgraph.MODEL_NODE``)
  and the composed ``awrap_model_call`` handler it closes over (``middleware.HANDLER``), which
  the per-call tool narrowing re-composes with ``_chain_async_model_call_handlers``;
* LangGraph's ``DeltaChannel`` checkpoints — its snapshot type and the metadata counting the
  updates since one (``middleware.RunCheckpointer`` keeps a delta channel's history beside the
  latest checkpoint), Deep Agents' ``messages`` and ``files`` kept so;
* Deep Agents' summarization named ``SummarizationMiddleware`` (``ReAct`` stands one in for it,
  and places it after the harness's narrowing: ``react.SUMMARIZATION``);
* ``ContextEditingMiddleware._resolve_token_counter`` (``adaptive.ClearOldResults``'s counter).
"""

from __future__ import annotations

import inspect
from typing import Any, get_args, get_type_hints

from deepagents.backends import StateBackend
from deepagents.graph import DeepAgentState
from deepagents.middleware.filesystem import FilesystemState
from deepagents.middleware.summarization import create_summarization_middleware
from langchain.agents import create_agent
from langchain.agents.factory import _chain_async_model_call_handlers
from langchain.agents.middleware import ContextEditingMiddleware
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from langgraph.pregel._checkpoint import create_metadata_for_update_state_api

from tests.support.chat_model import ScriptedChatModel
from trellis.harness.adapters.langgraph import MODEL_NODE, _narrowed
from trellis.harness.middleware import HANDLER
from trellis.harness.react import SUMMARIZATION


def test_create_agents_model_node_is_of_the_shape_the_narrowing_reads() -> None:
    graph = create_agent(ScriptedChatModel(turns=[]), tools=[])
    node = graph.nodes["model"].bound.afunc  # pyright: ignore[reportAttributeAccessIssue]
    assert node.__qualname__ == MODEL_NODE
    assert HANDLER in node.__code__.co_freevars
    assert _narrowed(graph) is not None
    assert callable(_chain_async_model_call_handlers)


async def test_the_narrowing_copy_run_outside_a_harness_run_offers_the_graph_as_built() -> None:
    """Outside a run there is nothing to narrow, keep or read back: the copy's model is offered
    the tools the graph was built with, and no reader of the harness's."""
    from langchain_core.tools import tool

    @tool
    def lookup(key: str) -> str:
        """Look a key up."""
        return key

    model = ScriptedChatModel(turns=["done"])
    copy = _narrowed(create_agent(model, tools=[lookup]))
    assert copy is not None
    await copy.ainvoke({"messages": [("user", "x")]})
    assert model.requests[0]["tools"] == ["lookup"]


def test_deep_agents_keeps_messages_and_files_as_delta_channels() -> None:
    for schema, key in ((DeepAgentState, "messages"), (FilesystemState, "files")):
        assert _delta(get_type_hints(schema, include_extras=True)[key])
    assert _DeltaSnapshot([1]).value == [1]
    counted = create_metadata_for_update_state_api({}, set(), prev_metadata=None)
    assert counted == {}  # the counters live under counters_since_delta_snapshot


def _delta(hint: Any) -> bool:
    """Whether a state key's annotation holds a ``DeltaChannel`` (in ``Annotated``, under
    ``Required``/``NotRequired``)."""
    return any(isinstance(m, DeltaChannel) for m in getattr(hint, "__metadata__", ())) or any(
        _delta(arg) for arg in get_args(hint)
    )


def test_deep_agents_summarization_is_named_as_react_replaces_it() -> None:
    made = create_summarization_middleware(ScriptedChatModel(turns=[]), StateBackend())
    assert made.name == SUMMARIZATION


def test_context_editing_resolves_its_token_counter_per_request() -> None:
    assert (
        "request" in inspect.signature(ContextEditingMiddleware._resolve_token_counter).parameters
    )
    params = inspect.signature(count_tokens_approximately).parameters
    assert {"use_usage_metadata_scaling", "tools"} <= set(params)
