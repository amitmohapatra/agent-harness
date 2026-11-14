"""``ReAct`` with a provider's own tools (web search, web fetch, code execution: LangChain's
``dict`` tools): bound to the model as LangChain binds them, each server call the reply reports
recorded as the run's tool call, the catalog's risk deciding whether it is offered, and a
``dict`` a client would have to run refused."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from pydantic import Field

from tests.support.arm import declaring
from tests.support.chat_model import ScriptedChatModel
from trellis import Harness, ReAct
from trellis.contracts import ConfigurationError, RunEventType

SEARCH = {"type": "web_search_20250305", "name": "web_search", "max_uses": 2}
EXECUTE = {"type": "code_execution_20250825", "name": "code_execution"}


class ProviderChat(ScriptedChatModel):
    """A scripted chat model that keeps each binding's tools as given (a provider's ``dict``
    tools are the provider's to read, not LangChain's to convert)."""

    bound: list[list[Any]] = Field(default_factory=list)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, Any]:  # type: ignore[override]
        self.bound.append(list(tools))
        plain = [t for t in tools if not isinstance(t, dict)]
        return super().bind_tools(plain, **kwargs)


def searched(query: str, *, failed: bool = False) -> AIMessage:
    """A reply that searched the web at the provider (Anthropic's content blocks) and answers."""
    result: Any = (
        {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}
        if failed
        else [
            {
                "type": "web_search_result",
                "url": "https://example.com/a",
                "title": "A",
                "encrypted_content": "opaque",
            }
        ]
    )
    return AIMessage(
        content=[
            {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {"query": query},
            },
            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": result},
            {"type": "text", "text": "Found it: example.com/a"},
        ],
        response_metadata={"model_provider": "anthropic"},
    )


def calls(events: list[Any]) -> dict[str, Any]:
    return {e.data["tool"]: e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT}


def offered(model: ProviderChat) -> list[list[str]]:
    return [[t["name"] for t in tools if isinstance(t, dict)] for tools in model.bound]


async def test_a_provider_tool_is_bound_to_the_model_and_its_calls_recorded(
    harness: Harness,
) -> None:
    model = ProviderChat(turns=[searched("trellis harness")])
    agent = harness.wrap(ReAct("You answer with sources.", model, tools=[SEARCH]), id="web")
    events = [e async for e in agent.stream("Search the web for trellis", user="u")]
    assert offered(model) == [["web_search"]]
    assert SEARCH in model.bound[0]  # as given: the provider reads its own settings
    recorded = calls(events)
    assert set(recorded) == {"web_search"} and recorded["web_search"]["status"] == "ok"
    said = next(e for e in events if e.type is RunEventType.TOOL_CALL_ARGS)
    assert said.data["args"] == {"query": "trellis harness"}
    assert "opaque" not in str(recorded["web_search"]["output"])  # the provider's own state


async def test_a_sub_agents_server_calls_are_recorded_too(harness: Harness) -> None:
    helper = {"name": "researcher", "description": "Searches the web.", "system_prompt": "s"}
    model = ProviderChat(
        turns=[
            ("task", {"description": "find trellis", "subagent_type": "researcher"}),
            searched("trellis"),
            "It is at example.com/a",
        ]
    )
    agent = harness.wrap(ReAct("s", model, tools=[SEARCH], subagents=[helper]), id="web")
    events = [e async for e in agent.stream("find trellis", user="u")]
    assert calls(events)["web_search"]["status"] == "ok"


async def test_a_failed_server_call_is_recorded_as_failed(harness: Harness) -> None:
    model = ProviderChat(turns=[searched("x", failed=True)])
    agent = harness.wrap(ReAct("s", model, tools=[SEARCH]), id="web")
    events = [e async for e in agent.stream("search", user="u")]
    assert calls(events)["web_search"]["status"] == "error"


async def test_the_catalogs_risk_decides_whether_a_provider_tool_is_offered(
    harness: Harness,
) -> None:
    # declared read: offered; the catalog says irreversible: a person would have to approve a
    # call the provider makes inside its step, so it is held back, said on the run
    declaring(harness, web_search="irreversible")
    model = ProviderChat(turns=["I cannot search."])
    agent = harness.wrap(ReAct("s", model, tools=[SEARCH, EXECUTE]), id="web")
    events = [e async for e in agent.stream("search", user="u")]
    assert offered(model) == [["code_execution"]]  # a write it keeps to itself: announced
    held = [
        e.data
        for e in events
        if e.type is RunEventType.CUSTOM and e.data.get("code") == "PROVIDER_TOOL_HELD"
    ]
    assert len(held) == 1 and "web_search" in held[0]["message"]


async def test_a_provider_tool_nobody_declared_is_held_until_the_catalog_does(
    harness: Harness,
) -> None:
    unknown = {"type": "mystery_20260101", "name": "mystery"}
    model = ProviderChat(turns=["done", "done"])
    agent = harness.wrap(ReAct("s", model, tools=[unknown]), id="m")
    await agent.run("x", user="u")
    assert offered(model) == [[]]
    declaring(harness, mystery="read")
    await agent.run("x", user="u")
    assert offered(model)[-1] == ["mystery"]


async def test_provider_tools_are_of_the_runs_tool_set(harness: Harness) -> None:
    graph = ReAct("s", ProviderChat(turns=[]), tools=[SEARCH, {"type": "web_search"}])
    native = await harness.wrap(graph, id="web").adapter.native(graph)
    found = {s.name: s.side_effects for s in native}
    assert found["web_search"] == "read"


@pytest.mark.parametrize(
    "given",
    [
        {"type": "bash_20250124", "name": "bash"},
        {"type": "computer_20250124", "name": "computer", "display_width_px": 1024},
        {"type": "function", "function": {"name": "f", "parameters": {}}},
        {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
    ],
)
def test_a_dict_a_client_runs_is_refused(given: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError, match="a tool a client runs"):
        ReAct("s", ProviderChat(turns=[]), tools=[given])


async def test_outside_a_run_a_provider_tool_is_bound_as_langchain_binds_it() -> None:
    model = ProviderChat(turns=["hi"])
    graph = ReAct("s", model, tools=[SEARCH])
    await graph.ainvoke({"messages": [("user", "x")]}, {"configurable": {"thread_id": "t"}})
    assert offered(model) == [["web_search"]]
