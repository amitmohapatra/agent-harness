"""The chatbot example, run end to end.

A real gateway process, a real Memory Service, a real LangGraph compile — the example is
executed, not imported and asserted about. If the flow in the module docstring is wrong, this
fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # examples/ is not a package
    sys.path.insert(0, str(ROOT))

from examples.langgraph_chatbot import build_graph, build_harness  # noqa: E402
from tests.support_gateway import FakeGateway, completion  # noqa: E402

from universal_agent_harness import BifrostModelClient, CollectingEvaluationSink  # noqa: E402

pytestmark = pytest.mark.usefixtures("service_available")


def _gateway_script(body: dict) -> dict:
    """Answer as the two model calls in the graph: a plan, then a reply."""
    if body.get("response_format"):  # the plan node asks for structured output
        return completion(
            json.dumps(
                {
                    "needs_tool": True,
                    "tool": "inventory_db",
                    "args": {"sku": "SKU-1"},
                    "reason": "stock question",
                }
            )
        )
    return completion("SKU-1 is at 3 units against a reorder point of 50, so yes - reorder now.")


CONFIG = {
    "configurable": {
        "thread_id": "chat-test",
        "harness": {"tenant_id": "acme", "user_id": "u-1", "work_id": "ticket-9"},
    }
}


async def test_the_whole_graph_runs_and_every_node_is_exercised(memory) -> None:
    with FakeGateway(_gateway_script) as gateway:
        model = BifrostModelClient(gateway.url, model="gateway-model")
        sink = CollectingEvaluationSink()
        harness = build_harness(model=model, memory=memory, evaluation_sink=sink)
        app = build_graph(harness)

        out = await app.ainvoke(
            {
                "messages": [{"role": "user", "content": "Do we need to reorder SKU-1?"}],
                "trace": [],
            },
            CONFIG,
        )
        await harness.drain()

        # every node ran, and the conditional edge took the tool branch
        assert out["trace"] == ["recall", "plan", "act", "answer", "verify"]
        assert out["plan"]["tool"] == "inventory_db"
        assert out["observations"] == {"sku": "SKU-1", "on_hand": 3, "reorder_point": 50}
        assert "reorder" in out["answer"].lower()
        assert out["grounded"] is True, "3 and 50 both come from the lookup"
        # the assistant turn was appended to the conversation
        assert out["messages"][-1] == {"role": "assistant", "content": out["answer"]}

        # two model calls: the plan and the answer. Nothing else calls the gateway.
        assert len(gateway.requests) == 2
        assert gateway.requests[0]["response_format"]["type"] == "json_schema"
        assert "response_format" not in gateway.requests[1]

        # the harness emitted one evaluation event per node
        assert {e.agent_id for e in sink.events} == {"recall", "plan", "act", "answer", "verify"}

        await harness.aclose()
        await model.aclose()


async def test_the_answer_is_flagged_when_it_states_an_unsupported_number(memory) -> None:
    """The verify node is a real gate, not decoration."""

    def script(body: dict) -> dict:
        if body.get("response_format"):
            return completion(
                json.dumps(
                    {
                        "needs_tool": True,
                        "tool": "inventory_db",
                        "args": {"sku": "SKU-1"},
                        "reason": "stock",
                    }
                )
            )
        return completion("You should order 9999 units before Friday.")

    with FakeGateway(script) as gateway:
        model = BifrostModelClient(gateway.url, model="gateway-model")
        harness = build_harness(model=model, memory=memory)
        out = await build_graph(harness).ainvoke(
            {"messages": [{"role": "user", "content": "How many should we order?"}], "trace": []},
            CONFIG,
        )
        await harness.drain()
        assert out["grounded"] is False, "9999 appears in no lookup result"
        assert out["trace"][-1] == "verify:ungrounded"
        await harness.aclose()
        await model.aclose()


async def test_memory_is_written_for_the_turn_but_not_for_the_routing_node(memory) -> None:
    """``plan`` carries a narrowed MemoryPolicy, so routing chatter stays out of memory."""
    with FakeGateway(_gateway_script) as gateway:
        model = BifrostModelClient(gateway.url, model="gateway-model")
        harness = build_harness(model=model, memory=memory)
        await build_graph(harness).ainvoke(
            {
                "messages": [{"role": "user", "content": "Do we need to reorder SKU-1?"}],
                "trace": [],
            },
            CONFIG,
        )
        await harness.drain()

        writers = {o.get("agent_id") for o in memory.observations}
        assert "plan" not in writers, "the routing node must not write observations"
        assert "answer" in writers
        # the reply reached memory as a claim as well as an observation
        assert any(
            (o.get("hints") or {}).get("memory_type") == "SEMANTIC" for o in memory.observations
        )

        await harness.aclose()
        await model.aclose()
