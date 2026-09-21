"""A LangGraph support chatbot on the harness, talking to models through Bifrost.

Five nodes, one conditional edge, and every harness configuration surface turned on so you
can see what each one does:

    recall ──▶ plan ──▶ act ──▶ answer ──▶ verify ──▶ END
                 │                 ▲
                 └─────────────────┘   no lookup needed: skip `act`

Run it:

    export BIFROST_BASE_URL=http://localhost:8090/v1   # the gateway
    export BIFROST_API_KEY=vk-...                      # its virtual key
    export BIFROST_MODEL=gpt-4o-mini                   # optional; this is the default
    export MEMORY_SERVICE_URL=http://localhost:8080    # optional; memory is skipped without it
    python examples/langgraph_chatbot.py

The graph is exercised end to end in ``tests/integration/test_langgraph_chatbot.py`` against a
real gateway process and a real Memory Service, so the flow below is tested, not illustrated.
"""

from __future__ import annotations

import asyncio
import json
import operator
import os
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from universal_agent_harness import (
    AgentHarness,
    AgentResponse,
    AllowListPolicyProvider,
    BifrostModelClient,
    Claim,
    CollectingEvaluationSink,
    DefaultRedactor,
    InMemoryArtifactStore,
    MemoryPolicy,
    ModelRequest,
    tool_schemas,
)

# --------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------


class ChatState(TypedDict, total=False):
    """``messages`` is the conversation; everything else is this turn's working set."""

    messages: Annotated[list[dict[str, str]], operator.add]
    context: str
    plan: dict[str, Any]
    observations: dict[str, Any]
    answer: str
    grounded: bool
    trace: Annotated[list[str], operator.add]


# --------------------------------------------------------------------------------------
# Tools — plain async functions. The harness turns them into specs, instruments every call
# and records them in tool memory; nothing here knows that.
# --------------------------------------------------------------------------------------


async def inventory_db(sku: str) -> dict[str, Any]:
    """Current stock level and reorder point for a SKU."""
    table = {"SKU-1": (3, 50), "SKU-2": (180, 40)}
    on_hand, reorder_point = table.get(sku, (0, 0))
    return {"sku": sku, "on_hand": on_hand, "reorder_point": reorder_point}


async def open_orders(sku: str) -> dict[str, Any]:
    """Purchase orders not yet delivered for a SKU."""
    return {"sku": sku, "open_orders": [{"po": "PO-7781", "qty": 25, "eta_days": 9}]}


TOOLS = (inventory_db, open_orders)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_tool": {"type": "boolean"},
        "tool": {"type": ["string", "null"], "enum": ["inventory_db", "open_orders", None]},
        "args": {"type": "object"},
        "reason": {"type": "string"},
    },
    "required": ["needs_tool", "tool", "args", "reason"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------


def build_harness(
    *,
    model: Any,
    memory: Any = None,
    evaluation_sink: Any = None,
) -> AgentHarness:
    """Every configuration section, set explicitly.

    Most deployments set three or four of these and leave the rest alone — the defaults are
    the safe ones (no payload capture, no retries, non-blocking observability). They are all
    spelled out here so the surface is visible in one place.
    """
    return AgentHarness(
        # -- providers: passing one is what enables it; there is no second "enabled" flag --
        model=model,
        memory=memory,
        tools=list(TOOLS),
        artifacts=InMemoryArtifactStore(),
        evaluation_sink=evaluation_sink,
        policy=AllowListPolicyProvider(
            agents={"recall", "plan", "act", "answer", "verify"},
            tools={"inventory_db", "open_orders"},
            models={"gpt-4o-mini", "gateway-model"},
            tenants={"acme"},
        ),
        redactor=DefaultRedactor(),
        # -- identity every run inherits unless the call overrides it --------------------
        defaults={"tenant_id": "acme", "user_id": "u-1", "agent_group_id": "support-crew"},
        config={
            "memory": {
                "enabled": True,
                "retrieve_before": True,  # fetch a bundle before each node that names a query
                "observe_input": True,
                "observe_output": True,
                "observe_claims": True,
                "record_outcome": True,
                "observe_tool_results": False,  # tool output often carries customer data
                "private_by_default": False,
                "record_messages": True,  # also store the turn as chat messages
                "token_budget": 1200,
                "writeback": True,  # writes happen after the answer is returned
                "failure_mode": "non_blocking",  # memory down => warning, not a failed chat
            },
            "telemetry": {
                "enabled": True,
                "service_name": "support-chatbot",
                "configure_sdk": False,  # the app owns the OTel provider
                "exporter": "none",
                "metrics_enabled": True,
                "capture": {
                    "inputs": True,
                    "outputs": True,
                    "memory_content": False,  # the most sensitive of the three
                    "user_id": False,
                    "thread_id": True,
                },
                "sampling": {
                    "sample_rate": 1.0,
                    "error_sample_rate": 1.0,
                    "critical_agent_sample_rate": 1.0,
                    "critical_agents": ["answer"],
                },
            },
            "observability": {
                "structured_logging": True,
                "log_level": "INFO",
                "failure_mode": "non_blocking",
                # "langfuse": {"enabled": True, "public_key": ..., "secret_key": ...}
                # rides on the same spans and obeys telemetry.capture above.
            },
            "models": {"default_model": os.environ.get("BIFROST_MODEL", "gpt-4o-mini")},
            "tools": {
                # build the tool-sequence memory for this task shape
                "record_to_memory": True,
            },
            "retries": {
                "enabled": True,
                "max_attempts": 3,
                "backoff_seconds": 0.2,
                # only these, and only for nodes marked idempotent=True
                "retry_categories": ["TIMEOUT", "RATE_LIMIT", "DEPENDENCY"],
            },
            "timeouts": {
                "default_seconds": 45.0,  # per node
                # Retrieval is embedding + vector search + rerank; on a cold or CPU-bound
                # service that is seconds, not milliseconds. Too tight a value here does not
                # fail the chat — failure_mode is non_blocking — it just silently answers
                # without context, which is worse than answering a little later.
                "memory_seconds": 10.0,
                "model_seconds": 30.0,
                "tool_seconds": 10.0,
            },
            "artifacts": {"enabled": True, "inline_max_bytes": 8_000},
            "evaluation_events": {"enabled": True, "synchronous": False, "sample_rate": 1.0},
        },
    )


# --------------------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------------------


def build_graph(harness: AgentHarness) -> Any:
    model_name = harness.config.models.default_model
    specs = harness.langgraph.tool_specs(list(TOOLS))

    @harness.langgraph.agent(
        agent_id="recall",
        skills=["support.context"],
        # what memory retrieval searches for: the last message in the conversation
        query="messages",
        # a read-only node: it is safe to run again if it times out
        idempotent=True,
    )
    async def recall(state: ChatState, agent: Any) -> dict[str, Any]:
        """The bundle was already fetched by the time this node body runs."""
        bundle = agent.memory_context
        rendered = getattr(bundle, "rendered", "") if bundle is not None else ""
        agent.log("recall.done", has_context=bool(rendered))
        return {"context": rendered, "trace": ["recall"]}

    @harness.langgraph.agent(
        agent_id="plan",
        skills=["support.routing"],
        query="messages",
        memory_policy=MemoryPolicy(
            # routing decisions are noise in long-term memory: read context, write nothing
            observe_input=False,
            observe_output=False,
            observe_claims=False,
        ),
        idempotent=True,
    )
    async def plan(state: ChatState, agent: Any) -> dict[str, Any]:
        """One structured model call that decides whether a tool is needed."""
        question = state["messages"][-1]["content"]
        reply = await agent.model.structured(
            ModelRequest(
                model=model_name,
                messages=[
                    {"role": "system", "content": _PLAN_PROMPT},
                    {"role": "user", "content": _plan_input(question, state.get("context", ""))},
                ],
                tools=tool_schemas(specs),
            ),
            schema=PLAN_SCHEMA,
        )
        decision = reply.data or {"needs_tool": False, "tool": None, "args": {}, "reason": ""}
        agent.log("plan.decided", needs_tool=decision["needs_tool"], tool=decision["tool"])
        return {"plan": decision, "trace": ["plan"]}

    @harness.langgraph.agent(
        agent_id="act",
        skills=["support.lookup"],
        # Not for retrieval — for tool memory. Without a query the node's request carries no
        # text, and the tool invocation is keyed on LangGraph's per-superstep task UUID: a
        # different key every run, so no procedure is ever mined from these calls.
        query="messages",
        idempotent=True,
    )
    async def act(state: ChatState, agent: Any) -> dict[str, Any]:
        """Run the tool the plan named, through the harness so it is instrumented."""
        decision = state["plan"]
        outcome = await agent.tools.call(decision["tool"], **decision["args"])
        if not outcome.ok:
            # a failed lookup is not a failed conversation: say so and answer without it
            agent.log("act.failed", tool=decision["tool"], error=outcome.error_class)
            return {
                "observations": {"error": outcome.output_summary or outcome.error_class},
                "trace": ["act:failed"],
            }
        return {"observations": outcome.output, "trace": ["act"]}

    @harness.langgraph.agent(
        agent_id="answer",
        skills=["support.answering"],
        query="messages",
        # not idempotent: re-running it would bill a second completion
        idempotent=False,
    )
    async def answer(state: ChatState, agent: Any) -> AgentResponse:
        """The customer-facing reply, returned as a result so it can carry a claim."""
        reply = await agent.model.invoke(
            ModelRequest(
                model=model_name,
                messages=[
                    {"role": "system", "content": _ANSWER_PROMPT},
                    *state["messages"],
                    {"role": "user", "content": _answer_input(state)},
                ],
            )
        )
        text = (reply.text or "").strip()
        return AgentResponse.ok(
            {
                "answer": text,
                "messages": [{"role": "assistant", "content": text}],
                "trace": ["answer"],
            },
            claims=[
                Claim(
                    claim_id=agent.idempotency_key("claim", text),
                    text=text,
                    confidence=0.8,
                )
            ],
        )

    @harness.langgraph.agent(agent_id="verify", skills=["support.grounding"], idempotent=True)
    async def verify(state: ChatState, agent: Any) -> dict[str, Any]:
        """Cheap grounding check: every number in the answer must appear in what we looked up.

        No model call — a verifier that costs as much as the answer does not get deployed.
        """
        facts = json.dumps(state.get("observations", {}))
        numbers = set(_numbers(state.get("answer", "")))
        unsupported = sorted(n for n in numbers if n not in facts)
        grounded = not unsupported
        agent.log("verify.done", grounded=grounded, unsupported=unsupported)
        return {"grounded": grounded, "trace": ["verify" if grounded else "verify:ungrounded"]}

    def needs_tool(state: ChatState) -> str:
        return "act" if state.get("plan", {}).get("needs_tool") else "answer"

    graph = StateGraph(ChatState)
    for node in (recall, plan, act, answer, verify):
        graph.add_node(node)
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "plan")
    graph.add_conditional_edges("plan", needs_tool, {"act": "act", "answer": "answer"})
    graph.add_edge("act", "answer")
    graph.add_edge("answer", "verify")
    graph.add_edge("verify", END)
    return graph.compile(checkpointer=InMemorySaver())


# --------------------------------------------------------------------------------------
# Prompts and small helpers
# --------------------------------------------------------------------------------------

_PLAN_PROMPT = (
    "You route a support question. Decide whether one lookup tool is needed to answer it. "
    "Answer only with the JSON schema you were given."
)
_ANSWER_PROMPT = (
    "You are an inventory support assistant. Answer in two sentences. "
    "Use only the facts provided; if a number is not in them, do not state it."
)


def _plan_input(question: str, context: str) -> str:
    return f"Known context:\n{context or '(none)'}\n\nQuestion: {question}"


def _answer_input(state: ChatState) -> str:
    return (
        f"Known context:\n{state.get('context') or '(none)'}\n\n"
        f"Lookup results:\n{json.dumps(state.get('observations', {}))}"
    )


def _numbers(text: str) -> list[str]:
    return [
        "".join(c for c in token if c.isdigit())
        for token in text.split()
        if any(c.isdigit() for c in token)
    ]


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


async def main() -> None:
    base_url = os.environ.get("BIFROST_BASE_URL")
    if not base_url:
        raise SystemExit(
            "set BIFROST_BASE_URL (e.g. http://localhost:8080/v1) — and BIFROST_API_KEY if "
            "your gateway requires one"
        )
    model = BifrostModelClient(
        base_url,
        api_key=os.environ.get("BIFROST_API_KEY"),
        model=os.environ.get("BIFROST_MODEL", "gpt-4o-mini"),
        timeout=30.0,
        max_retries=2,
        circuit_failure_threshold=5,
        circuit_open_seconds=15.0,
        default_params={"temperature": 0.0, "max_tokens": 400},
    )

    memory = None
    if memory_url := os.environ.get("MEMORY_SERVICE_URL"):
        from universal_memory import MemoryClient  # noqa: PLC0415 - optional extra

        memory = MemoryClient(memory_url, api_key=os.environ.get("MEMORY_API_KEY", "dev-key"))

    sink = CollectingEvaluationSink()
    harness = build_harness(model=model, memory=memory, evaluation_sink=sink)
    app = build_graph(harness)
    config = {
        "configurable": {
            # the examples keep their own thread namespace: running one must not
            # take ownership of a thread id the test suite uses on the same service
            "thread_id": "example-chatbot",
            # identity the graph carries into every node's execution context
            "harness": {"tenant_id": "acme", "user_id": "u-1", "work_id": "ticket-9"},
        }
    }

    for question in ("Do we need to reorder SKU-1?", "And what is already on order?"):
        out = await app.ainvoke(
            {"messages": [{"role": "user", "content": question}], "trace": []}, config
        )
        print(f"\nQ: {question}")
        print(f"A: {out['answer']}")
        print(f"   grounded={out['grounded']} path={out['trace']}")

    await harness.drain()  # let the memory writeback finish before we exit
    print(f"\nevaluation events: {len(sink.events)}")
    await harness.aclose()
    await model.aclose()
    if memory is not None:
        await memory.aclose()


if __name__ == "__main__":
    asyncio.run(main())
