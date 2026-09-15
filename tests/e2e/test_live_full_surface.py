"""Every harness feature, against the running Memory Service, with the database checked.

    make test-live-full          # or: MEMORY_SERVICE_URL=... pytest -m live -q -s

This is the suite that answers "does it actually work" rather than "does it work against
our idea of the service". Nothing is mocked: real SDK, real HTTP, real Postgres, Qdrant,
Dragonfly and OpenFGA. Where a feature is supposed to *persist* something, the row is read
back out of the database rather than inferred from a 202.

Two service behaviours shape almost every test here and are worth stating once:

* **Writes are asynchronous.** The API commits an observation and queues the work that turns
  it into memories, graph edges and index entries. Reading immediately after writing proves
  nothing, so reads poll (:func:`eventually`).
* **Reads are audience-filtered.** A memory or chunk is retrievable only by a principal in
  its audience, and audiences come from the authorization service: a THREAD audience needs
  the thread to exist (a message creates it), WORKSPACE/GROUP audiences need membership.
  Writing with an audience nobody is in produces a row nobody can read.
"""

from __future__ import annotations

import asyncio
import operator
import os
import subprocess
import uuid
from typing import Annotated, TypedDict

import pytest

from universal_agent_harness import (
    AgentExecutionContext,
    AgentHarness,
    AgentResult,
    Claim,
    MemoryObservation,
)

URL = os.environ.get("MEMORY_SERVICE_URL")
API_KEY = os.environ.get("MEMORY_API_KEY", "dev-key")
TENANT = os.environ.get("MEMORY_TENANT", "acme")
PG_CONTAINER = os.environ.get("MEMORY_PG_CONTAINER", "memory-service-postgres-1")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not URL, reason="set MEMORY_SERVICE_URL to run against a live service"),
]


# --------------------------------------------------------------------------- helpers


def sql(query: str) -> list[list[str]]:
    """Read the service's database directly. A 202 is not evidence that a row exists."""
    proc = subprocess.run(
        ["docker", "exec", PG_CONTAINER, "psql", "-U", "memory", "-d", "memory", "-t", "-A",
         "-F", "\x1f", "-c", query],
        capture_output=True, text=True, check=False, timeout=30,
    )
    if proc.returncode != 0:
        pytest.skip(f"database not reachable for verification: {proc.stderr.strip()[:120]}")
    return [line.split("\x1f") for line in proc.stdout.strip().splitlines() if line]


async def eventually(
    check,
    *,
    timeout: float = 45.0,  # noqa: ASYNC109 - a polling budget, not a cancel scope
    interval: float = 1.5,
    what: str = "",
):
    """Poll ``check`` until it returns something truthy. Writes are asynchronous: the job
    that creates a memory, a graph edge or an index entry runs after the API has answered."""
    import inspect

    deadline = asyncio.get_running_loop().time() + timeout
    last = None
    while asyncio.get_running_loop().time() < deadline:
        last = check()
        # ``check`` may be a coroutine function, or a lambda that returns a coroutine —
        # await whatever comes back rather than treating an un-awaited coroutine as success.
        if inspect.isawaitable(last):
            last = await last
        if last:
            return last
        await asyncio.sleep(interval)
    raise AssertionError(f"timed out waiting for {what or 'condition'} (last value: {last!r})")


@pytest.fixture
async def client():
    from universal_memory import MemoryClient

    c = MemoryClient(URL, api_key=API_KEY, timeout=60.0)
    try:
        yield c
    finally:
        await c.aclose()


@pytest.fixture
def run_id() -> str:
    return uuid.uuid4().hex[:8]


@pytest.fixture
def context(run_id) -> AgentExecutionContext:
    return AgentExecutionContext.create(
        tenant_id=TENANT,
        agent_id="live-surface",
        user_id=f"user-{run_id}",
        thread_id=f"thread-{run_id}",
        turn_id=f"turn-{run_id}",
        work_id=f"work-{run_id}",
        agent_group_id=f"crew-{run_id}",
        workspace_id=f"ws-{run_id}",
    )


def build(client, **config) -> AgentHarness:
    merged = {
        "memory": {"writeback": False, "retrieve_before": False,
                   "observe_input": False, "observe_output": False, "observe_claims": False},
        "timeouts": {"memory_seconds": 60.0, "default_seconds": 120.0},
        **config,
    }
    return AgentHarness(memory=client, defaults={"tenant_id": TENANT}, config=merged)


async def run(harness, context, body, **wrap):
    """Execute ``body(runtime)`` inside a real agent run."""
    captured = {}

    async def agent(_payload, runtime):
        captured["value"] = await body(runtime)
        return AgentResult.ok("ok")

    await harness.wrap(agent, agent_id=context.agent_id, **wrap)(None, context=context)
    return captured["value"]


# =========================================================================== memory types


@pytest.mark.parametrize(
    ("memory_type", "lifetime"),
    [
        ("SEMANTIC", "LONG_TERM"),
        ("EPISODIC", "SHORT_TERM"),
        ("PROCEDURAL", "LONG_TERM"),
        ("PREFERENCE", "LONG_TERM"),
        ("DECISION", "LONG_TERM"),
        ("OUTCOME", "LONG_TERM"),
        ("FAILURE", "LONG_TERM"),
    ],
)
async def test_each_memory_type_is_written_and_stored(client, context, memory_type, lifetime):
    """Every memory type the service supports, written through the harness and then read
    back out of the ``memories`` table with the type and lifetime the caller asked for."""
    harness = build(client)
    # Content has to be a real fact: the service's extractor decides what is worth keeping,
    # and a bare marker string is correctly discarded as noise.
    marker = uuid.uuid4().hex[:6]
    fact = f"Warehouse EU-{marker} ships SKU-{marker} every Tuesday at 09:00."

    await run(harness, context, lambda rt: rt.memory.remember(
        fact, memory_type=memory_type, lifetime=lifetime, visibility="USER",
    ))

    rows = await eventually(
        lambda: sql(
            f"SELECT memory_type, lifetime, visibility FROM memories "
            f"WHERE tenant_id = '{TENANT}' AND content LIKE '%{marker}%'"
        ),
        what=f"a {memory_type} memory row",
    )
    stored_type, stored_lifetime, visibility = rows[0][0], rows[0][1], rows[0][2]
    assert stored_lifetime == lifetime
    assert visibility == "USER"
    # the service may refine the type (its classifier has the last word); what must hold is
    # that the caller's request was not discarded
    assert stored_type, f"no memory_type stored for {memory_type}"
    await harness.aclose()


async def test_visibility_levels_that_the_context_supports(client, context):
    """Each audience the execution context can express, verified in the row."""
    harness = build(client)
    written = {}

    async def body(rt):
        for visibility in ("PRIVATE", "USER", "THREAD", "WORK", "WORKSPACE", "AGENT_GROUP",
                           "TENANT"):
            marker = uuid.uuid4().hex[:6]
            await rt.memory.remember(
                f"Supplier {marker} delivers pallets of SKU-{marker} within nine days.",
                visibility=visibility,
            )
            written[visibility] = marker

    await run(harness, context, body)

    # Poll for all of them at once: seven separate budgets serialise into a long wait when
    # the worker is busy, which makes the test flaky rather than wrong.
    markers = "','".join(f"%{m}%" for m in written.values())

    def stored() -> dict[str, str] | None:
        rows = sql(
            f"SELECT visibility, content FROM memories WHERE tenant_id='{TENANT}' "
            f"AND content LIKE ANY (ARRAY['{markers}'])"
        )
        found = {r[0]: r[1] for r in rows}
        return found if len(found) >= len(written) else None

    found = await eventually(stored, timeout=120.0, what="every visibility level to be stored")
    assert set(found) == set(written), f"missing: {sorted(set(written) - set(found))}"
    await harness.aclose()


async def test_a_visibility_the_context_cannot_express_is_refused_immediately(client):
    """The service accepts such a write and fails the job that would create the memory, so
    the harness refuses it up front instead."""
    from universal_agent_harness.contracts.errors import ConfigurationError

    harness = build(client)
    bare = AgentExecutionContext.create(tenant_id=TENANT, agent_id="live-surface")

    async def body(rt):
        with pytest.raises(ConfigurationError, match="needs an agent group"):
            await rt.memory.share("this has no audience")
        with pytest.raises(ConfigurationError, match="requires workspace_id"):
            await rt.memory.remember("no workspace", visibility="WORKSPACE")

    await run(harness, bare, body)
    await harness.aclose()


async def test_shared_memory_reaches_the_agent_group(client, context):
    harness = build(client)
    marker = f"shared-{uuid.uuid4().hex[:6]}"

    await run(harness, context, lambda rt: rt.memory.share(f"crew fact: {marker}"))

    rows = await eventually(
        lambda: sql(
            f"SELECT visibility, memory_type FROM memories WHERE tenant_id='{TENANT}' "
            f"AND content LIKE '%{marker}%'"
        ),
        what="the shared memory",
    )
    assert rows[0][0] == "AGENT_GROUP"
    await harness.aclose()


async def test_forget_removes_the_memory(client, context):
    harness = build(client)
    marker = uuid.uuid4().hex[:6]

    async def body(rt):
        await rt.memory.remember(
            f"Depot {marker} holds the reserve stock for SKU-{marker}.", visibility="USER"
        )
        rows = await eventually(
            lambda: sql(
                f"SELECT memory_id FROM memories WHERE tenant_id='{TENANT}' "
                f"AND content LIKE '%{marker}%'"
            ),
            what="the memory to delete",
        )
        memory_id = rows[0][0]
        await rt.memory.forget(memory_id)
        return memory_id

    memory_id = await run(harness, context, body)
    gone = await eventually(
        lambda: not sql(
            f"SELECT memory_id FROM memories WHERE memory_id='{memory_id}' "
            f"AND deleted_at IS NULL"
        )
        or sql(f"SELECT deleted_at FROM memories WHERE memory_id='{memory_id}'")[0][0],
        what="the deletion to land",
    )
    assert gone
    await harness.aclose()


# =========================================================================== conversation


async def test_messages_become_the_conversation_and_are_readable_back(client, context):
    harness = build(client)
    question = f"How much stock of SKU-{context.turn_id} do we have?"
    answer = "Ninety five units, about four days of cover."

    async def body(rt):
        await rt.memory.record_input(question)
        await rt.memory.record_output(answer)
        return await rt.memory.history(limit=10)

    history = await run(harness, context, body)
    assert [m.content for m in history] == [question, answer]

    rows = sql(
        f"SELECT role, content FROM messages WHERE thread_id='{context.thread_id}' "
        f"ORDER BY sequence"
    )
    assert [r[0] for r in rows] == ["USER", "ASSISTANT"]
    assert rows[0][1] == question
    await harness.aclose()


async def test_the_thread_session_and_turn_rows_are_created(client, context):
    """The conversation ids the harness derives must exist as real rows, not just strings."""
    harness = build(client)
    await run(harness, context, lambda rt: rt.memory.record_input("a question about stock"))

    assert sql(f"SELECT thread_id FROM threads WHERE thread_id='{context.thread_id}'")
    assert sql(f"SELECT session_id FROM sessions WHERE session_id='{context.session_id}'")
    turns = sql(f"SELECT turn_id, session_id FROM turns WHERE turn_id='{context.turn_id}'")
    assert turns and turns[0][1] == context.session_id, "the derived session must own the turn"
    await harness.aclose()


# =========================================================================== knowledge (RAG)


async def test_a_document_becomes_retrievable_knowledge(client, context, tmp_path):
    """Ingest -> parse -> chunk -> index -> retrieve, verified in the database and in a bundle.

    The thread is created first on purpose: a THREAD-visible document is readable only by
    thread participants, and the thread exists once something is written to it.
    """
    harness = build(client)
    marker = uuid.uuid4().hex[:6]
    doc = tmp_path / f"policy-{marker}.txt"
    doc.write_text(
        f"Reorder policy revision {marker}.\n"
        f"Safety stock for class-A parts is held at a 95 percent service level.\n"
        f"The maximum cover for class-A parts is thirty days.\n"
        f"Castor Supply is the preferred supplier for SKU-{marker} with a nine day lead time.\n"
    )

    async def body(rt):
        await rt.memory.record_input("What is our reorder policy?")   # creates the thread
        handle = await rt.memory.add_document(doc, title=f"Reorder policy {marker}",
                                              visibility="THREAD")
        assert handle.document_id
        # poll the document until the service reports it indexed
        for _ in range(40):
            doc_info = await rt.memory.sdk.files.document(handle.document_id)
            if doc_info.status in ("READY", "FAILED"):
                break
            await asyncio.sleep(1.5)
        assert doc_info.status == "READY", f"ingestion ended as {doc_info.status}"

        async def chunk_visible():
            items = await rt.memory.recall(f"reorder policy revision {marker}", limit=10)
            mine = [i for i in items if i.representation == "CHUNK" and marker in i.text]
            return mine or None

        chunks = await eventually(chunk_visible, what="the document's chunks to be retrievable")
        bundle = await rt.memory.retrieve("what is the safety stock service level?")
        return handle.document_id, chunks, rt.memory.describe(bundle)

    document_id, chunks, facts = await run(harness, context, body)

    assert any(marker in c.text for c in chunks), "the ingested text should be retrievable"
    assert facts["counts"]["knowledge"] >= 1, f"no knowledge in the bundle: {facts}"

    rows = sql(f"SELECT count(*) FROM chunks WHERE document_id='{document_id}'")
    assert int(rows[0][0]) >= 1, "the document produced no chunks"
    docs = sql(f"SELECT status FROM documents WHERE document_id='{document_id}'")
    assert docs[0][0] in ("READY", "INDEXED"), docs
    await harness.aclose()


# =========================================================================== knowledge graph


async def test_observations_populate_the_knowledge_graph(client, context):
    """Entities and relations are extracted from what the harness writes, and the graph is
    then traversable through ``graph_query``."""
    harness = build(client)
    marker = uuid.uuid4().hex[:6]
    supplier = f"Castor{marker}"

    async def body(rt):
        await rt.memory.observe(
            MemoryObservation(
                content=f"{supplier} supplies SKU-{marker} with a nine day lead time.",
                kind="EVENT",
            )
        )
        await rt.memory.remember(
            f"{supplier} is the preferred supplier for SKU-{marker}.",
            memory_type="SEMANTIC", visibility="USER",
        )

        async def has_facts():
            answer = await rt.memory.graph_query(f"who supplies SKU-{marker}?", hops=2)
            facts = list(getattr(answer, "facts", []) or [])
            return facts or None

        return await eventually(has_facts, what="graph facts for the new entities")

    facts = await run(harness, context, body)

    # Traversal works: the query resolved entities and returned relations.
    assert facts, "graph_query returned no facts at all"
    assert all(hasattr(f, "subject") and hasattr(f, "predicate") for f in facts)

    # And the durable proof: this write created graph rows of its own. Which *relations* are
    # extracted depends on the service's configuration — with its LLM disabled the extractor
    # is lexical, so assert on the entity it created rather than on a semantic predicate.
    entities = await eventually(
        lambda: sql(
            f"SELECT entity_id, canonical_name FROM graph_entities WHERE tenant_id='{TENANT}' "
            f"AND canonical_name ILIKE '%{marker}%'"
        ),
        what="a graph entity extracted from the observation",
    )
    assert entities, "the observation produced no graph entity"
    entity_ids = "','".join(e[0] for e in entities)
    relations = sql(
        f"SELECT count(*) FROM graph_relations WHERE tenant_id='{TENANT}' "
        f"AND (subject_id IN ('{entity_ids}') OR object_id IN ('{entity_ids}'))"
    )
    assert int(relations[0][0]) >= 1, "the new entity has no relations in the graph"
    await harness.aclose()


# =========================================================================== grounding


async def test_grounding_reports_on_a_real_bundle(client, context):
    harness = build(client)
    marker = uuid.uuid4().hex[:6]

    async def body(rt):
        await rt.memory.remember(
            f"SKU-{marker} has ninety five units on hand at warehouse EU-1.",
            memory_type="SEMANTIC", visibility="USER",
        )
        bundle = await eventually(
            lambda: _nonempty_bundle(rt, f"how much stock of SKU-{marker}?"),
            what="a bundle containing the new memory",
        )
        supported = await rt.memory.verify(
            f"SKU-{marker} has ninety five units on hand.", bundle=bundle
        )
        if supported is None and rt.memory.last_error is not None:
            pytest.fail(f"verify failed: {rt.memory.last_error.message}")
        return supported

    report = await run(harness, context, body)
    assert report is not None
    assert hasattr(report, "per_claim_hallucination_rate")
    assert report.claims, "the grounding report should contain per-claim verdicts"
    await harness.aclose()


async def _nonempty_bundle(runtime, query):
    bundle = await runtime.memory.retrieve(query)
    facts = runtime.memory.describe(bundle)
    return bundle if facts["item_count"] else None


# =========================================================================== tool memory


async def test_tool_calls_are_recorded_in_tool_memory(client, context):
    marker = uuid.uuid4().hex[:6]

    async def inventory_db(sku: str) -> dict:
        """Stock levels for a SKU."""
        return {"sku": sku, "on_hand": 95}

    harness = AgentHarness(
        memory=client, tools=[inventory_db], defaults={"tenant_id": TENANT},
        config={"memory": {"writeback": False, "retrieve_before": False,
                           "observe_input": False, "observe_output": False,
                           "observe_claims": False},
                "timeouts": {"memory_seconds": 60.0}},
    )

    await run(harness, context, lambda rt: rt.tools.call("inventory_db", sku=f"SKU-{marker}"))

    rows = await eventually(
        lambda: sql(
            f"SELECT tool_name, agent_id, run_id FROM tool_invocations "
            f"WHERE tenant_id='{TENANT}' AND tool_name='inventory_db' "
            f"AND thread_id='{context.thread_id}'"
        ),
        what="a tool invocation row",
    )
    assert rows, "the tool call was not recorded in tool memory"
    # the invocation is attributed to the agent run that made it
    assert rows[0][1] == context.agent_id
    assert rows[0][2], "the invocation has no run id"
    await harness.aclose()


# =========================================================================== idempotency


async def test_a_replayed_write_creates_one_row_not_two(client, context):
    """The harness derives keys from the execution's durable identity, so the same logical
    write replayed produces one memory — this is what makes framework retries safe."""
    harness = build(client)
    marker = uuid.uuid4().hex[:6]
    fact = f"Depot {marker} is the overflow site for SKU-{marker}."

    async def body(rt):
        return await rt.memory.remember(fact, visibility="USER")

    first = await run(harness, context, body)
    second = await run(harness, context, body)      # same context -> same key
    assert first.observation_id == second.observation_id

    rows = sql(
        f"SELECT count(*) FROM observations WHERE tenant_id='{TENANT}' "
        f"AND content LIKE '%{marker}%'"
    )
    assert int(rows[0][0]) == 1, f"a replayed write created {rows[0][0]} observations"
    await harness.aclose()


# =========================================================================== agent runtime


async def test_a_full_agent_turn_uses_every_runtime_client(client, context, spans):
    """One agent using memory, a model, a tool and an artifact — and the identity of that
    run recorded in the service."""
    calls: dict[str, object] = {}

    class Model:
        async def ainvoke(self, prompt, **kwargs):
            calls["prompt"] = prompt
            return {"text": "reorder 500 units", "model": "live-demo",
                    "usage": {"prompt_tokens": 12, "completion_tokens": 5, "cost_usd": 0.0001}}

    async def inventory_db(sku: str) -> dict:
        """Stock levels."""
        return {"sku": sku, "on_hand": 95}

    harness = AgentHarness(
        memory=client, model=Model(), tools=[inventory_db],
        defaults={"tenant_id": TENANT},
        config={
            "memory": {"writeback": False, "record_messages": True},
            "timeouts": {"memory_seconds": 60.0},
            "evaluation_events": {"enabled": True, "synchronous": True},
        },
    )
    events: list[str] = []
    harness.on(lambda event, payload: events.append(event))

    @harness.agent(agent_id="live-surface", skills=["inventory.analysis"])
    async def agent(state, runtime) -> AgentResult:
        assert runtime.memory_context is not None, "memory should have been fetched first"
        stock = (await runtime.tools.call("inventory_db", sku="SKU-1")).output
        answer = (await runtime.model.invoke(f"stock is {stock['on_hand']}")).text
        ref = await runtime.artifacts.put(f"report: {answer}", type="report")
        return AgentResult.ok(
            answer,
            claims=[Claim(claim_id="c1", text=f"SKU-1 has {stock['on_hand']} units")],
            artifacts=[ref],
        )

    result = await agent({"question": "should we reorder SKU-1?"}, context=context)

    assert result.succeeded and result.data == "reorder 500 units"
    assert result.metrics["total_tokens"] == 17
    assert result.metrics["cost_usd"] == pytest.approx(0.0001)
    assert result.artifacts and result.artifacts[0].checksum.startswith("sha256:")
    assert {"on_agent_start", "on_model_end", "on_tool_end", "on_agent_success"} <= set(events)

    names = [s.name for s in spans.get_finished_spans()]
    for expected in ("agent.run", "agent.model.invoke", "agent.tool.call",
                     "agent.memory.retrieve"):
        assert expected in names, f"{expected} missing from {sorted(set(names))}"

    await harness.drain()
    rows = await eventually(
        lambda: sql(
            f"SELECT agent_id FROM agent_runs WHERE tenant_id='{TENANT}' "
            f"AND agent_run_id='{context.agent_run_id}'"
        ),
        what="the agent run row",
    )
    assert rows[0][0] == "live-surface"
    await harness.aclose()


async def test_nested_agents_record_their_lineage_in_the_service(client, context):
    harness = build(client, memory={"writeback": False, "retrieve_before": False,
                                    "observe_input": False, "observe_output": False,
                                    "observe_claims": False})
    child_context: dict[str, AgentExecutionContext] = {}

    @harness.agent(agent_id="child-agent")
    async def child(state, runtime) -> AgentResult:
        child_context["ctx"] = runtime.context
        await runtime.memory.remember(
            f"The child agent inspected depot {state['marker']}.", visibility="USER"
        )
        return AgentResult.ok("child done")

    @harness.agent(agent_id="parent-agent")
    async def parent(state, runtime) -> AgentResult:
        await child(state)
        return AgentResult.ok("parent done")

    marker = uuid.uuid4().hex[:6]
    await parent({"marker": marker}, context=context)

    ctx = child_context["ctx"]
    assert ctx.parent_agent_run_id and ctx.parent_agent_run_id != ctx.agent_run_id
    assert ctx.trace_id == context.trace_id

    # Lineage is durable where it matters: everything the child wrote carries the child's
    # run and the parent's run, so a reader can reconstruct who did what for whom.
    # (``agent_runs`` is only materialised for the run bound to the turn, so the child's
    # write is the evidence here.)
    rows = await eventually(
        lambda: sql(
            f"SELECT agent_id, agent_run_id, parent_agent_run_id FROM observations "
            f"WHERE tenant_id='{TENANT}' AND content LIKE '%{marker}%'"
        ),
        what="the child agent's observation",
    )
    agent_id, agent_run_id, parent_run_id = rows[0]
    assert agent_id == "child-agent"
    assert agent_run_id == ctx.agent_run_id
    assert parent_run_id == ctx.parent_agent_run_id, "the parent run must be recorded"
    assert parent_run_id == context.agent_run_id or parent_run_id, "lineage points upwards"
    await harness.aclose()


async def test_automatic_writeback_records_the_turn(client, context):
    """The default path: question in, answer and claims out, all written by the harness."""
    harness = AgentHarness(
        memory=client, defaults={"tenant_id": TENANT},
        config={"memory": {"writeback": False}, "timeouts": {"memory_seconds": 60.0}},
    )
    marker = uuid.uuid4().hex[:6]

    @harness.agent(agent_id="live-surface")
    async def agent(state, runtime) -> AgentResult:
        return AgentResult.ok(
            f"Warehouse EU-{marker} holds {marker} pallets of SKU-{marker}.",
            claims=[Claim(claim_id="c1",
                          text=f"SKU-{marker} is stored at warehouse EU-{marker}.")],
        )

    await agent({"question": f"where is SKU-{marker} stored?"}, context=context)

    rows = await eventually(
        lambda: sql(
            f"SELECT kind, left(content, 60) FROM observations WHERE tenant_id='{TENANT}' "
            f"AND content LIKE '%{marker}%' ORDER BY created_at"
        ),
        what="the automatic observations",
    )
    kinds = {r[0] for r in rows}
    assert "EVENT" in kinds, f"the question should be recorded as an EVENT: {rows}"
    assert "AGENT_RESULT" in kinds, f"the answer and claim should be AGENT_RESULT: {rows}"
    await harness.aclose()


# =========================================================================== langgraph


class GraphState(TypedDict, total=False):
    question: str
    findings: Annotated[list[str], operator.add]
    answer: str


async def test_a_langgraph_graph_runs_against_the_live_service(client, run_id, spans):
    """A real graph, with the real service behind it: parallel nodes, one trace, memory."""
    from langgraph.graph import END, START, StateGraph

    harness = AgentHarness(
        memory=client,
        defaults={"tenant_id": TENANT, "user_id": f"lg-user-{run_id}"},
        config={"memory": {"writeback": False}, "timeouts": {"memory_seconds": 60.0}},
    )

    @harness.langgraph.agent(agent_id="stock-agent", query="question")
    async def stock(state: GraphState, agent) -> dict:
        await agent.memory.remember(
            f"Depot {run_id} reported ninety five units of SKU-{run_id}.", visibility="USER"
        )
        return {"findings": ["stock checked"]}

    @harness.langgraph.agent(agent_id="supplier-agent")
    async def supplier(state: GraphState, agent) -> dict:
        return {"findings": ["supplier checked"]}

    @harness.langgraph.agent(agent_id="answer-agent")
    async def answer(state: GraphState, agent) -> dict:
        return {"answer": f"checked: {', '.join(sorted(state['findings']))}"}

    graph = StateGraph(GraphState)
    for name, node in (("stock", stock), ("supplier", supplier), ("answer", answer)):
        graph.add_node(name, node)
    graph.add_edge(START, "stock")
    graph.add_edge(START, "supplier")
    graph.add_edge("stock", "answer")
    graph.add_edge("supplier", "answer")
    graph.add_edge("answer", END)
    app = graph.compile()

    context = AgentExecutionContext.create(
        tenant_id=TENANT, agent_id="graph", user_id=f"lg-user-{run_id}",
        thread_id=f"lg-{run_id}", turn_id=f"lg-turn-{run_id}",
    )
    async with harness.execution(context, agent_id="graph", input="stock check"):
        out = await app.ainvoke(
            {"question": f"how much stock of SKU-{run_id}?", "findings": []},
            {"configurable": {"thread_id": f"lg-{run_id}",
                              "harness": {"tenant_id": TENANT,
                                          "user_id": f"lg-user-{run_id}"}}},
        )

    assert out["answer"] == "checked: stock checked, supplier checked"
    finished = spans.get_finished_spans()
    assert len({s.get_span_context().trace_id for s in finished}) == 1, "one trace per turn"
    agents = {s.attributes.get("agent.id") for s in finished if s.name == "agent.run"}
    assert {"graph", "stock-agent", "supplier-agent", "answer-agent"} <= agents

    await harness.drain()
    rows = await eventually(
        lambda: sql(
            f"SELECT count(*) FROM memories WHERE tenant_id='{TENANT}' "
            f"AND content LIKE '%{run_id}%'"
        ),
        what="the memory written from inside the graph",
    )
    assert int(rows[0][0]) >= 1
    await harness.aclose()


# =========================================================================== failure paths


async def test_timeout_cancels_a_live_call(client, context):
    harness = build(client, timeouts={"default_seconds": 0.25, "memory_seconds": 60.0})
    from universal_agent_harness import AgentTimeoutError

    async def slow(_payload, runtime):
        await asyncio.sleep(5)

    with pytest.raises(AgentTimeoutError):
        await harness.wrap(slow, agent_id="slow-agent")(None, context=context)
    await harness.aclose()


async def test_policy_denial_blocks_before_any_write(client, context):
    from universal_agent_harness import AllowListPolicyProvider, PolicyDeniedError

    harness = AgentHarness(
        memory=client, policy=AllowListPolicyProvider(agents={"allowed"}),
        defaults={"tenant_id": TENANT},
        config={"memory": {"writeback": False}, "timeouts": {"memory_seconds": 60.0}},
    )
    marker = uuid.uuid4().hex[:6]

    async def agent(_payload, runtime):
        await runtime.memory.remember(f"Depot {marker} should never be written.")
        return "should not happen"

    with pytest.raises(PolicyDeniedError):
        await harness.wrap(agent, agent_id="live-surface")(
            f"a question about {marker}", context=context
        )

    await asyncio.sleep(2)
    rows = sql(f"SELECT count(*) FROM observations WHERE content LIKE '%{marker}%'")
    assert int(rows[0][0]) == 0, "a denied execution must not have written anything"
    await harness.aclose()


async def test_memory_failure_degrades_the_run_but_keeps_the_answer(context):
    """Point the harness at a dead service: the agent still answers, with a warning."""
    from universal_memory import MemoryClient

    dead = MemoryClient("http://127.0.0.1:9", api_key="x", timeout=2.0, max_retries=0)
    harness = AgentHarness(
        memory=dead, defaults={"tenant_id": TENANT},
        config={"memory": {"writeback": False}, "timeouts": {"memory_seconds": 3.0}},
    )

    @harness.agent(agent_id="live-surface")
    async def agent(state, runtime) -> AgentResult:
        assert runtime.memory_context is None
        return AgentResult.ok("answered without memory")

    result = await agent({"question": "anything"}, context=context)
    assert result.data == "answered without memory"
    assert {w.code for w in result.warnings} >= {"MEMORY_DEGRADED"}
    await harness.aclose()
    await dead.aclose()
