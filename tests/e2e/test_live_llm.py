"""The product on a real model: multi-turn recall, and multi-agent cooperation.

    make test-live-llm    # or: pytest -m live_llm -q

Separate from the rest of the live suite because it needs a *model* as well as a service,
and a model is the one dependency that is slow, metered and occasionally unavailable —
Gemini answers 503 "high demand" often enough that a test which cannot retry is a test that
fails for reasons that have nothing to do with this code.

What is being proven here cannot be proven with a fake model:

* **Turn 2 has no process state.** It is a separate execution with a fresh context, so an
  answer can only come out of the Memory Service.
* **Sub-agents never touch each other.** They cooperate through AGENT_GROUP memory, which
  is what lets them be retried, recorded and (later) run somewhere else entirely.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from universal_agent_harness import AgentExecutionContext, AgentHarness

MEMORY_URL = os.environ.get("MEMORY_SERVICE_URL", "http://localhost:8080")
BIFROST_URL = os.environ.get("BIFROST_URL", "http://localhost:8091/v1")
MODEL = os.environ.get("LLM_MODEL", "gemini/gemini-3.6-flash")
TENANT = os.environ.get("MEMORY_TENANT", "acme")
API_KEY = os.environ.get("MEMORY_API_KEY", "dev-key")


def _reachable(url: str, path: str = "/health/live") -> bool:
    """Ask the service. An env-var gate turns "it is down" and "I forgot to export it" into
    the same green run."""
    import httpx

    try:
        return httpx.get(f"{url}{path}", timeout=5).status_code < 500
    except Exception:
        return False


LIVE = _reachable(MEMORY_URL) and _reachable(BIFROST_URL.rsplit("/v1", 1)[0], "/v1/models")

pytestmark = [
    pytest.mark.live_llm,
    pytest.mark.skipif(
        not LIVE,
        reason=f"needs the Memory Service at {MEMORY_URL} and a gateway at {BIFROST_URL}",
    ),
]


@pytest.fixture
async def stack():
    """A harness on the real gateway and the real service.

    Everything is per-test: the user id included. Reusing one user made every previous run's
    facts compete with this run's, and the model answered with a *previous* run's codename —
    a green-looking bug in the test, not the service.
    """
    from universal_memory import MemoryClient

    from universal_agent_harness.models.bifrost import BifrostModelClient

    run = uuid.uuid4().hex[:8]
    model = BifrostModelClient(base_url=BIFROST_URL, model=MODEL, timeout=120.0)
    memory = MemoryClient(MEMORY_URL, api_key=API_KEY, timeout=60.0)
    harness = AgentHarness(
        model=model,
        memory=memory,
        defaults={"tenant_id": TENANT, "user_id": f"e2e-{run}"},
        config={"memory": {"writeback": True}},
    )
    try:
        yield harness, memory, run, f"e2e-{run}"
    finally:
        await harness.aclose()
        await model.aclose()
        await memory.aclose()


async def _until(predicate, *, within: float = 60.0, every: float = 2.0) -> bool:
    """Writes are asynchronous; polling is the honest way to wait for one."""
    deadline = asyncio.get_running_loop().time() + within
    while asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(every)
        if await predicate():
            return True
    return False


def _rate_limited_now() -> str | None:
    """Whether the provider is refusing traffic *right now*, and what it said.

    Checked before the test rather than caught during it, and on purpose. The SDK honours
    the delay Gemini puts in the response body, so a rate-limited call no longer fails
    fast — it waits the ~30s it was asked to, and then trips the harness's own model
    deadline. That surfaces as ``TimeoutError``, which is indistinguishable from a genuine
    hang once you are only reading exception text.

    So the quota is established up front, from a cheap call, and the test either runs
    against a model that will answer or says plainly why it did not run. Catching the
    failure afterwards would have meant either skipping on every timeout — hiding real
    ones — or reporting the free tier's "limit: 20" as a defect in this code.
    """
    import httpx

    try:
        response = httpx.post(
            f"{BIFROST_URL}/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            timeout=30,
        )
    except Exception as exc:  # the gateway itself is the dependency; say which one failed
        return f"gateway unreachable: {exc}"
    if response.status_code == 429 or '"code":"429"' in response.text:
        return response.text[-160:]
    return None


@pytest.fixture(autouse=True)
def _needs_quota():
    if (why := _rate_limited_now()) is not None:
        pytest.skip(f"model provider is rate limited: {why}")


async def _model_work(coro):
    """Kept as the single seam for model-touching calls in these tests."""
    return await coro


async def test_a_second_turn_answers_from_memory_alone(stack) -> None:
    harness, memory, run, user = stack
    thread = f"chat-{run}"
    codename = f"Orion-{run.upper()}"

    async def assistant(payload, runtime):
        bundle = await runtime.memory.retrieve(str(payload), token_budget=1500)
        reply = await runtime.model.invoke(
            "Answer in one short sentence, using the CONTEXT if it is relevant.\n"
            f"CONTEXT:\n{str(bundle)[:2000]}\n\nUSER: {payload}"
        )
        answer = reply.text.strip()
        await runtime.memory.remember(
            f"User said: {payload}. Assistant replied: {answer}",
            memory_type="EPISODIC",
            lifetime="LONG_TERM",
            visibility="USER",
        )
        return answer

    chat = harness.wrap(assistant, agent_id="assistant")

    def context(turn: str) -> AgentExecutionContext:
        return AgentExecutionContext.create(
            tenant_id=TENANT,
            agent_id="assistant",
            user_id=user,
            thread_id=thread,
            turn_id=f"{turn}-{run}",
        )

    first = await _model_work(
        chat(f"My project codename is {codename}. Acknowledge briefly.", context=context("t1"))
    )
    assert first.status == "SUCCESS", first.error

    bound = memory.bind(tenant_id=TENANT, user_id=user)

    async def indexed() -> bool:
        found = await bound.recall(codename, limit=5)
        return any(
            codename.lower() in str(i).lower() for i in (getattr(found, "items", found) or [])
        )

    assert await _until(indexed), "turn 1's memory was never indexed"

    # A separate execution: no shared state with the first, so this can only be memory.
    second = await _model_work(chat("What is my project codename?", context=context("t2")))
    assert second.status == "SUCCESS", second.error
    assert codename.lower() in str(second.data).lower(), str(second.data)


async def test_sub_agents_cooperate_through_shared_memory(stack) -> None:
    """The handoff is the point: nothing is passed between them in Python."""
    harness, memory, run, user = stack
    crew, thread = f"crew-{run}", f"ma-{run}"

    async def researcher(payload, runtime):
        reply = await runtime.model.invoke(
            f"In one short sentence, state a fact about: {payload}. No preamble."
        )
        finding = reply.text.strip()
        await runtime.memory.remember(
            finding, memory_type="SHARED", visibility="AGENT_GROUP", lifetime="LONG_TERM"
        )
        return finding

    async def coordinator(payload, runtime):
        bundle = await runtime.memory.retrieve(str(payload), token_budget=2000)
        reply = await runtime.model.invoke(
            "Summarise the crew's findings in one sentence, using only the CONTEXT.\n"
            f"CONTEXT:\n{str(bundle)[:2500]}\n\nTASK: {payload}"
        )
        return reply.text.strip()

    research = harness.wrap(researcher, agent_id="researcher")
    lead = harness.wrap(coordinator, agent_id="coordinator")

    parent = AgentExecutionContext.create(
        tenant_id=TENANT,
        agent_id="coordinator",
        user_id=user,
        thread_id=thread,
        turn_id=f"t-{run}",
        agent_group_id=crew,
    )
    topics = [
        "wind turbine gearbox maintenance intervals",
        "wind turbine gearbox failure causes",
    ]
    children = [parent.for_agent("researcher", agent_group_id=crew) for _ in topics]
    assert all(c.parent_agent_run_id == parent.agent_run_id for c in children)
    assert len({c.agent_run_id for c in children}) == len(children), "sub-runs must be distinct"

    results = await _model_work(
        asyncio.gather(
            *(research(topic, context=child) for topic, child in zip(topics, children, strict=True))
        )
    )
    assert all(r.status == "SUCCESS" for r in results), [r.error for r in results]

    bound = memory.bind(tenant_id=TENANT, user_id=user, agent_id="coordinator", agent_group_id=crew)

    async def shared_visible() -> bool:
        found = await bound.recall("wind turbine gearbox", limit=10)
        return len(getattr(found, "items", found) or []) >= len(topics)

    assert await _until(shared_visible), "the crew's findings never reached the coordinator"

    summary = await _model_work(lead("brief on wind turbine gearboxes", context=parent))
    assert summary.status == "SUCCESS", summary.error
    assert "gearbox" in str(summary.data).lower()
