# ruff: noqa: PLC0415 - a research script: it reports with print, imports an SDK where a case needs it
"""LangChain 1.5 ``create_agent`` (a LangGraph graph): its own features, as its docs show them,
on its own (``bare``: ``graph.ainvoke``) and wrapped (``off``, ``stack``, ``on``).

Usage: python lc_agent.py [feature[:arm] ...]   (all of them by default)
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import Any

from common import MODEL, case, chosen, harness, memories, records, uid
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelFallbackMiddleware,
    PIIMiddleware,
    ToolRetryMiddleware,
)
from langchain.tools import ToolRuntime
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt
from pydantic import BaseModel

SDK = "langgraph"


def model(**kw: Any) -> ChatAnthropic:
    return ChatAnthropic(model=MODEL, max_tokens=1024, **kw)  # type: ignore[call-arg]


def read(t: Any) -> Any:
    t.metadata = {"side_effects": "read"}
    return t


def write(t: Any) -> Any:
    t.metadata = {"side_effects": "write"}
    return t


def text(state: dict[str, Any]) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and m.text:
            return m.text
    return ""


def middleware_on(**kw: Any) -> list[Any]:
    from trellis import middleware

    return list(middleware(**kw))


async def drive(
    graph: Any, arm: str, question: str, *, user: str, thread: str | None = None, **opts: Any
) -> dict[str, Any]:
    """One turn: ``graph.ainvoke`` (bare) or ``agent.run``; the answer and how it ended."""
    if arm == "bare":
        cfg = {"configurable": {"thread_id": thread or uid()}}
        state = await graph.ainvoke(
            {"messages": [HumanMessage(question)]}, cfg, **opts.get("call", {})
        )
        return {"answer": state.get("structured_response") or text(state), "state": state}
    raise AssertionError("wrapped turns go through the agent")


# --------------------------------------------------------------------------- features
def stock_tool(calls: list[Any]) -> Any:
    @tool
    def stock(sku: str) -> str:
        """How many units of a SKU are in stock."""
        calls.append(sku)
        return f"{sku}: 7 units in stock"

    return read(stock)


async def tools(arm: str) -> dict[str, Any]:
    calls: list[str] = []
    mw = middleware_on() if arm == "on" else []
    graph = create_agent(model(), tools=[stock_tool(calls)], middleware=mw)
    q = "How many units of SKU-1 are in stock? Use the tool."
    if arm == "bare":
        got = await drive(graph, arm, q, user="u")
        answer = got["answer"]
        seen: dict[str, Any] = {}
    else:
        user = f"lc-{uid()}"
        async with harness(arm) as h:
            agent = h.wrap(graph, id="lc-tools")
            r = await agent.run(q, user=user)
            answer, seen = r.answer, {"status": str(r.status), "run_id": r.run_id}
            if arm == "stack":
                seen["records"] = records(r.run_id)
    ok = "7" in str(answer) and calls == ["SKU-1"]
    return {"ok": ok, "answer": str(answer), "calls": calls, **seen}


class Weather(BaseModel):
    city: str
    celsius: float


async def structured(arm: str) -> dict[str, Any]:
    calls: list[str] = []

    @tool
    def temperature(city: str) -> float:
        """Today's temperature in a city, in Celsius."""
        calls.append(city)
        return 21.5

    mw = middleware_on() if arm == "on" else []
    graph = create_agent(model(), tools=[read(temperature)], response_format=Weather, middleware=mw)
    q = "What is the temperature in Oslo today?"
    if arm == "bare":
        answer = (await drive(graph, arm, q, user="u"))["answer"]
        seen: dict[str, Any] = {}
    else:
        async with harness(arm) as h:
            r = await h.wrap(graph, id="lc-structured").run(q, user=f"lc-{uid()}")
            answer, seen = r.answer, {"status": str(r.status), "run_id": r.run_id}
            if arm == "stack":
                seen["records"] = records(r.run_id)
    w = answer if isinstance(answer, Weather) else None
    ok = w is not None and w.celsius == 21.5 and "oslo" in w.city.lower()
    return {"ok": ok, "answer": repr(answer), "type": type(answer).__name__, "calls": calls, **seen}


async def stream(arm: str) -> dict[str, Any]:
    mw = middleware_on() if arm == "on" else []
    graph = create_agent(model(), tools=[], middleware=mw)
    q = "Write three short sentences about rivers."
    chunks, final = 0, ""
    if arm == "bare":
        async for msg, _meta in graph.astream(
            {"messages": [HumanMessage(q)]}, stream_mode="messages"
        ):
            if getattr(msg, "text", ""):
                chunks += 1
                final += msg.text
        seen: dict[str, Any] = {}
    else:
        async with harness(arm) as h:
            kinds: dict[str, int] = {}
            async for ev in h.wrap(graph, id="lc-stream").stream(q, user=f"lc-{uid()}"):
                kinds[str(ev.type)] = kinds.get(str(ev.type), 0) + 1
                if str(ev.type).endswith("TEXT_MESSAGE_CONTENT"):
                    chunks += 1
                    final += (ev.data or {}).get("delta", "")
            seen = {"events": kinds}
    return {"ok": chunks > 1 and len(final) > 40, "chunks": chunks, "text": final[:300], **seen}


async def hitl(arm: str) -> dict[str, Any]:
    """``HumanInTheLoopMiddleware``: the refund is held, approved, made once."""
    made: list[str] = []

    @tool
    def refund(order: str) -> str:
        """Refund an order."""
        made.append(order)
        return f"refunded {order}"

    mw: list[Any] = [HumanInTheLoopMiddleware(interrupt_on={"refund": True})]
    if arm == "on":
        mw += middleware_on()
    graph = create_agent(
        model(), tools=[write(refund)], middleware=mw, checkpointer=InMemorySaver()
    )
    q = "Refund order O-77 now."
    seen: dict[str, Any] = {}
    if arm == "bare":
        cfg = {"configurable": {"thread_id": uid()}}
        state = await graph.ainvoke({"messages": [HumanMessage(q)]}, cfg)
        paused = bool(state.get("__interrupt__"))
        before = list(made)
        state = await graph.ainvoke(Command(resume={"decisions": [{"type": "approve"}]}), cfg)
        answer = text(state)
    else:
        async with harness(arm) as h:
            agent = h.wrap(graph, id="lc-hitl")
            r = await agent.run(q, user=f"lc-{uid()}", thread=uid())
            paused, before = r.interrupt is not None, list(made)
            seen["interrupt"] = r.interrupt.model_dump(mode="json") if r.interrupt else None
            if r.interrupt:
                r = await agent.resume(r.interrupt.interrupt_id, "approve", reviewer="lee")
            answer, seen["status"], seen["run_id"] = r.answer, str(r.status), r.run_id
            if arm == "stack":
                seen["records"] = records(r.run_id)
    ok = paused and before == [] and made == ["O-77"]
    return {"ok": ok, "paused": paused, "made": made, "answer": str(answer), **seen}


async def checkpointer(arm: str) -> dict[str, Any]:
    """The graph's own checkpointer keeps the thread: turn 2 knows turn 1."""
    mw = middleware_on() if arm == "on" else []
    graph = create_agent(model(), tools=[], middleware=mw, checkpointer=InMemorySaver())
    q1, q2 = "My favourite colour is teal. Just say noted.", "What is my favourite colour?"
    seen: dict[str, Any] = {}
    if arm == "bare":
        cfg = {"configurable": {"thread_id": uid()}}
        await graph.ainvoke({"messages": [HumanMessage(q1)]}, cfg)
        state = await graph.ainvoke({"messages": [HumanMessage(q2)]}, cfg)
        answer, n = text(state), len(state["messages"])
    else:
        async with harness(arm) as h:
            agent = h.wrap(graph, id="lc-checkpointer")
            user, thread = f"lc-{uid()}", uid()
            r1 = await agent.run(q1, user=user, thread=thread)
            r = await agent.run(q2, user=user, thread=thread)
            answer = r.answer
            key = await h.conversation(user=user, thread=thread)
            st = await graph.aget_state({"configurable": {"thread_id": key}})
            n = len(st.values.get("messages", []))
            seen = {"status": str(r.status), "run_ids": [r1.run_id, r.run_id]}
            if arm == "stack":
                seen["records"] = [records(r1.run_id), records(r.run_id)]
    return {
        "ok": "teal" in str(answer).lower(),
        "answer": str(answer),
        "thread_messages": n,
        **seen,
    }


async def store(arm: str) -> dict[str, Any]:
    """The graph's own ``BaseStore``: a fact saved in one thread is read in another."""

    @tool
    def save_fact(fact: str, runtime: ToolRuntime) -> str:
        """Save a fact about the user for later conversations."""
        assert runtime.store is not None
        runtime.store.put(("facts",), "fact", {"text": fact})
        return "saved"

    @tool
    def read_facts(runtime: ToolRuntime) -> str:
        """Read the facts saved about the user."""
        assert runtime.store is not None
        item = runtime.store.get(("facts",), "fact")
        return item.value["text"] if item else "nothing saved"

    own = InMemoryStore()
    mw = middleware_on() if arm == "on" else []
    graph = create_agent(
        model(), tools=[write(save_fact), read(read_facts)], store=own, middleware=mw
    )
    q1 = "Save this fact about me: I am allergic to peanuts."
    q2 = "Read my saved facts and tell me what I am allergic to."
    seen: dict[str, Any] = {}
    if arm == "bare":
        await drive(graph, arm, q1, user="u")
        answer = (await drive(graph, arm, q2, user="u"))["answer"]
    else:
        async with harness(arm) as h:
            agent = h.wrap(graph, id="lc-store")
            user = f"lc-{uid()}"
            r1 = await agent.run(q1, user=user, thread=uid())
            if r1.interrupt:  # a write: governance may ask (said nowhere else)
                seen["asked"] = True
                r1 = await agent.resume(r1.interrupt.interrupt_id, "approve", reviewer="lee")
            r = await agent.run(q2, user=user, thread=uid())
            answer, seen["status"] = r.answer, str(r.status)
            if arm == "stack":
                seen["records"] = [records(r1.run_id), records(r.run_id)]
    kept = own.get(("facts",), "fact")
    ok = "peanut" in str(answer).lower() and kept is not None
    return {"ok": ok, "answer": str(answer), "in_own_store": kept is not None, **seen}


async def own_interrupt(arm: str) -> dict[str, Any]:
    """A tool asking with LangGraph's ``interrupt()``; the answer reaches the tool."""
    got: list[Any] = []

    @tool
    def schedule_delivery(order: str) -> str:
        """Schedule an order's delivery; asks the customer for the address."""
        address = interrupt({"question": f"Which address should {order} go to?"})
        got.append(address)
        return f"{order} will be delivered to {address}"

    mw = middleware_on() if arm == "on" else []
    graph = create_agent(
        model(), tools=[read(schedule_delivery)], middleware=mw, checkpointer=InMemorySaver()
    )
    q = "Schedule delivery of order O-5."
    seen: dict[str, Any] = {}
    if arm == "bare":
        cfg = {"configurable": {"thread_id": uid()}}
        state = await graph.ainvoke({"messages": [HumanMessage(q)]}, cfg)
        asked = str(state["__interrupt__"][0].value) if state.get("__interrupt__") else None
        state = await graph.ainvoke(Command(resume="12 Elm Street"), cfg)
        answer = text(state)
    else:
        async with harness(arm) as h:
            agent = h.wrap(graph, id="lc-interrupt")
            r = await agent.run(q, user=f"lc-{uid()}", thread=uid())
            asked = r.interrupt.question if r.interrupt else None
            if r.interrupt:
                r = await agent.resume(
                    r.interrupt.interrupt_id, "answer", answer="12 Elm Street", reviewer="cust"
                )
            answer, seen["status"] = r.answer, str(r.status)
            if arm == "stack":
                seen["records"] = records(r.run_id)
    ok = got == ["12 Elm Street"] and "elm" in str(answer).lower()
    return {"ok": ok, "asked": asked, "tool_got": got, "answer": str(answer), **seen}


async def tool_retry(arm: str) -> dict[str, Any]:
    """``ToolRetryMiddleware``: a flaky tool fails once, the middleware tries again."""
    tries: list[int] = []

    @tool
    def rate(pair: str) -> str:
        """The exchange rate of a currency pair."""
        tries.append(1)
        if len(tries) == 1:
            raise ConnectionError("upstream timed out")
        return f"{pair} = 1.0842"

    mw: list[Any] = [ToolRetryMiddleware(max_retries=2, initial_delay=0.01)]
    graph = create_agent(model(), tools=[read(rate)], middleware=mw)
    q = "What is the EUR/USD rate? Use the tool."
    seen: dict[str, Any] = {}
    if arm == "bare":
        answer = (await drive(graph, arm, q, user="u"))["answer"]
    else:
        async with harness(arm) as h:
            r = await h.wrap(graph, id="lc-retry").run(q, user=f"lc-{uid()}")
            answer, seen["status"] = r.answer, str(r.status)
            if arm == "stack":
                seen["records"] = records(r.run_id)
    return {
        "ok": "1.0842" in str(answer) and len(tries) == 2,
        "tries": len(tries),
        "answer": str(answer),
        **seen,
    }


async def fallback(arm: str) -> dict[str, Any]:
    """``ModelFallbackMiddleware``: the primary model does not exist; Haiku answers."""
    primary = ChatAnthropic(model="claude-nonexistent-0", max_tokens=256, max_retries=0)  # type: ignore[call-arg]
    graph = create_agent(primary, tools=[], middleware=[ModelFallbackMiddleware(model())])
    q = "Reply with the single word: pong"
    seen: dict[str, Any] = {}
    if arm == "bare":
        answer = (await drive(graph, arm, q, user="u"))["answer"]
    else:
        async with harness(arm) as h:
            r = await h.wrap(graph, id="lc-fallback").run(q, user=f"lc-{uid()}")
            answer, seen["status"] = r.answer, str(r.status)
            seen["error"] = str(r.error) if r.error else None
            if arm == "stack":
                seen["records"] = records(r.run_id)
    return {"ok": "pong" in str(answer).lower(), "answer": str(answer), **seen}


async def pii(arm: str) -> dict[str, Any]:
    """``PIIMiddleware``: the e-mail in the input is redacted before the model sees it."""
    mw: list[Any] = [PIIMiddleware("email", strategy="redact", apply_to_input=True)]
    graph = create_agent(model(), tools=[], middleware=mw)
    q = "My e-mail is jane.doe@example.com. Repeat back exactly what my e-mail is."
    seen: dict[str, Any] = {}
    user = f"lc-{uid()}"
    if arm == "bare":
        answer = (await drive(graph, arm, q, user="u"))["answer"]
    else:
        async with harness(arm) as h:
            r = await h.wrap(graph, id="lc-pii").run(q, user=user, thread=uid())
            answer, seen["status"] = r.answer, str(r.status)
            if arm == "stack":
                rec = records(r.run_id)
                seen["records"] = rec
                kept = str(rec["messages"]) + str(memories(user))
                seen["email_in_memory"] = "jane.doe@example.com" in kept
    leaked = "jane.doe@example.com" in str(answer) or seen.get("email_in_memory", False)
    return {"ok": not leaked, "answer": str(answer), **seen}


async def custom_stream(arm: str) -> dict[str, Any]:
    """``get_stream_writer()``: a tool's progress on the ``custom`` stream mode."""

    @tool
    def crunch(n: int) -> str:
        """Crunch n batches of numbers."""
        w = get_stream_writer()
        for i in range(n):
            w({"progress": i + 1, "of": n})
        return f"crunched {n} batches"

    graph = create_agent(model(), tools=[read(crunch)])
    q = "Crunch 3 batches."
    customs: list[Any] = []
    seen: dict[str, Any] = {}
    if arm == "bare":
        async for mode, chunk in graph.astream(
            {"messages": [HumanMessage(q)]}, stream_mode=["custom", "updates"]
        ):
            if mode == "custom":
                customs.append(chunk)
    else:
        async with harness(arm) as h:
            async for ev in h.wrap(graph, id="lc-custom").stream(q, user=f"lc-{uid()}"):
                if str(ev.type).endswith("CUSTOM"):
                    customs.append(ev.data)
            seen["note"] = "harness CUSTOM events"
    vals = [c.get("value", c) if isinstance(c, dict) else c for c in customs]
    prog = [v.get("progress") for v in vals if isinstance(v, dict) and "progress" in v]
    return {"ok": prog == [1, 2, 3], "progress": prog, "customs": customs[:6], **seen}


async def context_schema(arm: str) -> dict[str, Any]:
    """``context_schema``: the run's own context read by a tool through ``ToolRuntime``."""

    @tool
    def balance(runtime: ToolRuntime[Ctx]) -> str:
        """The balance of the caller's account."""
        return f"account {runtime.context.account}: 420 EUR"

    graph = create_agent(model(), tools=[read(balance)], context_schema=Ctx)
    q = "What is my account balance?"
    seen: dict[str, Any] = {}
    if arm == "bare":
        got = await drive(graph, arm, q, user="u", call={"context": Ctx(account="AC-9")})
        answer = got["answer"]
    else:
        async with harness(arm) as h:
            agent = h.wrap(graph, id="lc-context")
            r = await agent.run(
                q, user=f"lc-{uid()}", framework_options={"context": Ctx(account="AC-9")}
            )
            answer, seen["status"] = r.answer, str(r.status)
            seen["error"] = str(r.error) if r.error else None
    return {"ok": "420" in str(answer), "answer": str(answer), **seen}


@dataclass
class Ctx:
    account: str


FEATURES: dict[str, Any] = {
    "tools": {"bare", "off", "stack", "on"},
    "structured": {"bare", "off", "stack", "on"},
    "stream": {"bare", "off", "on"},
    "hitl": {"bare", "off", "stack", "on"},
    "checkpointer": {"bare", "off", "stack", "on"},
    "store": {"bare", "off", "stack"},
    "own_interrupt": {"bare", "off", "stack", "on"},
    "tool_retry": {"bare", "off", "stack"},
    "fallback": {"bare", "off"},
    "pii": {"bare", "off", "stack"},
    "custom_stream": {"bare", "off"},
    "context_schema": {"bare", "off"},
}


async def main() -> None:
    for feature, arm in chosen(sys.argv[1:], FEATURES):
        await case(SDK, feature, arm, lambda f=feature, a=arm: globals()[f](a))


if __name__ == "__main__":
    asyncio.run(main())
