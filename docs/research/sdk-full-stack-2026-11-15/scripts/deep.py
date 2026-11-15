"""Deep Agents 0.7.38 (``create_deep_agent``): its own features as its docs show them, on its
own (``bare``) and wrapped (``off``, ``stack``, ``on`` = ``middleware=trellis.middleware()``).

Usage: python deep.py [feature[:arm] ...]
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Any

from common import case, chosen, uid
from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend, StoreBackend
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from lc_agent import middleware_on, model, read, text, write
from pydantic import BaseModel
from turns import wrapped

SDK = "deepagents"


def mw(arm: str) -> list[Any]:
    return middleware_on() if arm == "on" else []


async def bare_turns(graph: Any, questions: list[str], *, resume: Any = None) -> dict[str, Any]:
    cfg = {"configurable": {"thread_id": uid()}}
    answers, asked = [], []
    state: dict[str, Any] = {}
    for q in questions:
        state = await graph.ainvoke({"messages": [HumanMessage(q)]}, cfg)
        while state.get("__interrupt__") and resume is not None:
            asked.append(str(state["__interrupt__"][0].value)[:300])
            state = await graph.ainvoke(Command(resume=resume), cfg)
        answers.append(state.get("structured_response") or text(state))
    snap = await graph.aget_state(cfg) if graph.checkpointer else None
    values = snap.values if snap else state
    return {"answers": answers, "asked": asked, "values": values}


async def run(graph: Any, arm: str, name: str, qs: list[str], **kw: Any) -> dict[str, Any]:
    if arm == "bare":
        kw.pop("state_of", None)
        got = await bare_turns(graph, qs, resume=kw.pop("resume", None))
        values = got.pop("values")
        got["todos"] = values.get("todos")
        got["files"] = sorted((values.get("files") or {}).keys())
        return got
    kw.pop("resume", None)
    return await wrapped(graph, arm, name, qs, **kw)


# --------------------------------------------------------------------------- features
async def todos(arm: str) -> dict[str, Any]:
    """``write_todos``: the agent plans a multi-step task in its own todo list. Deep Agents
    0.7.38 no longer adds ``TodoListMiddleware`` itself (only its Codex profile does): its docs'
    way is ``middleware=[TodoListMiddleware()]``. With ``middleware()`` on, that one would be a
    duplicate (LangChain's ``AssertionError``), so the ``on`` arm has the harness's alone."""
    own = mw(arm) if arm == "on" else [TodoListMiddleware()]
    graph = create_deep_agent(model(), tools=[], middleware=own, checkpointer=InMemorySaver())
    q = (
        "Use write_todos to plan these three steps, then do them: 1) name a fruit, "
        "2) name a vegetable, 3) name a grain. Mark each todo completed as you go."
    )
    got = await run(graph, arm, "da-todos", [q], state_of=graph)
    todo = got.get("todos") or []
    done = [t for t in todo if t.get("status") == "completed"]
    return {"ok": len(todo) >= 3 and len(done) == len(todo), **got}


async def files(arm: str) -> dict[str, Any]:
    """The state backend's files: written in turn 1, read back in turn 2 of the thread."""
    graph = create_deep_agent(model(), tools=[], middleware=mw(arm), checkpointer=InMemorySaver())
    qs = [
        "Write the text 'launch code: ORCHID-42' to the file /notes.md. Then say done.",
        "Read /notes.md and tell me the launch code in it.",
    ]
    got = await run(graph, arm, "da-files", qs, state_of=graph)
    return {"ok": "ORCHID-42" in str(got["answers"][-1]), **got}


async def subagent(arm: str) -> dict[str, Any]:
    """A sub-agent of its own (``task``) with its own tool."""
    calls: list[str] = []

    @tool
    def lookup_order(order: str) -> str:
        """Look up an order's status."""
        calls.append(order)
        return f"order {order}: shipped on 2026-11-02 via DHL"

    sub = {
        "name": "order-checker",
        "description": "Looks up orders' status.",
        "system_prompt": "Look the order up with lookup_order and report its status.",
        "tools": [read(lookup_order)],
        "model": model(),
    }
    graph = create_deep_agent(
        model(), tools=[], subagents=[sub], middleware=mw(arm), checkpointer=InMemorySaver()
    )
    q = "Delegate to the order-checker sub-agent: what is the status of order O-31?"
    got = await run(graph, arm, "da-sub", [q])
    ok = calls == ["O-31"] and "dhl" in str(got["answers"][-1]).lower()
    return {"ok": ok, "sub_calls": calls, **got}


async def interrupt_on(arm: str) -> dict[str, Any]:
    """``interrupt_on``: Deep Agents' own approval of ``send_email``; approved, sent once."""
    sent: list[str] = []

    @tool
    def send_email(to: str, body: str) -> str:
        """Send an e-mail."""
        sent.append(to)
        return f"sent to {to}"

    graph = create_deep_agent(
        model(),
        tools=[write(send_email)],
        interrupt_on={"send_email": True},
        middleware=mw(arm),
        checkpointer=InMemorySaver(),
    )
    q = "Send an e-mail to bob@example.com saying 'the build is green'."
    got = await run(graph, arm, "da-hitl", [q], resume={"decisions": [{"type": "approve"}]})
    return {"ok": sent == ["bob@example.com"] and bool(got["asked"]), "sent": sent, **got}


async def store_backend(arm: str) -> dict[str, Any]:
    """A ``CompositeBackend`` with a ``/memories/`` route on its own ``StoreBackend``: a file
    written in one thread is read in another."""
    own = InMemoryStore()

    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(namespace=lambda _ctx: ("mem",), store=own)},
    )

    graph = create_deep_agent(
        model(),
        tools=[],
        backend=backend,
        store=own,
        middleware=mw(arm),
        checkpointer=InMemorySaver(),
    )
    q1 = "Save the text 'favourite city: Lisbon' to /memories/prefs.md. Then say done."
    q2 = "Read /memories/prefs.md and tell me my favourite city."
    if arm == "bare":
        a = await bare_turns(graph, [q1])
        b = await bare_turns(graph, [q2])
        got: dict[str, Any] = {"answers": a["answers"] + b["answers"]}
    else:
        got = await wrapped(graph, arm, "da-store", [q1, q2], thread=None)
    kept = own.search(("mem",))
    return {
        "ok": "lisbon" in str(got["answers"][-1]).lower() and bool(kept),
        "in_own_store": [i.key for i in kept],
        **got,
    }


def disk(tag: str) -> Path:
    root = Path(tempfile.mkdtemp(prefix=f"da-{tag}-"))
    (root / "AGENTS.md").write_text("# Project notes\nThe deploy password hint is: blue-heron.\n")
    skill = root / "skills" / "haiku-poet"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: haiku-poet\ndescription: Write a haiku about any topic the user names.\n---\n"
        "# Haiku poet\nWrite exactly three lines (5-7-5 syllables), then on a fourth line "
        "write the signature `-- SKILL:HAIKU`.\n"
    )
    return root


async def memory_files(arm: str) -> dict[str, Any]:
    """``memory=["/AGENTS.md"]``: Deep Agents loads its memory file into the prompt."""
    root = disk("mem")
    graph = create_deep_agent(
        model(),
        tools=[],
        memory=["/AGENTS.md"],
        middleware=mw(arm),
        backend=FilesystemBackend(root_dir=root, virtual_mode=True),
    )
    got = await run(graph, arm, "da-memory", ["What is the deploy password hint in your notes?"])
    return {"ok": "blue-heron" in str(got["answers"][-1]).lower(), **got}


async def skills(arm: str) -> dict[str, Any]:
    """``skills=["/skills/"]``: Deep Agents' own ``SkillsMiddleware`` finds and follows it."""
    root = disk("skill")
    graph = create_deep_agent(
        model(),
        tools=[],
        skills=["/skills/"],
        middleware=mw(arm),
        backend=FilesystemBackend(root_dir=root, virtual_mode=True),
    )
    got = await run(graph, arm, "da-skills", ["Write a haiku about autumn rain."])
    return {"ok": "SKILL:HAIKU" in str(got["answers"][-1]), **got}


async def permissions(arm: str) -> dict[str, Any]:
    """``permissions``: a deny rule on ``/secrets/**`` refuses the read."""
    root = disk("perm")
    (root / "secrets").mkdir()
    (root / "secrets" / "key.txt").write_text("TOPSECRET-991")
    graph = create_deep_agent(
        model(),
        tools=[],
        middleware=mw(arm),
        backend=FilesystemBackend(root_dir=root, virtual_mode=True),
        permissions=[FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")],
    )
    got = await run(graph, arm, "da-perm", ["Read /secrets/key.txt and tell me what it says."])
    return {"ok": "TOPSECRET-991" not in str(got["answers"][-1]), **got}


class Verdict(BaseModel):
    sentiment: str
    score: int


async def structured(arm: str) -> dict[str, Any]:
    graph = create_deep_agent(model(), tools=[], response_format=Verdict, middleware=mw(arm))
    q = "Classify the sentiment of 'I love this product' as positive/negative and score 1-10."
    got = await run(graph, arm, "da-structured", [q])
    a = got["answers"][-1]
    return {
        "ok": isinstance(a, Verdict) and a.sentiment.lower() == "positive",
        "type": type(a).__name__,
        **got,
    }


FEATURES: dict[str, Any] = {
    "todos": {"bare", "off", "stack", "on"},
    "files": {"bare", "off", "stack", "on"},
    "subagent": {"bare", "off", "stack", "on"},
    "interrupt_on": {"bare", "off", "stack", "on"},
    "store_backend": {"bare", "off", "stack"},
    "memory_files": {"bare", "off", "stack"},
    "skills": {"bare", "off", "on"},
    "permissions": {"bare", "off"},
    "structured": {"bare", "off", "on"},
}


async def main() -> None:
    for feature, arm in chosen(sys.argv[1:], FEATURES):
        await case(SDK, feature, arm, lambda f=feature, a=arm: globals()[f](a))


if __name__ == "__main__":
    asyncio.run(main())
