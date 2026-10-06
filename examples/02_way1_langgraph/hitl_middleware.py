"""LangChain's own approvals: ``HumanInTheLoopMiddleware`` (Deep Agents: ``interrupt_on``)
pauses before ``send_email``; the harness records that pause as an approval in the run store
and answers it with the middleware's own decisions — here an edit (the email goes out with the
reviewer's text), then a reject with a reason the model reads.

The gated tool is ``side_effects="write"``: the middleware asks, so the harness does not ask
again.

    python -m examples.02_way1_langgraph.hitl_middleware
"""

from __future__ import annotations

import asyncio

from examples._support.offline import langchain_model
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from trellis import Harness, tool

outbox: list[str] = []


@tool(side_effects="write")
def send_email(to: str, body: str) -> str:
    """Send an email."""
    outbox.append(f"{to}: {body}")
    return f"sent to {to}"


async def main() -> None:
    async with Harness() as h:
        draft = ("send_email", {"to": "supplier@acme.test", "body": "Send 20 units."})
        model = langchain_model([draft, "Sent.", draft, "Not sent: the order is on hold."])
        graph = create_agent(
            model,
            tools=await h.tools(send_email, framework="langgraph"),
            middleware=[HumanInTheLoopMiddleware(interrupt_on={"send_email": True})],
            checkpointer=InMemorySaver(),  # a middleware pause resumes from the checkpoint
        )
        agent = h.wrap(graph, id="mailer")

        first = await agent.run("Ask ACME for 20 units.", user="ada", thread="mail-1")
        assert first.interrupt is not None
        print("asks:", first.interrupt.question)  # "Approve send_email?"
        edited = {"to": "supplier@acme.test", "body": "Send 20 units by Friday."}
        done = await agent.resume(
            first.interrupt.interrupt_id, "edit", answer=edited, reviewer="lead"
        )
        print(done.status.value, done.answer, outbox)

        second = await agent.run("Ask ACME again.", user="ada", thread="mail-2")
        assert second.interrupt is not None
        held = await agent.resume(
            second.interrupt.interrupt_id, "reject", answer="the order is on hold", reviewer="lead"
        )
        print(held.status.value, held.answer, outbox)


if __name__ == "__main__":
    asyncio.run(main())
