"""Hooks against the running gateway: a ``before_tool`` hook denies one MCP call (it never
reaches the gateway) and rewrites another (the gateway runs it with the hook's arguments); the
model hooks see every model call a ``ReAct`` agent makes through the gateway to the live model,
and what ``before_model`` returns is what the model is sent. The local model's answers are not
what is checked: the harness's behaviour is."""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from tests.live.conftest import MODEL, WIKI_TOOL, live_harness, needs_gateway
from trellis import Deny, Hooks, ModelCall, ReAct, Rewrite, Runtime
from trellis.contracts import RunStatus, ToolCall

pytestmark = [pytest.mark.live, needs_gateway]

#: What the model is asked instead of the run's question.
PROMPT = "Reply with the single word: pong"


class Guard(Hooks):
    """No wiki of a secret repository; ``react`` means facebook/react."""

    def __init__(self) -> None:
        self.decided: list[str] = []

    async def before_tool(self, call: ToolCall) -> Deny | Rewrite | None:
        repo = call.args.get("repoName")
        if repo == "secret/repo":
            self.decided.append("deny")
            return Deny("secret repositories are not read")
        if repo == "react":
            self.decided.append("rewrite")
            return Rewrite({**call.args, "repoName": "facebook/react"})
        return None


@pytest.mark.timeout(120)
async def test_a_hook_denies_and_rewrites_mcp_calls_through_the_gateway(
    deepwiki: str, wiki_key: str
) -> None:
    name = f"{deepwiki}-{WIKI_TOOL}"
    guard = Guard()

    async def reads(input: str, agent: Runtime) -> list[Any]:
        denied = await agent.tools.call(name, repoName="secret/repo")
        rewritten = await agent.tools.call(name, repoName="react")
        return [denied, rewritten]

    async with live_harness(wiki_key) as h:
        agent = h.wrap(reads, id=f"live-guard-{uuid.uuid4().hex[:8]}", hooks=[guard])
        result = await agent.run("read the wikis", user="live-user")
    assert result.status is RunStatus.SUCCESS, result.error
    denied, rewritten = result.answer
    assert denied == f"{name} was not run: secret repositories are not read"
    assert "React Overview" in str(rewritten)  # the gateway ran it on facebook/react
    assert guard.decided == ["deny", "rewrite"]


class Steering(Hooks):
    """Every model call: the question replaced (what the model is sent), the reply kept."""

    def __init__(self) -> None:
        self.calls: list[ModelCall] = []
        self.replies: list[Any] = []

    async def before_model(self, call: ModelCall) -> ModelCall:
        self.calls.append(call)
        messages = [
            m.model_copy(update={"content": PROMPT}) if isinstance(m, HumanMessage) else m
            for m in call.messages
        ]
        return dataclasses.replace(call, messages=messages)

    async def after_model(self, call: ModelCall, reply: Any) -> None:
        self.replies.append((call, reply))


@pytest.mark.timeout(180)
async def test_react_model_calls_through_the_gateway_go_through_the_hooks() -> None:
    steering = Steering()
    async with live_harness() as h:
        target = ReAct(system="You reply in one word.", model=MODEL, max_steps=2)
        agent = h.wrap(target, id=f"live-steer-{uuid.uuid4().hex[:8]}", hooks=[steering])
        result = await agent.run("Tell me a long story about dragons.", user="live-user")
    assert result.status is RunStatus.SUCCESS, result.error
    assert steering.calls and len(steering.replies) == len(steering.calls)
    assert all(c.framework == "langgraph" and c.model == MODEL for c in steering.calls)
    for sent, reply in steering.replies:  # what the model was sent, and what came back
        assert all(m.content == PROMPT for m in sent.messages if isinstance(m, HumanMessage))
        assert isinstance(reply.result[-1], AIMessage)
    assert isinstance(result.answer, str) and result.answer
