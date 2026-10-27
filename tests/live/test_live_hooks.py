"""Hooks against the running gateway: a ``before_tool`` hook denies one MCP call (it never
reaches the gateway) and rewrites another (the gateway runs it with the hook's arguments); a
``ReAct``'s own middleware sees every model call it makes through the gateway to the live model,
and the request it hands on is what the model is sent. The local model's answers are not
what is checked: the harness's behaviour is."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from tests.live.conftest import MODEL, WIKI_TOOL, live_harness, needs_gateway
from trellis import Deny, Hooks, ReAct, Rewrite, Runtime
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


class Steering(AgentMiddleware):
    """Every model call: the question replaced (what the model is sent), the reply kept."""

    def __init__(self) -> None:
        super().__init__()
        self.replies: list[tuple[Any, Any]] = []

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        messages = [
            m.model_copy(update={"content": PROMPT}) if isinstance(m, HumanMessage) else m
            for m in request.messages
        ]
        reply = await handler(request.override(messages=messages))
        self.replies.append((messages, reply))
        return reply


@pytest.mark.timeout(180)
async def test_react_model_calls_through_the_gateway_go_through_its_middleware() -> None:
    steering = Steering()
    async with live_harness() as h:
        target = ReAct(
            system_prompt="You reply in one word.", model=MODEL, max_steps=2, middleware=[steering]
        )
        agent = h.wrap(target, id=f"live-steer-{uuid.uuid4().hex[:8]}")
        result = await agent.run("Tell me a long story about dragons.", user="live-user")
    assert result.status is RunStatus.SUCCESS, result.error
    assert steering.replies
    for sent, reply in steering.replies:  # what the model was sent, and what came back
        assert all(m.content == PROMPT for m in sent if isinstance(m, HumanMessage))
        assert isinstance(reply.result[-1], AIMessage)
    assert isinstance(result.answer, str) and result.answer
