"""The harness's last call offered no tool (a stall, the step limit, a spent budget) on an
Anthropic chat model: LangChain's ``"none"`` is OpenAI's word, which ``ChatAnthropic`` reads as
the name of a tool to force — Anthropic's API answers 400 ("Tool 'none' not found") and the run
fails with no answer. It is sent as Anthropic's own ``{"type": "none"}``."""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from tests.support.arm import arm
from trellis import Harness, ReAct, tool
from trellis.contracts import RunStatus


@tool(side_effects="read")
def look(item: str) -> str:
    """Look an item up."""
    return f"{item} found"


def reply(content: list[dict[str, Any]], stop: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-x",
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }


@pytest.fixture
def anthropic(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Anthropic's Messages API in process: the same ``look`` call three times, then an answer;
    a request forcing a tool the request does not define is refused, as Anthropic's API does."""
    from langchain_anthropic import _client_utils, chat_models

    sent: list[dict[str, Any]] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        sent.append(body)
        choice = body.get("tool_choice") or {}
        names = {t["name"] for t in body.get("tools") or []}
        if choice.get("type") == "tool" and choice.get("name") not in names:
            error = {"type": "invalid_request_error", "message": "Tool 'none' not found"}
            return httpx2.Response(400, json={"type": "error", "error": error})
        if len(sent) <= 3:
            call = {"type": "tool_use", "id": f"toolu_{len(sent)}", "name": "look"}
            return httpx2.Response(200, json=reply([{**call, "input": {"item": "a"}}], "tool_use"))
        return httpx2.Response(200, json=reply([{"type": "text", "text": "a found"}], "end_turn"))

    def client(**kwargs: Any) -> Any:
        kwargs.pop("anthropic_proxy", None)
        return _client_utils._AsyncHttpxClientWrapper(
            transport=httpx2.MockTransport(answer), **kwargs
        )

    monkeypatch.setattr(chat_models, "_get_default_async_httpx_client", client)
    return sent


@pytest.mark.parametrize("given", ["name", "object"])
async def test_a_stall_on_an_anthropic_model_ends_with_its_answer(
    harness: Harness,
    anthropic: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    given: str,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    model = (
        "anthropic:claude-x"  # a model name the harness routes to ChatAnthropic
        if given == "name"
        else ChatAnthropic(model_name="claude-x", api_key="k", max_retries=0)  # type: ignore[call-arg,arg-type]
    )
    target = ReAct("You look items up.", model, context_window=50_000, max_repeats=3)
    agent = arm(harness.wrap(target, id=f"stall-anthropic-{given}"), [look])
    result = await agent.run("a?", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    assert result.answer == "a found"
    assert anthropic[-1]["tool_choice"] == {"type": "none"}
    assert anthropic[-1]["tools"]  # the history's calls keep their tools defined


class Found(BaseModel):
    item: str
    found: bool


@pytest.fixture
def stubborn(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Anthropic's Messages API in process, for a run whose answer is a schema: while ``look``
    is offered, the model calls it — whatever ``tool_choice`` says, as LangChain sends ``any``
    with a structured answer's tool (the live model, Haiku, did this) — and offered the
    schema's tool alone, it answers with it."""
    from langchain_anthropic import _client_utils, chat_models

    sent: list[dict[str, Any]] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        sent.append(body)
        names = {t["name"] for t in body.get("tools") or []}
        if "look" in names:
            call = {"type": "tool_use", "id": f"toolu_{len(sent)}", "name": "look"}
            return httpx2.Response(200, json=reply([{**call, "input": {"item": "a"}}], "tool_use"))
        call = {"type": "tool_use", "id": f"toolu_{len(sent)}", "name": "Found"}
        given = {"item": "a", "found": True}
        return httpx2.Response(200, json=reply([{**call, "input": given}], "tool_use"))

    def client(**kwargs: Any) -> Any:
        kwargs.pop("anthropic_proxy", None)
        return _client_utils._AsyncHttpxClientWrapper(
            transport=httpx2.MockTransport(answer), **kwargs
        )

    monkeypatch.setattr(chat_models, "_get_default_async_httpx_client", client)
    return sent


async def test_a_stall_of_a_run_with_a_schema_ends_with_its_answer_in_the_schema(
    harness: Harness, stubborn: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last call offers the schema's tool alone: LangChain forces a tool for a structured
    answer (``tool_choice: any``), so offered every tool the model went on calling them, and
    the run stopped with no answer though the model would have given one."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    target = ReAct(
        "You look items up.",
        "anthropic:claude-x",
        response_format=Found,
        context_window=50_000,
        max_repeats=3,
    )
    agent = arm(harness.wrap(target, id="stall-anthropic-schema"), [look])
    result = await agent.run("a?", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    assert result.answer == Found(item="a", found=True)
    assert [t["name"] for t in stubborn[-1]["tools"]] == ["Found"]
    assert [t["name"] for t in stubborn[0]["tools"]].count("look") == 1  # offered before
