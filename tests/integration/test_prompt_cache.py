"""Anthropic's prompt cache for every Anthropic chat model ``ReAct`` is given — a model name the
harness routes as much as a ``ChatAnthropic`` — on LangChain's loop and on Deep Agents': the
request marks the tools, the instructions and the answer that ends the history (what the next
turn repeats), with LangChain's ``AnthropicPromptCachingMiddleware``. Any other provider's
request is sent as it was."""

from __future__ import annotations

import json
from typing import Any, Final

import httpx
import httpx2
import pytest
import respx
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

from trellis import Harness, ReAct, Settings, tool
from trellis.contracts import RunStatus
from trellis.harness.automatic import JournalBackend

COMPLETIONS: Final = "https://api.openai.com/v1/chat/completions"
MARK: Final = {"type": "ephemeral", "ttl": "5m"}
ANSWERS: Final = ("7 units.", "You asked about a.")


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock."""
    return {"a": 7}.get(sku, 0)


def anthropic_reply(text: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-x",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }


def openai_reply(text: str) -> dict[str, Any]:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-x",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
    }


def graph(model: Any, deep: bool) -> Any:
    return ReAct(
        "You answer stock questions.",
        model,
        tools=[stock],
        context_window=50_000,
        backend=JournalBackend() if deep else None,
    )


async def two_turns(harness: Harness, target: Any, name: str) -> None:
    agent = harness.wrap(target, id=name)
    for text in ("How many a?", "Which sku did I ask about?"):
        result = await agent.run(text, user="u1", thread="conv-1")
        assert result.status is RunStatus.SUCCESS, result.error


def breakpoints(body: dict[str, Any]) -> int:
    """The cache breakpoints a Messages API request sets (Anthropic takes at most 4)."""
    return json.dumps(body).count('"cache_control"')


@pytest.fixture
def anthropic(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Anthropic's Messages API in process: each request's body, answered in turn by
    :data:`ANSWERS` (a ``ChatAnthropic``'s client is the SDK's httpx fork, which ``respx`` does
    not reach: it is given a mock transport of that library)."""
    from langchain_anthropic import _client_utils, chat_models

    sent: list[dict[str, Any]] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "test-key"
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=anthropic_reply(ANSWERS[len(sent) - 1]))

    def client(**kwargs: Any) -> Any:
        kwargs.pop("anthropic_proxy", None)
        return _client_utils._AsyncHttpxClientWrapper(
            transport=httpx2.MockTransport(answer), **kwargs
        )

    monkeypatch.setattr(chat_models, "_get_default_async_httpx_client", client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return sent


@pytest.mark.parametrize("deep", [False, True], ids=["langchain", "deepagents"])
@pytest.mark.parametrize("given", ["name", "object"])
async def test_an_anthropic_model_however_given_marks_what_the_next_call_repeats(
    memory_harness: Harness, anthropic: list[dict[str, Any]], deep: bool, given: str
) -> None:
    model = (
        "anthropic:claude-x"
        if given == "name"
        else ChatAnthropic(model_name="claude-x", api_key="test-key")  # type: ignore[call-arg,arg-type]
    )
    await two_turns(memory_harness, graph(model, deep), f"cached-{given}-{deep}")
    first, later = anthropic
    for body in (first, later):
        assert body["cache_control"] == MARK  # the request's own breakpoint (its last block)
        assert body["system"][-1]["cache_control"] == MARK  # the instructions
        assert body["tools"][-1]["cache_control"] == MARK  # the tools
        assert breakpoints(body) <= 4
    assert "cache_control" not in json.dumps(first["messages"])  # no history yet
    *history, question = later["messages"]
    assert question["role"] == "user" and history[-1]["role"] == "assistant"
    assert history[-1]["content"][-1] == {"type": "text", "text": ANSWERS[0], "cache_control": MARK}
    assert breakpoints(later) == 4


@pytest.mark.parametrize("deep", [False, True], ids=["langchain", "deepagents"])
async def test_your_own_caching_middleware_is_the_one_used(
    memory_harness: Harness, anthropic: list[dict[str, Any]], deep: bool
) -> None:
    """``AnthropicPromptCachingMiddleware`` given to ``ReAct`` is used, its settings kept, for
    a model name too — as Deep Agents uses one given in place of its own."""
    target = ReAct(
        "s",
        "anthropic:claude-x",
        context_window=50_000,
        middleware=[AnthropicPromptCachingMiddleware(ttl="1h")],
        backend=JournalBackend() if deep else None,
    )
    await two_turns(memory_harness, target, f"own-{deep}")
    body = anthropic[-1]
    assert body["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert body["messages"][-2]["content"][-1]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


@pytest.mark.parametrize("deep", [False, True], ids=["langchain", "deepagents"])
@respx.mock
async def test_another_providers_request_is_sent_as_it_was(
    memory_harness: Harness, monkeypatch: pytest.MonkeyPatch, deep: bool
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    route = respx.post(COMPLETIONS).mock(
        side_effect=[httpx.Response(200, json=openai_reply(a)) for a in ANSWERS]
    )
    await two_turns(memory_harness, graph("openai:gpt-x", deep), f"openai-{deep}")
    assert len(route.calls) == 2
    for call in route.calls:
        assert "cache_control" not in call.request.content.decode()


@respx.mock
async def test_anthropics_own_endpoint_given_as_openai_compatible_is_cached(
    anthropic: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``LLM_BASE_URL`` at Anthropic's OpenAI compatibility layer, which drops prompt caching:
    its model names are called with Anthropic's own client (the endpoint's key), and cached."""
    monkeypatch.delenv("ANTHROPIC_API_KEY")  # the key is the endpoint's (LLM_API_KEY)
    compatible = respx.post("https://api.anthropic.com/v1/chat/completions").mock(
        side_effect=[httpx.Response(200, json=openai_reply(a)) for a in ANSWERS]
    )
    config = Settings(llm_base_url="https://api.anthropic.com/v1/", llm_api_key="test-key")
    async with Harness(config=config) as h:
        agent = h.wrap(graph("claude-x", deep=False), id="endpoint")
        for text in ("How many a?", "Which sku did I ask about?"):
            result = await agent.run(text, user="u1", thread="conv-1")
            assert result.status is RunStatus.SUCCESS, result.error
    assert not compatible.called
    assert len(anthropic) == 2
    assert all(body["cache_control"] == MARK for body in anthropic)
    assert all(body["tools"][-1]["cache_control"] == MARK for body in anthropic)
