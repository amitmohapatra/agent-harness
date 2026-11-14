"""A model name in LangChain's ``ModelFallbackMiddleware`` and ``ModelRouterMiddleware``: given
as the harness's routed chat model (``gateways.routed_chat_model``, what ``ReAct`` makes of
its own model name), the fallback and the picked model go through the gateway as the main
model does — the route's form of the name, its key, the gateway's headers for the run."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.model_router import ModelChoice, ModelRouterMiddleware
from langchain_core.decisions import BaseDecisionModel, ClassifierResponse

from trellis import Harness, ReAct, Settings
from trellis.harness.gateways import routed_chat_model

GATEWAY = "http://gw.test/v1"


def answer(model: str, text: str) -> httpx.Response:
    reply = {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
    }
    return httpx.Response(200, json=reply)


def by_model(request: httpx.Request) -> httpx.Response:
    """The gateway: the main model is down, every other answers with its own name."""
    model = json.loads(request.content)["model"]
    if model == "provider/main":
        return httpx.Response(500, json={"error": {"message": "provider down"}})
    return answer(model, f"from {model}")


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("LLM_BASE_URL", GATEWAY)
    monkeypatch.setenv("LLM_API_KEY", "vk")
    monkeypatch.setenv("LLM_GATEWAY", "bifrost")
    with respx.mock(assert_all_called=False) as mocked:
        mocked.post("http://gw.test/mcp").mock(
            return_value=httpx.Response(
                200, json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
            )
        )
        yield mocked.post(f"{GATEWAY}/chat/completions").mock(side_effect=by_model)


async def run(graph: Any) -> Any:
    settings = Settings(llm_gateway="bifrost", llm_base_url=GATEWAY, llm_api_key="vk")
    async with Harness(config=settings) as h:
        return await h.wrap(graph, id="named", without={"schedules"}).run("x", user="u")


def gateway_headers(request: httpx.Request) -> dict[str, str]:
    return {k: v for k, v in request.headers.items() if k == "authorization" or "x-bf-" in k}


async def test_a_routed_fallback_goes_through_the_gateway_as_the_main_model(gateway: Any) -> None:
    fallback = ModelFallbackMiddleware(routed_chat_model("provider/backup"))
    graph = ReAct("s", "provider/main", context_window=50_000, middleware=[fallback])
    result = await run(graph)
    assert result.answer == "from provider/backup"
    first, second = (call.request for call in gateway.calls)
    assert json.loads(first.content)["model"] == "provider/main"
    assert json.loads(second.content)["model"] == "provider/backup"
    assert gateway_headers(second) == gateway_headers(first)
    assert gateway_headers(second)["authorization"] == "Bearer vk"
    assert "x-bf-mcp-include-tools" in gateway_headers(second)


class Picks(BaseDecisionModel):
    """A decision model that always picks ``route``."""

    route: str

    @property
    def _provider(self) -> str:
        return "scripted"

    def _classify(self, state: Any, questions: Any) -> ClassifierResponse:
        picked = {"type": "choice", "choice": self.route, "probabilities": {}, "confidence": 1.0}
        return ClassifierResponse.model_validate(
            {"answers": dict.fromkeys(questions, picked), "model": self.model}
        )


async def test_a_routed_models_pick_goes_through_the_gateway(gateway: Any) -> None:
    router = ModelRouterMiddleware(
        choices={
            "small": ModelChoice(model=routed_chat_model("provider/small"), criteria="short"),
            "large": ModelChoice(model=routed_chat_model("provider/large"), criteria="long"),
        },
        instructions="Pick by the task's length.",
        classifier=Picks(model="scripted", route="small"),
    )
    graph = ReAct("s", "provider/main", context_window=50_000, middleware=[router])
    result = await run(graph)
    assert result.answer == "from provider/small"
    [request] = (call.request for call in gateway.calls)
    assert gateway_headers(request)["authorization"] == "Bearer vk"
    assert "x-bf-mcp-include-tools" in gateway_headers(request)
