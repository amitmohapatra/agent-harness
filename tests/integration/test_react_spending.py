"""What a ``ReAct`` run spends and how it recovers: model retries, fallbacks and the prompt
cache are the gateway's (the client tries once), a fixed effort is passed through, an answer
that does not fit ``output`` is LangChain's to ask for again, and the run's budget — given,
never by default — counts the dollars the gateway says."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, cast

import httpx
import pytest
import respx
from bifrost_sdk import NO_GATEWAY_TOOLS
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, Field

from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from trellis import Budget, Harness, ReAct, Runtime, Settings, current, tool
from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness import budget, react
from trellis.harness.prompts import GatewayHeaders

GATEWAY = "http://gw.test/v1"


@pytest.fixture(autouse=True)
def _gateway(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("BIFROST_URL", GATEWAY)
    monkeypatch.setenv("BIFROST_VIRTUAL_KEY", "vk")
    for cached in (react._unknown, budget._no_cost):
        cached.cache_clear()
    yield


runs: list[Runtime] = []


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock.

    Args:
        sku: The SKU, such as a-1.
    """
    runtime = current()
    if runtime is not None:
        runs.append(runtime)
    return 7


def harness() -> Harness:
    return Harness(config=Settings(bifrost_url=GATEWAY, bifrost_virtual_key="vk"), gateway=False)


def completion(content: str = "ok", *, model: str = "granite/g", **usage: int) -> dict[str, Any]:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, **usage},
    }


def tool_call(name: str, args: dict[str, Any], *, model: str = "granite/g") -> dict[str, Any]:
    body = completion(model=model)
    body["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }
    body["choices"][0]["finish_reason"] = "tool_calls"
    return body


class Spending(ScriptedChatModel):
    """A scripted model whose replies say their usage and model; while ``fails`` has any,
    a call fails with the next."""

    model_name: str = "provider/scripted"
    fails: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(
        default_factory=lambda: {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
    )

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
        if self.fails:
            self.seen.append(list(messages))
            self.requests.append(kw)
            raise RuntimeError(self.fails.pop(0))
        result: ChatResult = super()._generate(messages, stop, run_manager, **kw)
        message = cast(AIMessage, result.generations[0].message)
        message.usage_metadata = cast(Any, dict(self.usage))
        message.response_metadata = {"model_name": self.model_name}
        return ChatResult(generations=[ChatGeneration(message=message)])


def body_of(route: respx.Route, n: int = 0) -> dict[str, Any]:
    return json.loads(route.calls[n].request.content)


# --------------------------------------------------------------------------- the gateway owns
@respx.mock
async def test_the_client_tries_once_the_gateway_retries_and_falls_back() -> None:
    chat = respx.post(f"{GATEWAY}/chat/completions").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json=completion("unused"))]
    )
    async with harness() as h:
        result = await h.wrap(ReAct("s", "granite/g"), id="r").run("hi?", user="u")
    assert result.status is RunStatus.ERROR and chat.call_count == 1
    assert "fallbacks" not in body_of(chat)  # the virtual key's, not the code's


async def test_without_fallbacks_a_failed_call_is_the_runs_error_as_before() -> None:
    model = Spending(turns=[], fails=["down"])
    async with Harness(config=Settings()) as h:
        result = await h.wrap(ReAct("s", model), id="own").run("hi?", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "down" in result.error.message


@respx.mock
async def test_effort_is_passed_through_as_it_is() -> None:
    chat = respx.post(f"{GATEWAY}/chat/completions").mock(
        return_value=httpx.Response(200, json=completion())
    )
    async with harness() as h:
        await h.wrap(ReAct("s", "granite/g", effort="low"), id="e").run("hi?", user="u")
        await h.wrap(ReAct("s", "granite/g"), id="n").run("hi?", user="u")
    assert body_of(chat, 0)["reasoning_effort"] == "low"
    assert "reasoning_effort" not in body_of(chat, 1)
    with pytest.raises(ConfigurationError, match="effort is for a model name"):
        ReAct("s", Spending(turns=[]), effort="low")


# --------------------------------------------------------------------------- structured output
class Count(BaseModel):
    sku: str
    units: int


SCHEMA = {
    "title": "Count",
    "type": "object",
    "properties": {"sku": {"type": "string"}, "units": {"type": "integer"}},
    "required": ["sku", "units"],
}


@pytest.mark.parametrize("output", [Count, SCHEMA], ids=["pydantic", "json-schema"])
async def test_a_valid_structured_answer_is_the_answer(output: Any) -> None:
    model = Spending(turns=[("Count", {"sku": "a-1", "units": 7})])
    async with Harness(config=Settings()) as h:
        result = await h.wrap(ReAct("s", model, response_format=output), id="so").run(
            "a-1?", user="u"
        )
    answer = result.answer if isinstance(result.answer, dict) else result.answer.model_dump()
    assert answer == {"sku": "a-1", "units": 7}


async def test_an_answer_that_does_not_fit_is_asked_for_again_by_langchain() -> None:
    model = Spending(
        turns=[("Count", {"sku": "a-1", "units": "many"}), ("Count", {"sku": "a-1", "units": 7})]
    )
    async with Harness(config=Settings()) as h:
        result = await h.wrap(ReAct("s", model, response_format=Count), id="so").run(
            "a-1?", user="u"
        )
    assert result.answer == Count(sku="a-1", units=7)
    told = model.seen[1][-1].text  # LangChain's tool strategy says what was wrong
    assert "Failed to parse structured output" in told and "units" in told


# --------------------------------------------------------------------------- budget
async def test_at_its_token_limit_a_run_answers_with_what_it_has_and_is_partial() -> None:
    runs.clear()
    model = Spending(turns=[("stock", {"sku": "a-1"}), ("stock", {"sku": "a-2"}), "a-1: 7"])
    async with Harness(config=Settings()) as h:
        agent = arm(h.wrap(ReAct("s", model), id="b", budget=Budget(max_tokens=150)), [stock])
        result = await agent.run("a-1 and a-2?", user="u")
    assert result.status is RunStatus.PARTIAL and result.answer == "a-1: 7"
    assert result.error is not None and result.error.code == budget.BUDGET_EXCEEDED
    assert "max_tokens 150 reached (220 spent)" in result.error.message
    assert len(runs) == 1  # a-2's call, asked past the limit, was not run
    refused = model.seen[-1][-2]
    assert refused.text.startswith("stock was not run: " + budget.SPENT)
    assert model.seen[-1][-1].text == budget.SPENT
    assert model.requests[-1].get("tool_choice") == "none"


async def test_exactly_at_its_limit_a_run_starts_nothing_more() -> None:
    runs.clear()
    model = Spending(turns=[("stock", {"sku": "a-1"}), "nothing counted"])
    async with Harness(config=Settings()) as h:
        agent = arm(h.wrap(ReAct("s", model), id="b", budget=Budget(max_tokens=110)), [stock])
        result = await agent.run("a-1?", user="u")
    assert result.status is RunStatus.PARTIAL and runs == []


async def test_below_its_limit_a_run_is_not_stopped() -> None:
    model = Spending(turns=[("stock", {"sku": "a-1"}), "a-1: 7"])
    async with Harness(config=Settings()) as h:
        agent = arm(h.wrap(ReAct("s", model), id="b", budget=Budget(max_tokens=111)), [stock])
        result = await agent.run("a-1?", user="u")
    assert result.status is RunStatus.SUCCESS


async def test_without_a_budget_there_is_no_limit_but_spending_is_counted() -> None:
    runs.clear()
    model = Spending(turns=[("stock", {"sku": "a-1"}), "a-1: 7"])
    async with Harness(config=Settings()) as h:
        agent = arm(h.wrap(ReAct("s", model), id="b"), [stock])  # no budget: no cap
        result = await agent.run("a-1?", user="u")
    assert result.status is RunStatus.SUCCESS
    assert runs[0].replay.journal.spent["tokens"] == 220  # both replies, counted


async def test_a_reply_the_gateway_put_no_cost_on_is_said_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = Spending(turns=["a", "b"], model_name="local/costless")
    async with Harness(config=Settings()) as h:
        agent = h.wrap(ReAct("s", model), id="b", budget=Budget(max_cost=1.0))
        with caplog.at_level("WARNING", logger="trellis.run"):
            await agent.run("x", user="u")
            await agent.run("x", user="u")
    said = [r for r in caplog.records if "no cost on local/costless" in r.message]
    assert len(said) == 1


@respx.mock
async def test_the_gateways_cost_is_what_the_run_spent() -> None:
    runs.clear()
    said = {"cost": {"total_cost": 0.25}}
    respx.post(f"{GATEWAY}/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=tool_call("stock", {"sku": "a-1"})),
            httpx.Response(200, json=completion("7", **said)),  # type: ignore[arg-type]
        ]
    )
    async with harness() as h:
        agent = arm(h.wrap(ReAct("s", "granite/g"), id="cost"), [stock])
        assert (await agent.run("a-1?", user="u")).answer == "7"
    spent = runs[0].replay.journal.spent
    assert spent["cost"] == 0.25 and spent["model_calls"] == 2


def test_a_reply_without_the_gateways_cost_says_none() -> None:
    assert budget.gateway_cost({}) is None
    assert budget.gateway_cost({"token_usage": {"cost": {"total_cost": True}}}) is None
    assert budget.gateway_cost({"token_usage": {"cost": {"total_cost": 1}}}) == 1.0


async def test_a_structured_answer_at_the_budget_is_asked_for_without_other_tools() -> None:
    model = Spending(turns=[("stock", {"sku": "a-1"}), ("Count", {"sku": "a-1", "units": 7})])
    async with Harness(config=Settings()) as h:
        agent = arm(
            h.wrap(ReAct("s", model, response_format=Count), id="b", budget=Budget(max_tokens=1)),
            [stock],
        )
        result = await agent.run("a-1?", user="u")
    assert result.status is RunStatus.PARTIAL and result.answer == Count(sku="a-1", units=7)
    assert model.requests[-1]["tools"] == ["Count"]  # only the answer's own tool


# --------------------------------------------------------------------------- attribution
@respx.mock
async def test_every_gateway_call_says_whose_run_and_joins_its_trace(
    spans: InMemorySpanExporter,
) -> None:
    import hashlib

    chat = respx.post(f"{GATEWAY}/chat/completions").mock(
        return_value=httpx.Response(200, json=completion())
    )
    async with harness() as h:
        result = await h.wrap(ReAct("s", "granite/g"), id="dims").run("hi?", user="ada")
        given = GatewayHeaders()
        assert dict(given) == dict(NO_GATEWAY_TOOLS) and len(given) == len(NO_GATEWAY_TOOLS)
        assert repr(given).startswith("GatewayHeaders(")  # read per request: nothing outside a run
    headers = chat.calls[0].request.headers
    assert headers["x-bf-dim-agent"] == "dims" and headers["x-bf-dim-run"] == result.run_id
    assert headers["x-bf-dim-tenant"] == "default"
    assert headers["x-bf-dim-user"] == hashlib.sha256(b"ada").hexdigest()[:16]  # never "ada"
    [model_span] = [s for s in spans.get_finished_spans() if s.name.startswith("chat ")]
    assert model_span.context is not None
    trace_id = format(model_span.context.trace_id, "032x")
    assert headers["traceparent"].split("-")[1] == trace_id  # the gateway's span joins it
