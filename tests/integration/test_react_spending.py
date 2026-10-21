"""What a ``ReAct`` run spends and how it recovers: fallback models (the gateway's own for a
model it serves, LangChain's ``ModelFallbackMiddleware`` only for a model of the user's own),
the model client's retries not doubling the gateway's, the prompt cache on an Anthropic model,
an answer that does not fit ``output`` asked for once more, and the run's budget."""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
import respx
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, Field

from tests.integration.test_react_native import Anthropic
from tests.support.chat_model import ScriptedChatModel
from trellis import Budget, Harness, ReAct, Runtime, Settings, current, tool
from trellis.contracts import ModelError, RunStatus
from trellis.harness import automatic, budget, models, react, telemetry

GATEWAY = "http://gw.test/v1"


@pytest.fixture(autouse=True)
def _gateway(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("BIFROST_URL", GATEWAY)
    monkeypatch.setenv("BIFROST_VIRTUAL_KEY", "vk")
    for cached in (models._listed, models.serves, models._assumed, budget._unpriced):
        cached.cache_clear()
    yield


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("t"))
    yield exporter


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


def walked_error(status: int = 503) -> httpx.Response:
    """A gateway error after it tried its own fallbacks (Bifrost's ``routing_info``)."""
    body = {
        "is_bifrost_error": False,
        "status_code": status,
        "error": {"message": "every model failed"},
        "extra_fields": {"routing_info": {"is_fallback": True, "model": "gpt-6-sol"}},
    }
    return httpx.Response(status, json=body)


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


# --------------------------------------------------------------------------- fallback
@respx.mock
async def test_a_gateway_models_fallbacks_are_the_gateways_own_and_never_two_layers() -> None:
    chat = respx.post(f"{GATEWAY}/chat/completions").mock(return_value=httpx.Response(503))
    graph = ReAct("s", "granite/g", fallback_models=["openai/gpt-6-sol", "granite/other"])
    async with harness() as h:
        result = await h.wrap(graph, id="fb").run("hi?", user="u")
    assert result.status is RunStatus.ERROR
    # the fallbacks went with the request, for the gateway to try; it tried them and failed,
    # so the client did not try them all again, and nothing in the graph fell back either
    assert body_of(chat)["fallbacks"] == ["openai/gpt-6-sol", "granite/other"]
    assert chat.call_count == 1


def test_react_adds_no_fallback_middleware_for_a_gateway_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[Any] = []
    monkeypatch.setattr(automatic.Fallback, "__init__", lambda self, *m: built.append(m))
    ReAct("s", "granite/g", fallback_models=["openai/gpt-6-sol"])
    assert built == []
    ReAct("s", "granite/g", fallback_models=[Spending(turns=[])])  # a model of the user's own
    assert len(built) == 1


@respx.mock
async def test_without_fallback_none_are_sent() -> None:
    chat = respx.post(f"{GATEWAY}/chat/completions").mock(
        return_value=httpx.Response(200, json=completion())
    )
    graph = ReAct("s", "granite/g", fallback_models=["openai/gpt-6-sol"])
    async with harness() as h:
        await h.wrap(graph, id="fb").run("hi?", user="u", without={"fallback"})
    assert "fallbacks" not in body_of(chat)


@respx.mock
async def test_the_model_that_answered_is_recorded(spans: InMemorySpanExporter) -> None:
    runs.clear()
    respx.post(f"{GATEWAY}/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=tool_call("stock", {"sku": "a-1"}, model="gpt-6-sol")),
            httpx.Response(200, json=completion("7", model="gpt-6-sol")),
        ]
    )
    graph = ReAct("s", "granite/g", fallback_models=["openai/gpt-6-sol"])
    async with harness() as h:
        assert (await h.wrap(graph, id="fb", tools=[stock]).run("a-1?", user="u")).answer == "7"
    [runtime] = runs
    assert runtime.replay.journal.answered_by == {"gpt-6-sol": 2}
    chats = [s for s in spans.get_finished_spans() if s.name.startswith("chat ")]
    assert {dict(s.attributes or {})["gen_ai.response.model"] for s in chats} == {"gpt-6-sol"}


async def test_a_model_of_the_users_own_falls_back_in_the_graph() -> None:
    first = Spending(turns=[], fails=["down"])
    second = Spending(turns=["from the fallback"], model_name="provider/second")
    async with Harness(config=Settings()) as h:
        agent = h.wrap(ReAct("s", first, fallback_models=[second]), id="own")
        assert (await agent.run("hi?", user="u")).answer == "from the fallback"
        first.fails = ["down"]
        failed = await agent.run("hi?", user="u", without={"fallback"})
    assert failed.status is RunStatus.ERROR and second.turns == []


async def test_when_every_fallback_fails_the_last_error_is_the_runs() -> None:
    first = Spending(turns=[], fails=["first down"])
    second = Spending(turns=[], fails=["second down"])
    async with Harness(config=Settings()) as h:
        result = await h.wrap(ReAct("s", first, fallback_models=[second]), id="own").run(
            "hi?", user="u"
        )
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "second down" in result.error.message


async def test_without_fallbacks_a_failed_call_is_the_runs_error_as_before() -> None:
    model = Spending(turns=[], fails=["down"])
    async with Harness(config=Settings()) as h:
        result = await h.wrap(ReAct("s", model), id="own").run("hi?", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "down" in result.error.message


@respx.mock
async def test_a_user_models_named_fallback_is_a_gateway_model_in_the_graph() -> None:
    chat = respx.post(f"{GATEWAY}/chat/completions").mock(
        return_value=httpx.Response(200, json=completion("from the gateway"))
    )
    first = Spending(turns=[], fails=["down"])
    async with harness() as h:
        agent = h.wrap(ReAct("s", first, fallback_models=["granite/g"]), id="own")
        assert (await agent.run("hi?", user="u")).answer == "from the gateway"
    assert "fallbacks" not in body_of(chat)  # its own call: nothing for the gateway to try


async def test_a_native_clients_fallback_across_vendors_goes_through_the_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = Anthropic(errors=[529])
    monkeypatch.setattr(react, "_anthropic_http", endpoint.client)
    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://gw.test/anthropic/v1/messages").mock(return_value=httpx.Response(400))
        generic = mock.post(f"{GATEWAY}/chat/completions")
        graph = ReAct("s", "anthropic/claude-sonnet-5-5", fallback_models=["openai/gpt-6-sol"])
        async with harness() as h:
            result = await h.wrap(graph, id="native").run("hi?", user="u")
        assert not generic.called
    assert result.status is RunStatus.ERROR
    [sent] = endpoint.requests  # the gateway tried the fallback; the client not again
    assert sent["fallbacks"] == ["openai/gpt-6-sol"] and sent["model"] == "claude-sonnet-5-5"


# --------------------------------------------------------------------------- model retries
@respx.mock
async def test_the_client_retries_an_error_the_gateway_did_not_fall_back_from() -> None:
    chat = respx.post(f"{GATEWAY}/chat/completions").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json=completion("after a retry"))]
    )
    async with harness() as h:
        result = await h.wrap(ReAct("s", "granite/g"), id="r").run("hi?", user="u")
    assert result.answer == "after a retry" and chat.call_count == 2


@respx.mock
async def test_the_client_does_not_retry_after_the_gateways_own_fallbacks() -> None:
    chat = respx.post(f"{GATEWAY}/chat/completions").mock(return_value=walked_error())
    async with harness() as h:
        result = await h.wrap(ReAct("s", "granite/g"), id="r").run("hi?", user="u")
    assert result.status is RunStatus.ERROR and chat.call_count == 1


async def test_a_gateway_answer_the_hook_cannot_read_is_left_as_it_is() -> None:
    request = httpx.Request("POST", GATEWAY, content=b"not json")
    response = httpx.Response(503, request=request, content=b"not json either")
    await react._gateway_answered(response)
    assert "x-should-retry" not in response.headers
    fine = httpx.Response(200, request=request)
    await react._gateway_answered(fine)
    assert "x-should-retry" not in fine.headers


# --------------------------------------------------------------------------- prompt cache
async def test_an_anthropic_models_calls_are_cached_and_their_cached_tokens_counted(
    monkeypatch: pytest.MonkeyPatch, spans: InMemorySpanExporter
) -> None:
    endpoint = Anthropic(usage={"cache_read_input_tokens": 40, "cache_creation_input_tokens": 5})
    monkeypatch.setattr(react, "_anthropic_http", endpoint.client)
    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://gw.test/anthropic/v1/messages").mock(return_value=httpx.Response(400))
        async with harness() as h:
            agent = h.wrap(ReAct("s", "anthropic/claude-sonnet-5-5"), id="cache")
            result = await agent.run("hi?", user="u")
            await agent.run("hi?", user="u", without={"prompt_cache"})
    assert result.answer == "hi"
    cached, uncached = endpoint.requests
    assert cached.get("cache_control") == {"type": "ephemeral", "ttl": "5m"}
    assert "cache_control" not in uncached
    [chat, _] = [s for s in spans.get_finished_spans() if s.name.startswith("chat ")]
    attributes = dict(chat.attributes or {})
    assert attributes["gen_ai.usage.cache_read.input_tokens"] == 40
    assert attributes["gen_ai.usage.cache_creation.input_tokens"] == 5


@respx.mock
async def test_the_prompt_cache_leaves_any_other_model_as_it_is() -> None:
    chat = respx.post(f"{GATEWAY}/chat/completions").mock(
        return_value=httpx.Response(200, json=completion())
    )
    async with harness() as h:
        await h.wrap(ReAct("s", "granite/g"), id="cache").run("hi?", user="u")
    assert "cache_control" not in chat.calls[0].request.content.decode()


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
        result = await h.wrap(ReAct("s", model, output=output), id="so").run("a-1?", user="u")
    answer = result.answer if isinstance(result.answer, dict) else result.answer.model_dump()
    assert answer == {"sku": "a-1", "units": 7}


async def test_an_answer_that_does_not_fit_is_asked_for_once_more() -> None:
    model = Spending(
        turns=[("Count", {"sku": "a-1", "units": "many"}), ("Count", {"sku": "a-1", "units": 7})]
    )
    async with Harness(config=Settings()) as h:
        result = await h.wrap(ReAct("s", model, output=Count), id="so").run("a-1?", user="u")
    assert result.answer == Count(sku="a-1", units=7)
    told = model.seen[1][-1].text  # LangChain's tool strategy says what was wrong
    assert "Failed to parse structured output" in told and "units" in told


async def test_a_second_answer_that_does_not_fit_fails_the_run_saying_why() -> None:
    wrong = ("Count", {"sku": "a-1", "units": "many"})
    model = Spending(turns=[wrong, wrong, wrong])
    async with Harness(config=Settings()) as h:
        result = await h.wrap(ReAct("s", model, output=Count), id="so").run("a-1?", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "did not fit the output schema twice" in result.error.message
    assert len(model.seen) == 2


async def test_without_structured_output_langchain_asks_again_as_it_does() -> None:
    wrong = ("Count", {"sku": "a-1", "units": "many"})
    model = Spending(turns=[wrong, wrong, ("Count", {"sku": "a-1", "units": 7})])
    async with Harness(config=Settings()) as h:
        result = await h.wrap(ReAct("s", model, output=Count), id="so").run(
            "a-1?", user="u", without={"structured_output"}
        )
    assert result.answer == Count(sku="a-1", units=7) and len(model.seen) == 3


async def test_a_provider_strategys_refusal_is_repaired_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain.agents.structured_output import StructuredOutputValidationError

    calls: list[Any] = []

    async def handler(request: Any) -> Any:
        calls.append(request.messages)
        if len(calls) < 3:
            raise StructuredOutputValidationError(
                "Count", ValueError("units: not a number"), AIMessage("x")
            )
        return "fixed"

    monkeypatch.setattr(automatic, "on", lambda feature: True)
    request = SimpleNamespace(
        response_format=Count,
        messages=[],
        state={"messages": []},
        override=lambda **c: SimpleNamespace(**{**vars(request), **c}),
    )
    middleware = automatic.StructuredRepair()
    with pytest.raises(ModelError, match="did not fit the output schema twice: units"):
        await middleware.awrap_model_call(request, handler)  # type: ignore[arg-type]
    assert calls[1][-1].text.startswith(automatic.REPAIR)
    calls.clear()
    calls.append([])  # the second call fails, the repair answers
    assert await middleware.awrap_model_call(request, handler) == "fixed"  # type: ignore[arg-type]


# --------------------------------------------------------------------------- budget
async def test_at_its_token_limit_a_run_answers_with_what_it_has_and_is_partial() -> None:
    runs.clear()
    model = Spending(turns=[("stock", {"sku": "a-1"}), ("stock", {"sku": "a-2"}), "a-1: 7"])
    async with Harness(config=Settings()) as h:
        agent = h.wrap(ReAct("s", model), id="b", tools=[stock], budget=Budget(max_tokens=150))
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
        agent = h.wrap(ReAct("s", model), id="b", tools=[stock], budget=Budget(max_tokens=110))
        result = await agent.run("a-1?", user="u")
    assert result.status is RunStatus.PARTIAL and runs == []


async def test_below_its_limit_a_run_is_not_stopped() -> None:
    model = Spending(turns=[("stock", {"sku": "a-1"}), "a-1: 7"])
    async with Harness(config=Settings()) as h:
        agent = h.wrap(ReAct("s", model), id="b", tools=[stock], budget=Budget(max_tokens=111))
        result = await agent.run("a-1?", user="u")
    assert result.status is RunStatus.SUCCESS


async def test_without_a_budget_there_is_no_limit_but_spending_is_counted() -> None:
    runs.clear()
    model = Spending(turns=[("stock", {"sku": "a-1"}), "a-1: 7"])
    async with Harness(config=Settings()) as h:
        agent = h.wrap(ReAct("s", model), id="b", tools=[stock], budget=Budget(max_tokens=1))
        result = await agent.run("a-1?", user="u", without={"budget"})
    assert result.status is RunStatus.SUCCESS
    assert runs[0].replay.journal.spent["tokens"] == 220  # both replies, counted


async def test_a_model_the_price_table_has_not_is_said_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = Spending(turns=["a", "b"], model_name="local/unpriced")
    async with Harness(config=Settings()) as h:
        agent = h.wrap(ReAct("s", model), id="b", budget=Budget(max_cost=1.0))
        with caplog.at_level("WARNING", logger="trellis.run"):
            await agent.run("x", user="u")
            await agent.run("x", user="u")
    said = [r for r in caplog.records if "local/unpriced is not in the price table" in r.message]
    assert len(said) == 1


@respx.mock
async def test_the_gateways_cost_is_the_runs_before_the_price_table() -> None:
    runs.clear()
    said = {"cost": {"total_cost": 0.25}}
    respx.post(f"{GATEWAY}/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=tool_call("stock", {"sku": "a-1"})),
            httpx.Response(200, json=completion("7", **said)),  # type: ignore[arg-type]
        ]
    )
    async with harness() as h:
        agent = h.wrap(ReAct("s", "granite/g"), id="cost", tools=[stock])
        assert (await agent.run("a-1?", user="u")).answer == "7"
    spent = runs[0].replay.journal.spent
    assert spent["cost"] == 0.25 and spent["model_calls"] == 2  # granite: not in the table


def test_a_reply_without_the_gateways_cost_says_none() -> None:
    assert budget.gateway_cost({}) is None
    assert budget.gateway_cost({"token_usage": {"cost": {"total_cost": True}}}) is None
    assert budget.gateway_cost({"token_usage": {"cost": {"total_cost": 1}}}) == 1.0


async def test_a_structured_answer_at_the_budget_is_asked_for_without_other_tools() -> None:
    model = Spending(turns=[("stock", {"sku": "a-1"}), ("Count", {"sku": "a-1", "units": 7})])
    async with Harness(config=Settings()) as h:
        agent = h.wrap(
            ReAct("s", model, output=Count), id="b", tools=[stock], budget=Budget(max_tokens=1)
        )
        result = await agent.run("a-1?", user="u")
    assert result.status is RunStatus.PARTIAL and result.answer == Count(sku="a-1", units=7)
    assert model.requests[-1]["tools"] == ["Count"]  # only the answer's own tool


def test_the_answer_tools_of_a_given_strategy_are_its_schemas() -> None:
    from langchain.agents.structured_output import ToolStrategy

    assert automatic._answer_tools(ToolStrategy(Count)) == {"Count"}
    assert automatic._answer_tools(SCHEMA) == {"Count"}


def test_the_gateway_anthropic_client_reads_the_gateways_errors() -> None:
    client = react._anthropic_http(base_url="http://gw.test/anthropic")
    assert react._gateway_answered in client.event_hooks["response"]
