"""``ReAct`` and the model it is given by name: its context window detected (and every
threshold scaled by it), a vendor's model asked through the vendor's own API when the gateway
serves it (``native_client``), and the reasoning effort of each step (``effort``)."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from trellis import Harness, ReAct, Settings, tool
from trellis.contracts import ConfigurationError
from trellis.harness import models
from trellis.harness.automatic import DECIDING, ROUTINE, Effort, NativeClient, step_effort
from trellis.harness.middleware import ANSWER_CHECK
from trellis.harness.react import _native_model, detected_window

GATEWAY = "http://gw.test/v1"
STEPS = "Do this step by step: count a-1."


@pytest.fixture(autouse=True)
def _gateway(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("BIFROST_URL", GATEWAY)
    monkeypatch.setenv("BIFROST_VIRTUAL_KEY", "vk")
    for cached in (models._listed, models.serves, models._assumed):
        cached.cache_clear()
    yield


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock.

    Args:
        sku: The SKU, such as a-1.
    """
    return 7


def chat_reply(content: str = "ok") -> httpx.Response:
    body = {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    return httpx.Response(200, json=body)


def harness() -> Harness:
    return Harness(config=Settings(bifrost_url=GATEWAY, bifrost_virtual_key="vk"), gateway=False)


def offered(request: httpx.Request) -> set[str]:
    body = json.loads(request.content)
    return {t["function"]["name"] for t in body.get("tools") or []}


# --------------------------------------------------------------------------- the window
@respx.mock
@pytest.mark.parametrize(("n_ctx", "long_tools"), [(8_192, False), (1_000_000, True)])
async def test_the_window_the_gateway_reports_scales_what_react_does(
    n_ctx: int, long_tools: bool
) -> None:
    respx.get(f"{GATEWAY}/models").mock(
        return_value=httpx.Response(
            200, json={"data": [{"id": "granite/g", "meta": {"n_ctx": n_ctx}}]}
        )
    )
    chat = respx.post(f"{GATEWAY}/chat/completions").mock(return_value=chat_reply())
    assert detected_window("granite/g") == models.Window(n_ctx, "server")
    graph = ReAct("s", "granite/g")
    async with harness() as h:
        await h.wrap(graph, id=f"w{n_ctx}", tools=[stock]).run(STEPS, user="u")
    # a long task's tools need room: offered on a large window only
    assert ("write_todos" in offered(chat.calls[0].request)) is long_tools


def test_a_model_object_is_its_own_window_and_is_never_replaced() -> None:
    model = ChatAnthropic(model_name="claude-opus-5-5", api_key="x")  # type: ignore[call-arg]
    with respx.mock(assert_all_called=False) as mock:
        probe = mock.post("http://gw.test/anthropic/v1/messages")
        assert detected_window(model) == models.Window(1_000_000, "profile")
        assert detected_window(model, 9_000) == models.Window(9_000, "given")
        ReAct("s", model)
        assert not probe.called  # nothing probed: a model object is used as it is


# --------------------------------------------------------------------------- the client
def anthropic_stream(text: str, usage: dict[str, int] | None = None) -> bytes:
    events = [
        ("message_start", {"type": "message_start", "message": {"id": "m1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5", "content": [], "stop_reason": None, "usage": {"input_tokens": 12, "output_tokens": 0, **(usage or {})}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}),
        ("message_stop", {"type": "message_stop"}),
    ]  # fmt: skip
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events).encode()


class Anthropic:
    """The gateway's Anthropic endpoint, as the anthropic SDK reaches it (its own ``httpx2``
    client, which respx does not see): an empty request is invalid (the probe), a message is
    answered with ``text``, streamed or not."""

    def __init__(
        self, text: str = "hi", errors: Sequence[int] = (), usage: dict[str, int] | None = None
    ) -> None:
        self.text = text
        self.usage = usage or {}
        self.errors = list(errors)
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []

    def __call__(self, request: Any) -> Any:
        import httpx2

        body = json.loads(request.content or b"{}")
        self.requests.append(body)
        self.headers.append(dict(request.headers))
        if self.errors:
            error = {"type": "error", "error": {"type": "api_error", "message": "upstream down"}}
            return httpx2.Response(self.errors.pop(0), json=error)
        if body.get("stream"):
            return httpx2.Response(
                200,
                content=anthropic_stream(self.text, self.usage),
                headers={"content-type": "text/event-stream"},
            )
        message = {
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": self.text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 3, **self.usage},
        }
        return httpx2.Response(200, json=message)

    def client(self, **params: Any) -> Any:
        import httpx2
        from langchain_anthropic._client_utils import _AsyncHttpxClientWrapper

        return _AsyncHttpxClientWrapper(
            base_url=params["base_url"], transport=httpx2.MockTransport(self)
        )

    def patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The anthropic SDK's clients reach this endpoint."""
        import langchain_anthropic.chat_models as anthropic_models

        monkeypatch.setattr(anthropic_models, "_get_default_async_httpx_client", self.client)


@respx.mock
async def test_a_vendors_model_is_asked_through_its_own_api_when_the_gateway_serves_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = Anthropic()
    endpoint.patch(monkeypatch)
    respx.post("http://gw.test/anthropic/v1/messages").mock(return_value=httpx.Response(400))
    generic = respx.post(f"{GATEWAY}/chat/completions").mock(return_value=chat_reply("generic"))
    graph = ReAct("s", "anthropic/claude-sonnet-5-5")
    async with harness() as h:
        agent = h.wrap(graph, id="native", tools=[stock])
        assert (await agent.run("hi?", user="u")).answer == "hi"
        sent = endpoint.requests[-1]
        assert sent["model"] == "claude-sonnet-5-5"  # the vendor's own name, on its own route
        # its LangChain profile lists no effort levels: none is sent (no copied table)
        assert "output_config" not in sent
        assert endpoint.headers[-1]["x-api-key"] == "vk"
        assert not generic.called
        # without it, the generic client — the same graph
        result = await agent.run("hi?", user="u", without={"native_client"})
        assert result.answer == "generic" and generic.called


@respx.mock
async def test_a_gateway_that_does_not_serve_the_api_gets_the_generic_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    respx.post("http://gw.test/anthropic/v1/messages").mock(return_value=httpx.Response(404))
    generic = respx.post(f"{GATEWAY}/chat/completions").mock(return_value=chat_reply("generic"))
    with caplog.at_level("INFO", logger="trellis.models"):
        graph = ReAct("s", "anthropic/claude-sonnet-5-5")
        ReAct("s", "anthropic/claude-sonnet-5-5")
    assert [r.message for r in caplog.records].count(
        "anthropic/claude-sonnet-5-5: the gateway does not serve anthropic's own API; the "
        "generic client is used"
    ) == 1
    async with harness() as h:
        assert (await h.wrap(graph, id="g").run("hi?", user="u")).answer == "generic"
    assert json.loads(generic.calls[0].request.content)["model"] == "anthropic/claude-sonnet-5-5"


@respx.mock
def test_an_openai_model_is_asked_through_the_responses_api() -> None:
    respx.post(f"{GATEWAY}/responses").mock(return_value=httpx.Response(400))
    client = _native_model("openai/gpt-6-sol", timeout=None, window=922_000)
    assert (
        client is not None and client.use_responses_api and client.model_name == "openai/gpt-6-sol"
    )
    assert _native_model("granite/g", timeout=None, window=8_192) is None  # no vendor


def test_without_a_gateway_there_is_no_native_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BIFROST_URL")
    assert _native_model("anthropic/claude-sonnet-5-5", timeout=None, window=1) is None


# --------------------------------------------------------------------------- effort
def request_of(messages: list[Any], model: Any, state: list[Any] | None = None) -> Any:
    calls: list[Any] = []

    def override(**changes: Any) -> Any:
        calls.append(changes)
        return SimpleNamespace(**{**vars(request), **changes})

    request = SimpleNamespace(
        messages=messages,
        model=model,
        model_settings={"temperature": 0},
        state={"messages": state if state is not None else messages},
        override=override,
    )
    return request


async def sent_effort(middleware: Any, request: Any) -> Any:
    seen: list[Any] = []

    async def handler(req: Any) -> Any:
        seen.append(req)
        return "reply"

    await middleware.awrap_model_call(request, handler)
    return seen[0].model_settings.get("reasoning_effort")


OPUS = ChatAnthropic(model_name="claude-opus-5-5", api_key="x")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("messages", "level"),
    [
        ([HumanMessage("count")], DECIDING),  # the first call: the plan
        ([HumanMessage("count"), AIMessage("a"), ToolMessage("7", tool_call_id="1")], ROUTINE),
        (
            [
                HumanMessage("count"),
                AIMessage("a"),
                ToolMessage("boom", tool_call_id="1", status="error"),
            ],
            DECIDING,  # after a tool failed
        ),
        ([HumanMessage("count"), AIMessage("draft"), HumanMessage(ANSWER_CHECK)], DECIDING),
    ],
)
async def test_each_step_has_its_effort(messages: list[Any], level: str) -> None:
    state = [m for m in messages if not (isinstance(m, HumanMessage) and m.text == ANSWER_CHECK)]
    request = request_of(messages, OPUS, state)
    assert step_effort(request) == level
    assert await sent_effort(Effort(), request) == level


async def test_a_fixed_effort_a_model_without_effort_and_without_effort() -> None:
    first = [HumanMessage("count")]
    assert await sent_effort(Effort("max"), request_of(first, OPUS)) == "max"
    assert await sent_effort(Effort("minimal"), request_of(first, OPUS)) == "low"  # the nearest
    granite = SimpleNamespace(profile=None, model_name="granite/granite-3.3-2b")
    assert await sent_effort(Effort("high"), request_of(first, granite)) is None
    with pytest.raises(ConfigurationError, match="effort is one of"):
        Effort("extreme")


async def test_without_effort_none_is_sent() -> None:
    from tests.support.chat_model import ScriptedChatModel

    profile: Any = {"reasoning_effort_levels": ["low", "high"]}
    model = ScriptedChatModel(turns=["a", "b"], profile=profile)
    async with Harness(config=Settings()) as h:
        agent = h.wrap(ReAct("s", model), id="effort")
        await agent.run("x", user="u")
        await agent.run("x", user="u", without={"effort"})
    assert model.requests[0].get("reasoning_effort") == DECIDING
    assert "reasoning_effort" not in model.requests[1]


async def test_the_native_client_steps_aside_without_it() -> None:
    seen: list[Any] = []

    async def handler(req: Any) -> Any:
        seen.append(req.model)
        return "reply"

    native = object()
    request = request_of([HumanMessage("x")], "generic")
    await NativeClient(native).awrap_model_call(request, handler)
    assert seen == [native]
