"""Where a model name goes — one gateway per deployment, never a fallback: Bifrost when
``BIFROST_URL`` is set, else the OpenAI-compatible endpoint the OpenAI SDK's own variables
name, else none (a model object, and a name is an error naming the three). What a gateway
cannot do is off, and ``trellis doctor`` says so."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from tests.support.gateway import URL, FakeGateway
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Settings
from trellis.contracts import ConfigurationError
from trellis.harness import doctor
from trellis.harness.evals import EvalServices, llm_judge
from trellis.harness.gateways import (
    BIFROST_CLIENT_RETRIES,
    OPENAI_DEFAULT_URL,
    Bifrost,
    Capability,
    GatewayKind,
    OpenAICompatible,
    missing,
    model_gateway,
)

REPLY = {
    "id": "c",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
}


@pytest.fixture(autouse=True)
def _no_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("BIFROST_URL", "BIFROST_VIRTUAL_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_one_gateway_is_chosen_never_a_chain() -> None:
    bifrost = Settings(bifrost_url="http://bf/v1")
    both = model_gateway(bifrost, {"OPENAI_BASE_URL": "http://litellm/v1"})
    assert isinstance(both, Bifrost) and both.kind is GatewayKind.BIFROST
    assert both.capabilities == frozenset(Capability) and both.probe_key == "no-key"
    compat = model_gateway(Settings(), {"OPENAI_BASE_URL": "http://litellm/v1"})
    assert compat is not None and (compat.kind, compat.url) == (
        "openai_compatible",
        "http://litellm/v1",
    )
    assert isinstance(compat, OpenAICompatible) and compat.source == "OPENAI_BASE_URL"
    assert compat.capabilities == {Capability.MODELS} and compat.probe_key == "no-key"
    openai = model_gateway(Settings(), {"OPENAI_API_KEY": "sk"})
    assert openai is not None and openai.url == OPENAI_DEFAULT_URL and openai.probe_key == "sk"
    assert model_gateway(Settings(), {"OPENAI_API_KEY": " "}) is None
    assert (
        missing(None, Capability.MCP, "mcp('x')")
        == "mcp('x') needs a gateway with mcp: Bifrost (BIFROST_URL)"
    )
    said = missing(compat, Capability.CODE_MODE, "Code Mode")
    assert said is not None and "OpenAI-compatible gateway at http://litellm/v1 has none" in said
    assert missing(both, Capability.CODE_MODE, "Code Mode") is None


def test_a_name_with_no_gateway_says_where_names_go() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_BASE_URL / OPENAI_API_KEY"):
        ReAct(system="s", model="some/model")
    assert ReAct(system="s", model=ScriptedChat(["hi"])) is not None  # an object needs none


@respx.mock
async def test_a_name_on_an_openai_compatible_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://litellm.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-compat")
    route = respx.post("http://litellm.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=REPLY)
    )
    graph = ReAct(system="s", model="gpt-x", context_window=50_000)
    async with Harness(config=Settings()) as h:
        assert h.model_gateway is not None
        assert h.model_gateway.kind is GatewayKind.OPENAI_COMPATIBLE
        result = await h.wrap(graph, id="compat").run("x", user="u")
    assert result.answer == "hi", result.error
    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer sk-compat"
    assert "x-bf-mcp-include-clients" not in request.headers  # no Bifrost header elsewhere
    assert not any(k.startswith("x-bf-dim-") for k in request.headers)
    assert json.loads(request.content)["model"] == "gpt-x"


def test_the_client_retries_where_no_gateway_does(monkeypatch: pytest.MonkeyPatch) -> None:
    from trellis.harness.react import _gateway_model

    monkeypatch.setenv("OPENAI_BASE_URL", "http://litellm.test/v1")
    compat = _gateway_model("m", effort=None, settings=Settings())
    assert compat.max_retries is None  # the OpenAI client's own default: nothing else retries
    assert compat.request_timeout is None  # a call's bound is the run's, applied once
    bifrost = _gateway_model("m", effort=None, settings=Settings(bifrost_url="http://bf/v1"))
    assert bifrost.max_retries == BIFROST_CLIENT_RETRIES == 0  # the gateway retries


async def test_a_judge_by_name_on_an_openai_compatible_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeGateway(turns=['{"score": 1, "reason": "right"}', "again"])
    fake.serve_openai_compatible(monkeypatch)
    services = EvalServices.of(Settings())
    judge = llm_judge("Answers the question.", model="judge-model")
    model = services.model("judge-model")
    reply = await model.complete([{"role": "user", "content": "grade"}], temperature=0)
    await model.complete([{"role": "user", "content": "again"}])  # the same client
    assert reply["choices"][0]["message"]["content"].startswith('{"score"')
    assert json.loads(fake.completions[0].content)["model"] == "judge-model"
    assert judge.name
    with pytest.raises(ConfigurationError, match="is Bifrost's"):
        services.model()  # the alias is Bifrost's routing


def test_a_judge_by_name_without_a_gateway() -> None:
    with pytest.raises(ConfigurationError, match="llm_judge with a model name needs a gateway"):
        EvalServices.of(Settings()).model("m")


@respx.mock
def test_doctor_says_which_gateway_and_what_is_off(
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = {"OPENAI_BASE_URL": "http://litellm.test/v1", "OPENAI_API_KEY": "sk-1234567890"}
    respx.get("http://litellm.test/v1/models").mock(return_value=httpx.Response(200, json={}))
    assert doctor.main([], env) == 0
    out = capsys.readouterr().out
    assert "OpenAI-compatible at http://litellm.test/v1 (OPENAI_BASE_URL and OPENAI_API_KEY)" in out
    assert "off: Code Mode" in out and "off: model names" not in out
    assert "ok    gateway: http://litellm.test/v1/models answered" in out
    said = doctor.gateway_said(Settings(), {})
    assert said[0].startswith("none: models are objects")
    assert any(line.startswith("off: model names") for line in said)
    assert doctor.gateway_said(Settings(bifrost_url=URL), {}) == [f"Bifrost at {URL} (BIFROST_URL)"]
    assert "Gateway" not in doctor.report([("a", "b", "c")], [], [])  # none said: no section
