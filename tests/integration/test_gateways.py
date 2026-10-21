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
    BIFROST,
    OPENAI_COMPATIBLE,
    OPENAI_DEFAULT_URL,
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
    both = model_gateway("http://bf/v1", {"OPENAI_BASE_URL": "http://litellm/v1"})
    assert both is not None and both.kind == "bifrost" and both.capabilities == BIFROST
    compat = model_gateway(None, {"OPENAI_BASE_URL": "http://litellm/v1"})
    assert compat is not None and (compat.kind, compat.url) == (
        "openai_compatible",
        "http://litellm/v1",
    )
    assert compat.capabilities == OPENAI_COMPATIBLE and compat.source == "OPENAI_BASE_URL"
    openai = model_gateway(None, {"OPENAI_API_KEY": "sk"})
    assert openai is not None and openai.url == OPENAI_DEFAULT_URL
    assert model_gateway(None, {"OPENAI_API_KEY": " "}) is None
    assert (
        missing(None, "mcp", "mcp('x')")
        == "mcp('x') needs a gateway with mcp: Bifrost (BIFROST_URL)"
    )
    said = missing(compat, "code_mode", "Code Mode")
    assert said is not None and "openai-compatible gateway at http://litellm/v1 has none" in said
    assert missing(both, "code_mode", "Code Mode") is None


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
        assert h.model_gateway is not None and h.model_gateway.kind == "openai_compatible"
        result = await h.wrap(graph, id="compat").run("x", user="u")
    assert result.answer == "hi", result.error
    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer sk-compat"
    assert "x-bf-mcp-include-clients" not in request.headers  # no Bifrost header elsewhere
    assert not any(k.startswith("x-bf-dim-") for k in request.headers)
    assert json.loads(request.content)["model"] == "gpt-x"


def test_the_client_retries_where_no_gateway_does(monkeypatch: pytest.MonkeyPatch) -> None:
    from trellis.harness.react import MODEL_RETRIES, _gateway_model

    monkeypatch.setenv("OPENAI_BASE_URL", "http://litellm.test/v1")
    compat = _gateway_model("m", timeout=None, effort=None, settings=Settings())
    assert compat.max_retries is None  # the OpenAI client's own default: nothing else retries
    bifrost = _gateway_model(
        "m", timeout=None, effort=None, settings=Settings(bifrost_url="http://bf/v1")
    )
    assert bifrost.max_retries == MODEL_RETRIES == 0  # the gateway retries


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
