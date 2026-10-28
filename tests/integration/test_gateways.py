"""Where a model name goes — model routing, one strategy, never a fallback: through Bifrost when
``BIFROST_URL`` is set (it wins over a provider key: said once, and by ``trellis doctor``), else
to its provider through the provider's own SDK and key. The settings given in code
(``Harness(config=)``) route the run's models over the environment's, the name is mapped to the
route's form, every framework's model calls follow the route (``ReAct``'s chat model per call,
an OpenAI agent's through ``RunConfig.model_provider``, Claude's CLI through its ``env``), and
a model the harness cannot route is said once. What a route cannot do is off."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from agents import Agent

from tests.support.gateway import URL, FakeGateway
from tests.support.models import ScriptedChat
from tests.support.openai_model import ScriptedModel
from trellis import Harness, ReAct, Settings
from trellis.contracts import ConfigurationError
from trellis.harness import doctor, gateways
from trellis.harness.adapters.claude import routed_env
from trellis.harness.evals import EvalServices, llm_judge
from trellis.harness.gateways import (
    BIFROST_CLIENT_RETRIES,
    PROVIDER_CLIENT_RETRIES,
    Bifrost,
    Capability,
    GatewayKind,
    ProviderKey,
    missing,
    model_gateway,
    routed_chat_model,
    split,
)
from trellis.harness.prompts import CUSTOM_HEADERS

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
    gateways._bifrost_wins.cache_clear()


def test_one_route_is_chosen_never_a_chain() -> None:
    bifrost = model_gateway(Settings(bifrost_url="http://bf/v1"))
    assert isinstance(bifrost, Bifrost) and bifrost.kind is GatewayKind.BIFROST
    assert bifrost.capabilities == frozenset(Capability) and bifrost.probe_key == "no-key"
    assert bifrost.origin == "http://bf"
    direct = model_gateway(Settings())
    assert isinstance(direct, ProviderKey) and direct.kind is GatewayKind.PROVIDER_KEY
    assert direct.capabilities == {Capability.MODELS} and direct.probe_key == "no-key"
    said = missing(direct, Capability.CODE_MODE, "Code Mode")
    assert said == "Code Mode needs a gateway with code_mode: Bifrost (BIFROST_URL)"
    assert missing(bifrost, Capability.CODE_MODE, "Code Mode") is None


@pytest.mark.parametrize(
    ("name", "provider", "model", "on_bifrost", "on_key"),
    [
        ("gpt-4o", "openai", "gpt-4o", "openai/gpt-4o", "openai:gpt-4o"),
        ("openai:gpt-4o", "openai", "gpt-4o", "openai/gpt-4o", "openai:gpt-4o"),
        ("openai/gpt-4o", "openai", "gpt-4o", "openai/gpt-4o", "openai:gpt-4o"),
        ("o3-mini", "openai", "o3-mini", "openai/o3-mini", "openai:o3-mini"),
        ("claude-sonnet", "anthropic", "claude-sonnet", "anthropic/claude-sonnet", None),
        ("gemini-pro", "gemini", "gemini-pro", "gemini/gemini-pro", "google_genai:gemini-pro"),
        ("mistral-large", "mistral", "mistral-large", None, "mistralai:mistral-large"),
        ("ollama/granite", "ollama", "granite", "ollama/granite", "ollama:granite"),
        ("granite", None, "granite", "granite", None),
    ],
)
def test_a_name_is_mapped_to_the_routes_form(
    name: str, provider: str | None, model: str, on_bifrost: str | None, on_key: str | None
) -> None:
    assert split(name) == (provider, model)
    if on_bifrost is not None:
        assert Bifrost("http://bf/v1", None).routed(name) == on_bifrost
    if on_key is not None:
        assert ProviderKey().routed(name) == on_key


def test_a_name_whose_provider_cannot_be_told_says_how_to_name_it() -> None:
    with pytest.raises(ConfigurationError, match=r"name its provider \('openai:granite'"):
        ProviderKey().routed("granite")
    assert ProviderKey().routed("anthropic/claude-x") == "anthropic:claude-x"


@respx.mock
async def test_a_name_through_a_provider_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://litellm.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-compat")
    route = respx.post("http://litellm.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=REPLY)
    )
    graph = ReAct(system_prompt="s", model="openai/gpt-x", context_window=50_000)
    async with Harness(config=Settings()) as h:
        assert h.model_gateway.kind is GatewayKind.PROVIDER_KEY
        result = await h.wrap(graph, id="direct").run("x", user="u")
    assert result.answer == "hi", result.error
    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer sk-compat"  # the SDK's own key
    assert "x-bf-mcp-include-clients" not in request.headers  # no Bifrost header elsewhere
    assert not any(k.startswith("x-bf-dim-") for k in request.headers)
    assert json.loads(request.content)["model"] == "gpt-x"


@respx.mock
async def test_the_settings_given_in_code_route_over_the_environment(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``ReAct`` is built before any harness; the harness running it, its settings given in
    code, routes its calls — through Bifrost, which wins over the provider key also set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-provider")
    gateway = respx.post(f"{URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=REPLY)
    )
    respx.post("http://gw.test/mcp").mock(
        return_value=httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})
    )
    graph = ReAct(system_prompt="s", model="gpt-x", context_window=50_000)
    with caplog.at_level(logging.INFO, logger="trellis.harness"):
        async with Harness(config=Settings(bifrost_url=URL, bifrost_virtual_key="vk")) as h:
            assert (await h.wrap(graph, id="coded").run("x", user="u")).answer == "hi"
        async with Harness(config=Settings(bifrost_url=URL, bifrost_virtual_key="vk")):
            pass
    request = gateway.calls[0].request
    assert request.headers["authorization"] == "Bearer vk"
    assert json.loads(request.content)["model"] == "openai/gpt-x"  # Bifrost's form
    assert request.headers["x-bf-dim-agent"] == "coded"  # the run's dimensions
    said = [r for r in caplog.records if "Bifrost wins over a provider key" in r.getMessage()]
    assert len(said) == 1 and "OPENAI_API_KEY" in said[0].getMessage()  # once


def test_one_layer_retries_a_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    direct = ProviderKey().chat_model("gpt-x", effort=None)
    # ReAct's ModelRetryMiddleware retries (middleware.retried): the client none
    assert direct.max_retries == PROVIDER_CLIENT_RETRIES == 0
    assert direct.request_timeout is None  # a call's bound is the run's, applied once
    bifrost = Bifrost("http://bf/v1", None).chat_model("m", effort=None)
    assert bifrost.max_retries == BIFROST_CLIENT_RETRIES == 0  # the gateway retries
    other = ProviderKey().chat_model("anthropic:claude-x", effort=None)
    assert type(other).__name__ == "ChatAnthropic"  # any other provider: init_chat_model
    assert other.max_retries == 2  # its own client's retries: middleware.retried leaves it


def test_a_routed_name_knows_its_window_and_is_routed_when_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = routed_chat_model("openai/gpt-4o")
    assert model.profile is not None and model.profile["max_input_tokens"] > 0
    assert routed_chat_model("ollama/granite").profile is None  # LangChain knows nothing
    monkeypatch.setenv("BIFROST_URL", "http://bf/v1")  # outside a run: the environment's route
    resolved = model.resolved()
    assert resolved.model_name == "openai/gpt-4o" and resolved is model.resolved()  # kept


async def test_a_judge_by_name_through_a_provider_key(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeGateway(turns=['{"score": 1, "reason": "right"}', "again"])
    fake.serve_provider_key(monkeypatch)
    services = EvalServices.of(Settings())
    judge = llm_judge("Answers the question.", model="openai:judge-model")
    model = services.model("openai:judge-model")
    reply = await model.complete([{"role": "user", "content": "grade"}], temperature=0)
    await model.complete([{"role": "user", "content": "again"}])  # the same client
    assert reply["choices"][0]["message"]["content"].startswith('{"score"')
    assert json.loads(fake.completions[0].content)["model"] == "judge-model"
    assert judge.name
    with pytest.raises(ConfigurationError, match="is Bifrost's"):
        services.model()  # the alias is Bifrost's routing
    with pytest.raises(ConfigurationError, match="an OpenAI model on the OpenAI SDK's"):
        services.model("anthropic:claude-x")
    with pytest.raises(ConfigurationError, match="is Bifrost's"):
        EvalServices(models=Bifrost(URL, None)).model("m")  # Bifrost, with no judge client


async def test_an_openai_agent_by_name_goes_through_bifrost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeGateway(turns=["routed"])
    fake.serve_models(monkeypatch)
    settings = Settings(bifrost_url=URL, bifrost_virtual_key="vk")
    async with Harness(config=settings, gateway=fake.gateway()) as h:
        result = await h.wrap(Agent(name="a", model="gpt-x"), id="named").run("x", user="u")
        again = await h.wrap(Agent(name="b"), id="default").run("x", user="u")
    assert result.answer == "routed", result.error
    assert again.status.value == "ERROR"  # the script is spent: the second call reached it
    [first, second] = fake.completions
    assert json.loads(first.content)["model"] == "openai/gpt-x"  # Bifrost's form
    assert json.loads(second.content)["model"].startswith("openai/")  # the SDK's default
    assert first.headers["authorization"] == "Bearer vk"
    assert first.headers["x-bf-mcp-include-clients"] == ""  # the deny-all MCP scope
    assert first.headers["x-bf-dim-agent"] == "named"  # whose run, read per request


async def test_a_run_config_with_its_own_provider_or_a_model_object_is_kept(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from agents import RunConfig
    from agents.models.openai_provider import OpenAIProvider

    own = RunConfig(model_provider=OpenAIProvider(api_key="theirs"))
    async with Harness(config=Settings(bifrost_url=URL), gateway=FakeGateway().gateway()) as h:
        agent = h.wrap(Agent(name="mine", model=ScriptedModel(["mine", "again"])), id="mine")
        with caplog.at_level(logging.WARNING, logger="trellis.harness"):
            assert (await agent.run("x", user="u")).answer == "mine"
            answered = await agent.run("x", user="u", framework_options={"run_config": own})
            assert answered.answer == "again"
    said = [r for r in caplog.records if "has a model object" in r.getMessage()]
    assert len(said) == 1 and "not Bifrost" in said[0].getMessage()  # once per agent


async def test_a_langchain_model_object_is_said_once_as_not_routed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    graph = ReAct(system_prompt="s", model=ScriptedChat(["one", "two"]))
    async with Harness(config=Settings(bifrost_url=URL), gateway=FakeGateway().gateway()) as h:
        agent = h.wrap(graph, id="objects")
        with caplog.at_level(logging.WARNING, logger="trellis.harness"):
            await agent.run("x", user="u")
            await agent.run("x", user="u")
    said = [r for r in caplog.records if "calls its provider directly" in r.getMessage()]
    assert len(said) == 1 and "objects" in said[0].getMessage()


def _claude_runtime(gateway: Any) -> Any:
    harness = SimpleNamespace(model_gateway=gateway)
    return SimpleNamespace(agent=SimpleNamespace(harness=harness))


def test_claude_is_routed_through_bifrosts_anthropic_route() -> None:
    bifrost = Bifrost("http://bf/v1", "vk")
    given = {CUSTOM_HEADERS: "x-bf-prompt-id: p1\nx-bf-mcp-include-tools: all"}
    env = routed_env(given, _claude_runtime(bifrost))
    assert env is not None
    assert env["ANTHROPIC_BASE_URL"] == "http://bf/anthropic"
    assert env["ANTHROPIC_API_KEY"] == "vk"
    lines = env[CUSTOM_HEADERS].splitlines()
    assert "x-bf-prompt-id: p1" in lines  # theirs kept
    assert "x-bf-mcp-include-tools: " in lines and "x-bf-mcp-include-tools: all" not in lines
    # the options' own endpoint (code over the environment): kept, with no Bifrost header
    theirs = {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}
    assert routed_env(theirs, _claude_runtime(bifrost)) is None
    own = routed_env({"ANTHROPIC_BASE_URL": "http://bf/anthropic"}, _claude_runtime(bifrost))
    assert own is not None and "ANTHROPIC_API_KEY" not in own  # their key, from their env
    assert routed_env({}, _claude_runtime(ProviderKey())) is None  # the CLI's own key


async def test_a_claude_run_starts_its_cli_on_bifrosts_route(tmp_path: Any) -> None:
    from claude_agent_sdk import ClaudeAgentOptions

    from tests.support.adapters import CLI

    record = tmp_path / "cli.json"
    env = {"FAKE_CLAUDE_SCRIPT": json.dumps([{"text": "ok"}]), "FAKE_CLAUDE_RECORD": str(record)}
    settings = Settings(bifrost_url=URL, bifrost_virtual_key="vk")
    async with Harness(config=settings, gateway=FakeGateway().gateway()) as h:
        agent = h.wrap(ClaudeAgentOptions(cli_path=CLI, env=env), id="claude")
        assert (await agent.run("x", user="ada")).answer == "ok"
    sent = json.loads(record.read_text())["custom_headers"].splitlines()
    # Bifrost's own headers, routed automatically: the deny-all MCP scope, whose run it is
    assert "x-bf-mcp-include-tools: " in sent and "x-bf-dim-agent: claude" in sent


@respx.mock
def test_doctor_says_how_models_are_routed_and_what_is_off(
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = {"OPENAI_BASE_URL": "http://litellm.test/v1", "OPENAI_API_KEY": "sk-1234567890"}
    assert doctor.main([], env) == 0
    out = capsys.readouterr().out
    assert "provider keys: each provider's own SDK reads its own (OPENAI_API_KEY, " in out
    assert "off: Code Mode" in out and "off: model names" not in out
    said = doctor.gateway_said(Settings(), {})
    assert said[0] == "provider keys: each provider's own SDK reads its own (none set)"
    both = doctor.gateway_said(Settings(bifrost_url=URL), {"ANTHROPIC_API_KEY": "k"})
    assert both[0] == f"Bifrost at {URL} (BIFROST_URL)"
    assert both[1].startswith("note: ANTHROPIC_API_KEY set, not used for model calls")
    assert "Gateway" not in doctor.report([("a", "b", "c")], [], [])  # none said: no section


async def test_a_routed_name_is_the_routed_models_in_every_way_it_is_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoked, streamed, bound to tools or to a schema — sync or async — a routed name's chat
    model is the one the route makes of it."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    made: list[Any] = []

    class Route:
        kind, url, key = GatewayKind.PROVIDER_KEY, "here", None

        def chat_model(self, model: str, *, effort: str | None) -> Any:
            replies = iter([AIMessage(content=f"{model} says hi")] * 8)
            made.append(GenericFakeChatModel(messages=replies))
            return made[-1]

    monkeypatch.setattr(gateways, "route_now", Route)
    model = routed_chat_model("openai/fake")
    assert model.invoke("x").content == "openai/fake says hi"  # sync
    assert "".join(str(c.content) for c in model.stream("x")) == "openai/fake says hi"
    assert (await model.ainvoke("x")).content == "openai/fake says hi"
    streamed = [str(c.content) async for c in model.astream("x")]
    assert "".join(streamed) == "openai/fake says hi"
    with pytest.raises(NotImplementedError):
        model.bind_tools([])  # the routed model's own answer: it binds no tools
    with pytest.raises(NotImplementedError):
        model.with_structured_output({"type": "object"})
    assert len(made) == 1  # one per route and event loop


def test_a_model_langchain_cannot_make_offline_has_no_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langchain_openai

    def broken(**kwargs: Any) -> Any:
        raise ValueError("no")

    assert routed_chat_model("anthropic:claude-sonnet-4-5").profile is not None
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", broken)
    assert routed_chat_model("gpt-4o").profile is None
