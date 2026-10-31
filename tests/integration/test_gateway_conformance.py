"""Every registered route, auth strategy and profile, against stubs of each wire format.

Routes (:func:`trellis.harness.gateways.registry`, the entry points' among them) against
OpenAI's chat completions and the Anthropic, Gemini and Bifrost endpoints that speak it — the
chat model a model name becomes, invoked and streamed, the judge and ``doctor``'s probe; the
provider SDKs' route against OpenAI's and Anthropic's own wire formats, and the other providers
through LangChain's ``init_chat_model``, whose packages the extras install. Auth strategies
(bifrost-sdk's ``AUTH_STRATEGIES``) by name, as ``LLM_AUTH`` gives them: a bearer key, a Google
token that expires and is refreshed, an AWS SigV4 signature recomputed here from the
specification. Profiles: what the memory service is told a model accepts, from LangChain's
profile. The endpoints are ``respx`` stubs."""

from __future__ import annotations

import hashlib
import hmac
import json
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import google.auth
import httpx
import pytest
import respx
from bifrost_sdk import AUTH_STRATEGIES, NO_GATEWAY_TOOLS
from bifrost_sdk.auth import BEDROCK, CLOUD_PLATFORM
from google.auth import credentials as google_credentials

from trellis import Settings
from trellis.contracts import ConfigurationError, ModelProfile
from trellis.harness import doctor, gateways
from trellis.harness.gateways import (
    LANGCHAIN_PROVIDERS,
    Capability,
    ModelGateway,
    OpenAICompatible,
    ProviderSDK,
    model_gateway,
    registry,
)

ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------- the wire formats
@dataclass(frozen=True)
class Wire:
    """One endpoint speaking chat completions: where it is, the model it serves."""

    name: str
    base_url: str
    model: str

    @property
    def completions(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


WIRES = [
    Wire("openai", "https://api.openai.com/v1", "gpt-4o-mini"),
    Wire("anthropic-compatible", "https://api.anthropic.com/v1/", "claude-haiku-4-5"),
    Wire(
        "gemini-compatible",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-2.5-flash",
    ),
    Wire("bifrost", "http://bifrost.test/v1", "openai/gpt-4o-mini"),
    Wire("ollama", "http://localhost:11434/v1", "granite3.3:2b"),
]


def _reply(content: str) -> dict[str, Any]:
    return {
        "id": "c1",
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


def _stream(text: str) -> httpx.Response:
    chunks = [
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
        }
        for part in text
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def _answer(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    return _stream("pong") if body.get("stream") else httpx.Response(200, json=_reply("pong"))


@pytest.fixture(autouse=True)
def _no_endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_GATEWAY", "LLM_AUTH", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _judge_on_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    """The judge's OpenAI client sends through httpx, where ``respx`` stubs the endpoint, with
    the auth strategy the route gave it (as ``tests/conftest.py`` does for ``ChatOpenAI``)."""
    import openai

    class OnHttpx(openai.AsyncOpenAI):
        def __init__(self, **kwargs: Any) -> None:
            given: Any = kwargs.get("http_client")
            if not isinstance(given, httpx.AsyncClient):  # ChatOpenAI's is already
                kwargs["http_client"] = httpx.AsyncClient(auth=getattr(given.auth, "inner", None))
            super().__init__(**kwargs)

    monkeypatch.setattr(openai, "AsyncOpenAI", OnHttpx)


# ----------------------------------------------------------------- routes
#: How each route served over a URL names each wire's model: every route the registry holds
#: has a case (``test_every_registered_route_is_covered``).
NAMED: dict[str, dict[str, str]] = {
    "openai-compatible": {wire.name: wire.model for wire in WIRES},  # as given
    "bifrost": {  # Bifrost's provider/model
        "openai": "openai/gpt-4o-mini",
        "anthropic-compatible": "anthropic/claude-haiku-4-5",
        "gemini-compatible": "gemini/gemini-2.5-flash",
        "bifrost": "openai/gpt-4o-mini",
        "ollama": "granite3.3/2b",  # a provider before the first mark
    },
}
#: The routes served over a URL, by name.
URL_ROUTES = sorted(n for n, r in registry().items() if issubclass(r, OpenAICompatible))


def test_every_registered_route_is_covered() -> None:
    assert set(registry()) == {*NAMED, ProviderSDK.name}
    assert set(URL_ROUTES) == set(NAMED)


@pytest.mark.parametrize("name", URL_ROUTES)
@pytest.mark.parametrize("wire", WIRES, ids=lambda wire: wire.name)
async def test_every_route_completes_against_every_wire_format(wire: Wire, name: str) -> None:
    settings = Settings(llm_base_url=wire.base_url, llm_api_key="k", llm_gateway=name)
    route = model_gateway(settings)
    expected = NAMED[name][wire.name]
    with respx.mock(assert_all_called=True) as stub:
        sent = stub.post(wire.completions).mock(side_effect=_answer)
        chat = route.chat_model(wire.model, effort=None)
        assert (await chat.ainvoke("ping")).content == "pong"
        assert "".join([str(c.content) async for c in chat.astream("ping")]) == "pong"
        judged = await route.judge(wire.model).complete([{"role": "user", "content": "grade"}])
    assert judged["choices"][0]["message"]["content"] == "pong"
    assert len(sent.calls) == 3
    for call in sent.calls:
        request = call.request
        assert json.loads(request.content)["model"] == expected == route.routed(wire.model)
        assert request.headers["authorization"] == "Bearer k"
        bifrost = {k for k in request.headers if k.startswith("x-bf-")}
        # the deny-all MCP scope is Bifrost's; another endpoint gets no x-bf-* header at all
        assert bifrost == (set(NO_GATEWAY_TOOLS) if route.has(Capability.MCP) else set())


@pytest.mark.parametrize("name", URL_ROUTES)
@respx.mock
async def test_every_route_is_probed_by_doctor_with_its_auth(name: str) -> None:
    models = respx.get("http://llm.test/v1/models").mock(return_value=httpx.Response(200))
    mcp = respx.post("http://llm.test/mcp").mock(return_value=httpx.Response(200, json={}))
    settings = Settings(llm_base_url="http://llm.test/v1", llm_api_key="k", llm_gateway=name)
    [check] = await doctor.checks(settings)
    assert check.ok, check.said
    assert models.calls.last.request.headers["authorization"] == "Bearer k"
    assert mcp.calls.last.request.headers["authorization"] == "Bearer k"


@respx.mock
async def test_the_provider_sdks_route_speaks_openais_own_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An OpenAI model on the OpenAI SDK, with the key it reads itself."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    openai = respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_answer)
    route = model_gateway(Settings())
    chat = route.chat_model("gpt-4o-mini", effort=None)
    assert (await chat.ainvoke("ping")).content == "pong"
    assert openai.calls.last.request.headers["authorization"] == "Bearer sk-openai"
    assert json.loads(openai.calls.last.request.content)["model"] == "gpt-4o-mini"


@pytest.mark.parametrize(
    ("model", "provider", "name"),
    [
        ("gemini-2.5-flash", "google_genai", "gemini-2.5-flash"),
        ("vertex:gemini-2.5-pro", "google_vertexai", "gemini-2.5-pro"),
        ("bedrock/anthropic.claude-x", "bedrock_converse", "anthropic.claude-x"),
        ("google_vertexai:gemini-2.5-pro", "google_vertexai", "gemini-2.5-pro"),
        ("claude-haiku-4-5", "anthropic", "claude-haiku-4-5"),
    ],
)
def test_the_provider_sdks_route_asks_langchain_for_the_others(
    monkeypatch: pytest.MonkeyPatch, model: str, provider: str, name: str
) -> None:
    import langchain.chat_models

    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(
        langchain.chat_models,
        "init_chat_model",
        lambda model, *, model_provider: asked.append((model, model_provider)) or "made",
    )
    assert ProviderSDK().chat_model(model, effort=None) == "made"
    assert asked == [(name, provider)]


@pytest.mark.parametrize(
    ("extra", "provider"),
    [
        ("anthropic", "anthropic"),
        ("gemini", LANGCHAIN_PROVIDERS["gemini"]),
        ("vertex", LANGCHAIN_PROVIDERS["vertex"]),
        ("bedrock", LANGCHAIN_PROVIDERS["bedrock"]),
    ],
)
def test_each_provider_extra_installs_the_package_langchain_loads(
    extra: str, provider: str
) -> None:
    from langchain.chat_models.base import _BUILTIN_PROVIDERS

    module = _BUILTIN_PROVIDERS[provider][0].partition(".")[0]
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    names = [r.split(">")[0].split("[")[0] for r in extras[extra]]
    assert module.replace("_", "-") in names
    [everything] = extras["all"]
    assert extra in everything.partition("[")[2].rstrip("]").split(",")  # the dev env has it


# ----------------------------------------------------------------- the entry points
class Local(OpenAICompatible):
    """A route a package adds: an OpenAI-compatible endpoint of its own name."""

    name = "local"
    title = "a local server"


@dataclass
class EntryPoint:
    name: str
    value: str
    loaded: Any

    def load(self) -> Any:
        return self.loaded


def _entry_points(*found: EntryPoint) -> Any:
    def entry_points(*, group: str) -> list[EntryPoint]:
        assert group == gateways.ENTRY_POINTS
        return list(found)

    return entry_points


def test_a_package_adds_a_route_by_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gateways.importlib.metadata,
        "entry_points",
        _entry_points(EntryPoint("local", "pkg:Local", Local)),
    )
    gateways.registry.cache_clear()
    try:
        route = model_gateway(Settings(llm_base_url="http://l/v1", llm_gateway="local"))
        assert type(route) is Local and route.routed("m") == "m"
        said = gateways.missing(route, Capability.MCP, "MCP")
        assert said == "MCP needs a gateway with mcp: Bifrost (LLM_GATEWAY=bifrost)"
    finally:
        gateways.registry.cache_clear()


@pytest.mark.parametrize(
    ("entry", "said"),
    [
        (EntryPoint("bifrost", "pkg:Mine", Local), "which a route already has"),
        (EntryPoint("odd", "pkg:thing", object()), "is not a ModelGateway subclass"),
        (EntryPoint("odd", "pkg:Thing", dict), "is not a ModelGateway subclass"),
    ],
)
def test_an_entry_point_that_is_not_a_new_route_is_refused(
    monkeypatch: pytest.MonkeyPatch, entry: EntryPoint, said: str
) -> None:
    monkeypatch.setattr(gateways.importlib.metadata, "entry_points", _entry_points(entry))
    gateways.registry.cache_clear()
    try:
        with pytest.raises(ConfigurationError, match=said):
            registry()
    finally:
        gateways.registry.cache_clear()


# ----------------------------------------------------------------- auth strategies
class FakeGoogleCredentials(google_credentials.Credentials):
    """Credentials whose every refresh issues the next token, valid for an hour."""

    def __init__(self) -> None:
        super().__init__()
        self.refreshes = 0

    def refresh(self, request: Any) -> None:
        self.refreshes += 1
        self.token = f"ya29.token-{self.refreshes}"
        self.expiry = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1)

    def expire(self) -> None:
        self.expiry = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=1)


@pytest.fixture
def google_default(monkeypatch: pytest.MonkeyPatch) -> FakeGoogleCredentials:
    """Application default credentials: the fake, asked for the cloud-platform scope."""
    fake = FakeGoogleCredentials()

    def default(scopes: list[str] | None = None, **_: Any) -> tuple[Any, str]:
        assert scopes == [CLOUD_PLATFORM]
        return fake, "project"

    monkeypatch.setattr(google.auth, "default", default)
    return fake


AWS_KEY, AWS_SECRET, AWS_REGION = (
    "AKIDEXAMPLE",
    "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
    "us-east-1",
)


@pytest.fixture
def aws_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An AWS environment of known credentials, and nothing of the machine's."""
    for name in ("AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", AWS_KEY)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", AWS_SECRET)
    monkeypatch.setenv("AWS_DEFAULT_REGION", AWS_REGION)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def _sigv4(request: Any) -> bool:
    """Whether ``request``'s SigV4 signature is the one the specification gives, computed
    here, not by botocore."""
    authorization = request.headers["Authorization"]
    fields = dict(part.split("=", 1) for part in authorization.split(" ", 1)[1].split(", "))
    signed = fields["SignedHeaders"].split(";")
    amz_date = request.headers["X-Amz-Date"]
    scope = f"{amz_date[:8]}/{AWS_REGION}/{BEDROCK}/aws4_request"
    canonical = "\n".join(
        [
            request.method,
            request.url.path,
            request.url.query.decode(),
            "".join(f"{name}:{request.headers[name].strip()}\n" for name in signed),
            ";".join(signed),
            hashlib.sha256(request.content).hexdigest(),
        ]
    )
    to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()]
    )
    key = f"AWS4{AWS_SECRET}".encode()
    for part in (amz_date[:8], AWS_REGION, BEDROCK, "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    expected = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    return fields["Credential"] == f"{AWS_KEY}/{scope}" and authorization.endswith(expected)


AUTHENTICATED = {
    "bearer": lambda request: request.headers["authorization"] == "Bearer sk-1",
    "google": lambda request: request.headers["authorization"] == "Bearer ya29.token-1",
    "aws": _sigv4,
}


def test_every_auth_strategy_is_covered() -> None:
    assert set(AUTH_STRATEGIES) == set(AUTHENTICATED)


@pytest.mark.parametrize("auth", sorted(AUTH_STRATEGIES))
async def test_every_auth_strategy_by_name_authenticates_every_model_request(
    auth: str, google_default: FakeGoogleCredentials, aws_env: None
) -> None:
    url = "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
    settings = Settings(
        llm_base_url=url, llm_auth=auth, llm_api_key="sk-1" if auth == "bearer" else None
    )
    route = model_gateway(settings)
    with respx.mock() as stub:
        sent = stub.post(f"{url}/chat/completions").mock(side_effect=_answer)
        models = stub.get(f"{url}/models").mock(return_value=httpx.Response(200))
        stub.post("https://bedrock-runtime.us-east-1.amazonaws.com/openai/mcp").respond(404)
        chat = route.chat_model("anthropic.claude-x", effort=None)
        assert (await chat.ainvoke("ping")).content == "pong"
        assert chat.invoke("ping").content == "pong"  # the sync client is signed too
        await route.judge("anthropic.claude-x").complete([{"role": "user", "content": "x"}])
        [check] = await doctor.checks(settings)
    assert check.ok, check.said
    assert all(AUTHENTICATED[auth](call.request) for call in sent.calls)
    assert len(sent.calls) == 3 and AUTHENTICATED[auth](models.calls.last.request)
    endpoint = route.endpoint("anthropic.claude-x")
    assert endpoint is not None and endpoint.auth == auth
    # a login's secret is its own: none is sent to the memory service
    secret = endpoint.api_key.get_secret_value() if endpoint.api_key else None
    assert secret == ("sk-1" if auth == "bearer" else None)


async def test_a_google_token_is_refreshed_once_it_expires(
    google_default: FakeGoogleCredentials,
) -> None:
    wire = WIRES[2]
    route = model_gateway(Settings(llm_base_url=wire.base_url, llm_auth="google"))
    with respx.mock() as stub:
        sent = stub.post(wire.completions).mock(side_effect=_answer)
        chat = route.chat_model(wire.model, effort=None)
        await chat.ainvoke("a")
        await chat.ainvoke("b")  # still valid: not refreshed
        google_default.expire()
        await chat.ainvoke("c")
    tokens = [call.request.headers["authorization"] for call in sent.calls]
    assert tokens == ["Bearer ya29.token-1", "Bearer ya29.token-1", "Bearer ya29.token-2"]
    assert google_default.refreshes == 2


class Challenged(httpx.Auth):
    """A strategy that reads the answer and sends the request again (a challenge)."""

    requires_response_body = True

    def auth_flow(self, request: Any) -> Any:
        request.headers["authorization"] = "Bearer first"
        response = yield request
        if response.status_code == 401:
            request.headers["authorization"] = "Bearer answered"
            yield request


async def test_a_strategy_runs_whole_on_the_openai_clients_own_library() -> None:
    """``openai_auth`` hands the OpenAI SDK's HTTP library (its httpx fork) an ``Auth`` of its
    own that runs an ``httpx.Auth`` flow whole: sync and async, a challenge answered."""
    import openai

    seen: list[str] = []

    library = gateways.importlib.import_module(
        openai.DefaultAsyncHttpxClient.__mro__[1].__module__.partition(".")[0]
    )

    def answer(request: Any) -> Any:
        seen.append(request.headers["authorization"])
        status = 200 if request.headers["authorization"] == "Bearer answered" else 401
        return library.Response(status, json={}, request=request)

    auth = gateways.openai_auth(Challenged())
    assert isinstance(auth, library.Auth) and gateways.openai_auth(None) is None
    transport = library.MockTransport(answer)
    with library.Client(transport=transport, auth=auth) as client:
        assert client.get("http://llm.test/v1/models").status_code == 200
    async with library.AsyncClient(transport=transport, auth=auth) as client:
        assert (await client.get("http://llm.test/v1/models")).status_code == 200
    assert seen == ["Bearer first", "Bearer answered"] * 2


# ----------------------------------------------------------------- profiles
PLAIN, REASONING = (
    ModelProfile(max_output_tokens=16384),
    ModelProfile(reasoning_output=True, temperature=False, max_output_tokens=100000),
)


@pytest.mark.parametrize(
    ("model", "profile"),
    [
        ("gpt-4o", PLAIN),
        ("openai/o3", REASONING),
        (
            "anthropic:claude-sonnet-4-5",
            ModelProfile(reasoning_output=True, max_output_tokens=64000),
        ),
        ("granite3.3:2b", ModelProfile()),  # LangChain knows nothing: a plain chat model
        (None, ModelProfile()),  # no model named: the service picks
    ],
    ids=["plain", "reasoning", "reasoning with temperature", "unknown", "none"],
)
@pytest.mark.parametrize("name", URL_ROUTES)
def test_every_profile_is_told_with_every_route(
    name: str, model: str | None, profile: ModelProfile
) -> None:
    route: ModelGateway = model_gateway(
        Settings(llm_base_url="http://llm.test/v1", llm_api_key="k", llm_gateway=name)
    )
    endpoint = route.endpoint(model)
    assert endpoint is not None
    assert endpoint.profile == profile
    assert endpoint.model == (route.routed(model) if model else None)
    assert (endpoint.base_url, endpoint.gateway, endpoint.auth) == (
        "http://llm.test/v1",
        name,
        "bearer",
    )
    assert endpoint.api_key is not None and endpoint.api_key.get_secret_value() == "k"
    assert ProviderSDK().endpoint(model) is None  # the providers' keys are their SDKs'
