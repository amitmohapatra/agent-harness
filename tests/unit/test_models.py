"""What the harness knows of a model by its name (``trellis.harness.models``): its context
window — given, the model's profile, the gateway's listing (its own fields, or the llama.cpp
server's ``n_ctx`` it passes on), else the default said once — its
vendor, whether the gateway serves that vendor's own API, and the effort levels it takes."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from trellis.harness import models
from trellis.harness.models import DEFAULT_WINDOW, Window, window_of

GATEWAY = "http://gateway.test/v1"


@pytest.fixture(autouse=True)
def _fresh() -> Iterator[None]:
    """Each test asks the gateway again (the probes are cached per process)."""
    for cached in (models._listed, models.serves, models._assumed):
        cached.cache_clear()
    yield


def listing(*entries: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json={"object": "list", "data": list(entries)})


# --------------------------------------------------------------------------- the window
def test_a_given_window_wins_over_every_other_source() -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(f"{GATEWAY}/models").mock(
            return_value=listing({"id": "granite/g", "max_input_tokens": 4096})
        )
        assert window_of("granite/g", given=9000, profile=7000, base_url=GATEWAY) == Window(
            9000, "given"
        )
        assert window_of("granite/g", profile=7000, base_url=GATEWAY) == Window(7000, "profile")
        assert not route.called


@pytest.mark.parametrize(
    ("entry", "found"),
    [
        ({"id": "granite/g", "max_input_tokens": 4096}, Window(4096, "gateway")),
        ({"id": "granite/g", "context_length": 32768}, Window(32768, "gateway")),
        ({"id": "g", "context_window": 16384}, Window(16384, "gateway")),  # listed bare
        (
            {"id": "granite/g", "meta": {"n_ctx": 8192, "n_ctx_train": 131072}},
            Window(8192, "server"),
        ),
    ],
)
def test_the_gateways_listing_says_the_window(entry: dict[str, Any], found: Window) -> None:
    with respx.mock() as mock:
        mock.get(f"{GATEWAY}/models").mock(return_value=listing({"id": "other"}, entry))
        assert window_of("granite/g", base_url=GATEWAY, key="vk") == found
        assert mock.calls.last.request.headers["authorization"] == "Bearer vk"


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(200, json={"data": [{"id": "anthropic/claude-sonnet-5-5"}]}),  # no field
        httpx.Response(500),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.ConnectError("down"),
    ],
)
def test_a_listing_that_says_nothing_is_the_default(answer: Any) -> None:
    with respx.mock() as mock:
        route = mock.get(f"{GATEWAY}/models")
        if isinstance(answer, Exception):
            route.mock(side_effect=answer)
        else:
            route.mock(return_value=answer)
        found = window_of("anthropic/claude-sonnet-5-5", base_url=GATEWAY)
    assert found == Window(DEFAULT_WINDOW, "default")  # no copied table: the model says


def test_an_unknown_model_is_the_default_said_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="trellis.models"):
        assert window_of("local/unknown") == Window(DEFAULT_WINDOW, "default")
        assert window_of("local/unknown") == Window(DEFAULT_WINDOW, "default")
        assert window_of(None) == Window(DEFAULT_WINDOW, "default")
    said = [r.message for r in caplog.records if "local/unknown" in r.message]
    assert len(said) == 1 and "is not known" in said[0]


def test_the_gateway_is_asked_once_per_model() -> None:
    with respx.mock() as mock:
        route = mock.get(f"{GATEWAY}/models").mock(
            return_value=listing({"id": "granite/g", "max_input_tokens": 4096})
        )
        for _ in range(3):
            window_of("granite/g", base_url=GATEWAY)
        assert route.call_count == 1


# --------------------------------------------------------------------------- the vendor
@pytest.mark.parametrize(
    ("name", "vendor"),
    [
        ("anthropic/claude-sonnet-5-5", "anthropic"),
        ("openai/gpt-6-sol", "openai"),
        ("claude-opus-5-5", "anthropic"),
        ("gpt-4.1-nano", "openai"),
        ("o4-mini", "openai"),
        ("openrouter/openai/gpt-4.1-nano", None),  # another provider in front of it
        ("granite/granite-3.3-2b", None),
        ("mistral-large", None),
    ],
)
def test_the_vendor_is_the_provider_the_gateway_routes_to(name: str, vendor: str | None) -> None:
    assert models.vendor_of(name) == vendor


@pytest.mark.parametrize(
    ("vendor", "path", "status", "served"),
    [
        ("anthropic", "/anthropic/v1/messages", 400, True),
        ("anthropic", "/anthropic/v1/messages", 401, True),
        ("anthropic", "/anthropic/v1/messages", 404, False),
        ("openai", "/v1/responses", 422, True),
        ("openai", "/v1/responses", 405, False),
    ],
)
def test_the_gateway_serves_a_vendors_api_when_it_refuses_an_empty_request_as_invalid(
    vendor: Any, path: str, status: int, served: bool
) -> None:
    with respx.mock() as mock:
        route = mock.post(f"http://gateway.test{path}").mock(return_value=httpx.Response(status))
        assert models.serves(GATEWAY, "vk", vendor) is served
        assert route.calls.last.request.content == b"{}"


def test_an_unreachable_gateway_serves_nothing() -> None:
    with respx.mock() as mock:
        mock.post("http://gateway.test/v1/responses").mock(side_effect=httpx.ConnectError("x"))
        assert models.serves(GATEWAY, None, "openai") is False


# --------------------------------------------------------------------------- effort
def test_the_effort_levels_of_a_model_are_its_profiles() -> None:
    assert models.effort_levels(ChatAnthropic(model_name="claude-opus-5-5", api_key="x"))[:3] == (  # type: ignore[call-arg]
        "low",
        "medium",
        "high",
    )
    sonnet = ChatAnthropic(model_name="claude-sonnet-5-5", api_key="x")  # type: ignore[call-arg]
    listed = (sonnet.profile or {}).get("reasoning_effort_levels")
    assert models.effort_levels(sonnet) == tuple(listed or ())  # its profile, or none
    assert models.effort_levels(ChatOpenAI(model="granite/granite-3.3-2b", api_key="x")) == ()  # type: ignore[arg-type]
    assert models.effort_levels(object()) == ()


def test_a_level_the_model_does_not_take_is_the_nearest_it_does() -> None:
    levels = ("low", "medium", "high")
    assert models.nearest("high", levels) == "high"
    assert models.nearest("minimal", levels) == "low"
    assert models.nearest("max", levels) == "high"
    assert models.nearest("medium", ("low", "high")) == "low"  # the lower on a tie
