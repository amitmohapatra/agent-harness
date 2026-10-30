"""Stored prompts: a resolved prompt rendered and read as messages, what a call selects it by
and a span says, the journal's copy, ``stored()``, and the gateway's Prompt Repository."""

from __future__ import annotations

import httpx
import pytest
import respx

from tests.support.gateway import FakeGateway
from trellis import stored
from trellis.contracts import ConfigurationError
from trellis.harness.clients.bifrost import Gateway, PromptPin
from trellis.harness.prompts import Prompts, Rendered, ResolvedPrompt, Stored
from trellis.harness.repository import NotFound

SYSTEM = [{"role": "system", "content": "You triage."}]


def test_a_prompt_fills_its_messages() -> None:
    chat = (
        {"role": "system", "content": "You help {{ user }}.", "type": "chatmessage"},
        {"role": "user", "content": [{"type": "text", "text": "Hi {{user}}"}, {"type": "image"}]},
        {"role": "assistant", "content": None},
    )
    found = ResolvedPrompt("help", "1", "Bifrost", chat=chat)
    assert found.ref == "help@1" and found.variables == ["user"]
    assert found.messages(user="Ada", unused=1) == [
        {"role": "system", "content": "You help Ada."},
        {"role": "user", "content": [{"type": "text", "text": "Hi Ada"}, {"type": "image"}]},
        {"role": "assistant", "content": None},
    ]
    assert found.render(user="Ada") == "You help Ada.\n\nHi Ada"
    with pytest.raises(ConfigurationError, match=r"help@1 needs user: pass them"):
        found.messages()
    assert ResolvedPrompt("none", "1", "Bifrost").variables == []


def test_a_prompt_says_how_a_call_selects_it_and_what_a_span_says() -> None:
    found = ResolvedPrompt("triage", "2", "Bifrost", chat=tuple(SYSTEM), selection="p-triage")
    assert found.pin() == PromptPin("triage", "p-triage", 2)
    assert found.attributes() == {
        "trellis.prompt.name": "triage",
        "trellis.prompt.id": "p-triage",
        "trellis.prompt.version": 2,
        "trellis.prompt.source": "Bifrost",
    }
    unselected = ResolvedPrompt("greet", "1", "Bifrost")
    assert unselected.pin() is None
    assert unselected.attributes() == {
        "trellis.prompt.name": "greet",
        "trellis.prompt.version": "1",
        "trellis.prompt.source": "Bifrost",
    }


def test_a_prompt_comes_back_from_the_journal_as_it_was() -> None:
    found = ResolvedPrompt("b", "2", "Bifrost", chat=tuple(SYSTEM), selection="p-b")
    assert ResolvedPrompt.of_record(found.record()) == found
    # an earlier harness journaled a ReAct's stored prompt as its selection only
    old = ResolvedPrompt.of_record({"name": "triage", "id": "p-triage", "version": 2})
    assert old == ResolvedPrompt("triage", "2", "Bifrost", selection="p-triage")


def test_stored_names_a_prompt_and_its_values() -> None:
    assert stored("triage@3", team="EU") == Stored("triage@3", {"team": "EU"})
    with pytest.raises(ConfigurationError, match="is not a name"):
        stored("triage@")


async def test_the_gateways_prompts_by_number_rendered_and_as_messages() -> None:
    older = [{"role": "system", "content": "Old {{team}}."}]
    fake = FakeGateway(prompts={"triage": [older, SYSTEM]})
    prompts = Prompts(fake.gateway())
    assert await prompts.rendered("triage") == Rendered("You triage.", SYSTEM)
    assert (await prompts.rendered(stored("triage@1", team="EU"))).text == "Old EU."
    assert (await prompts.rendered("triage@1", team="US")).messages[0]["content"] == "Old US."
    assert fake.asked("/api/prompt-repo/prompts/p-triage/versions") == 1
    for version in ("staging", "0"):
        with pytest.raises(NotFound, match="a stored prompt's version is a number from 1"):
            await prompts.get(f"triage@{version}")
    with pytest.raises(NotFound, match="no committed prompt named 'ghost'"):
        await prompts.get("ghost")
    assert prompts.gateway is not None
    await prompts.gateway.aclose()


async def test_a_stored_prompt_needs_the_gateway() -> None:
    with pytest.raises(
        ConfigurationError, match="a stored prompt of the gateway: set LLM_GATEWAY=bifrost"
    ):
        await Prompts(None).get("triage@2")


@respx.mock
async def test_a_version_the_gateway_does_not_list_is_not_found() -> None:
    gw = "http://gw.test"
    latest = {"id": 9, "prompt_id": "p-1", "version_number": 3, "messages": []}
    respx.get(f"{gw}/api/prompt-repo/prompts").mock(
        return_value=httpx.Response(
            200, json={"prompts": [{"id": "p-1", "name": "triage", "latest_version": latest}]}
        )
    )
    respx.get(f"{gw}/api/prompt-repo/prompts/p-1/versions").mock(
        return_value=httpx.Response(200, json={"versions": [latest]})
    )
    gateway = Gateway(f"{gw}/v1", "vk")
    with pytest.raises(NotFound, match="has no version 2"):
        await gateway.prompt("triage@2")
    await gateway.aclose()
