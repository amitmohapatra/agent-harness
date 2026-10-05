"""The gateway client's Bifrost features: stored prompts, skills, Virtual MCPs, who a tool call
is for, the Agent Mode lists, and completions that never carry the gateway's MCP tools."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
import respx

from tests.support.gateway import FakeGateway, SkillVersions
from trellis.contracts import ConfigurationError
from trellis.harness import fresh
from trellis.harness.clients import bifrost
from trellis.harness.clients.bifrost import Gateway, PromptPin, pinned, prompt_ref
from trellis.harness.identity import IDENTITY_HEADER
from trellis.harness.runtime import _current

GATEWAY = "http://gw.test"
SYSTEM = [{"role": "system", "content": "You triage."}]


@pytest.mark.parametrize(
    ("ref", "expected"),
    [("triage", ("triage", None)), ("triage@3", ("triage", 3))],
)
def test_a_prompt_is_named_and_maybe_pinned(ref: str, expected: tuple[str, int | None]) -> None:
    assert prompt_ref(ref) == expected


@pytest.mark.parametrize("ref", ["", "@3", "triage@", "triage@x", "triage@0"])
def test_a_prompt_reference_that_names_no_version_number_is_refused(ref: str) -> None:
    with pytest.raises(ConfigurationError):
        prompt_ref(ref)


def test_a_skill_is_named_and_maybe_pinned() -> None:
    assert pinned("refunds@1.2.0") == ("refunds", "1.2.0")
    assert pinned("refunds") == ("refunds", None)


async def test_a_prompt_resolves_to_its_id_and_latest_version_once_then_after_its_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(fresh, "_now", lambda: clock[0])
    fake = FakeGateway(prompts={"triage": [SYSTEM, SYSTEM]})
    gateway = fake.gateway()
    pin = await gateway.prompt("triage")
    assert pin == PromptPin(name="triage", id="p-triage", version=2)
    assert await gateway.prompt("triage@1") == PromptPin(name="triage", id="p-triage", version=1)
    await gateway.prompt("triage")
    assert fake.asked("/api/prompt-repo/prompts") == 2  # one read per reference
    fake.prompts["triage"].append(SYSTEM)
    clock[0] += bifrost.REPOSITORY_TTL_SECONDS + 1
    assert (await gateway.prompt("triage")).version == 3
    assert pin.attributes() == {
        "trellis.prompt.name": "triage",
        "trellis.prompt.id": "p-triage",
        "trellis.prompt.version": 2,
    }
    await gateway.aclose()


@pytest.mark.parametrize(
    ("prompts", "ref", "problem"),
    [
        ({}, "triage", "no committed prompt named 'triage'"),
        ({"triage": []}, "triage", "no committed prompt"),
        ({"triage": [SYSTEM]}, "triage@2", "has no version 2"),
    ],
)
async def test_a_prompt_that_is_not_there_is_a_configuration_error(
    prompts: dict, ref: str, problem: str
) -> None:
    gateway = FakeGateway(prompts=prompts).gateway()
    with pytest.raises(ConfigurationError, match=problem):
        await gateway.prompt(ref)
    await gateway.aclose()


@respx.mock
async def test_two_prompts_of_one_name_are_a_configuration_error() -> None:
    twice = [{"id": f"p-{n}", "name": "triage"} for n in (1, 2)]
    respx.get(f"{GATEWAY}/api/prompt-repo/prompts").mock(
        return_value=httpx.Response(200, json={"prompts": twice})
    )
    gateway = Gateway(f"{GATEWAY}/v1", "vk")
    with pytest.raises(ConfigurationError, match="2 prompts are named"):
        await gateway.prompt("triage")
    await gateway.aclose()


async def test_a_completion_selects_the_prompt_and_never_the_gateways_mcp_tools() -> None:
    fake = FakeGateway(turns=["ok", "ok"])
    gateway = fake.gateway()
    await gateway.complete([{"role": "user", "content": "hi"}], model="m")
    pin = PromptPin(name="triage", id="p-triage", version=2)
    await gateway.complete([{"role": "user", "content": "hi"}], model="m", prompt=pin)
    plain, prompted = (r.headers for r in fake.completions)
    for headers in (plain, prompted):  # the deny-all scope: no tool added, none run
        assert headers["x-bf-mcp-include-clients"] == ""
        assert headers["x-bf-mcp-include-tools"] == ""
    assert "x-bf-prompt-id" not in plain
    assert (prompted["x-bf-prompt-id"], prompted["x-bf-prompt-version"]) == ("p-triage", "2")
    await gateway.aclose()


async def test_a_skill_reads_as_served_or_as_a_version_kept_for_good() -> None:
    fake = FakeGateway(
        skills={
            "sql": SkillVersions(
                {"1.0.0": ("Old.", "v1", {}), "1.1.0": ("Reviews SQL.", "v2", {"a.md": "A"})},
                served="1.1.0",
            )
        }
    )
    gateway = fake.gateway()
    served = await gateway.skill("sql")
    assert (served.version, served.body, [f.path for f in served.files]) == (
        "1.1.0",
        "v2",
        ["a.md"],
    )
    old = await gateway.skill("sql", "1.0.0")
    assert (old.version, old.description) == ("1.0.0", "Old.")
    asked = fake.asked("/api/skills/s-sql")
    assert await gateway.skill("sql", "1.0.0") == old
    assert fake.asked("/api/skills/s-sql") == asked  # a published version never changes
    assert await gateway.skill("sql") == served  # kept: read again after its TTL
    assert await gateway.served("sql") == "1.1.0"  # read now, every time
    assert await gateway.served("nope") is None
    assert await gateway.skill_file("sql", "a.md") == b"A"
    with pytest.raises(LookupError, match="no skill named 'nope'"):
        await gateway.skill("nope")
    await gateway.aclose()


async def test_the_agent_mode_lists_of_the_mcp_clients() -> None:
    gateway = FakeGateway(auto={"erp": ["pay"], "crm": []}).gateway()
    assert await gateway.auto_executed() == {"erp": frozenset({"pay"}), "crm": frozenset()}
    await gateway.aclose()


@respx.mock
async def test_a_call_in_a_run_says_who_it_is_for_and_runs_through_its_virtual_mcp() -> None:
    def rpc(request: httpx.Request) -> httpx.Response:
        result = {"content": [{"type": "text", "text": "ada"}]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": result})

    bundle = respx.post(f"{GATEWAY}/mcp/finance").mock(side_effect=rpc)
    executed = respx.post(f"{GATEWAY}/v1/mcp/tool/execute").mock(
        return_value=httpx.Response(200, json={"role": "tool", "content": "ok"})
    )
    gateway = Gateway(f"{GATEWAY}/v1", "vk")
    run = SimpleNamespace(
        run_id="run_1", idempotency_key="k", remaining=lambda: None, tenant="acme", user="ada"
    )
    token = _current.set(run)  # type: ignore[arg-type]
    try:
        assert await gateway.execute("ops-whoami", {}, clients=["ops"], slug="finance") == "ada"
        assert await gateway.execute("ops-whoami", {}, clients=["ops"]) == "ok"
    finally:
        _current.reset(token)
    through, direct = bundle.calls[0].request, executed.calls[0].request
    assert json.loads(through.content)["params"] == {"name": "ops-whoami", "arguments": {}}
    assert "x-bf-mcp-include-clients" not in through.headers  # the bundle is the scope
    assert direct.headers["x-bf-mcp-include-clients"] == "ops"
    for request in (through, direct):
        assert json.loads(request.headers[IDENTITY_HEADER]) == {
            "tenant_id": "acme",
            "user_id": "ada",
        }
        assert request.headers["x-bf-mcp-session-id"] == "acme:ada"
    await gateway.aclose()
