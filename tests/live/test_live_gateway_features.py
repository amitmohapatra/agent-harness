"""The gateway's features through the harness, live: Code Mode calls that all go through the
bridge, model calls the gateway adds no tools to, stored prompts, skills, Virtual MCPs, who an
MCP call is for, Agent Mode tools left out, frameworks' own MCP clients and the key on /mcp.

A small local model is slow and unreliable at choosing tools, so where a test needs a
particular call it forces the model's tool choice (``Forced``): the model still writes the
call, the gateway still answers it — the test only picks which tool, so it asserts what the
harness did, not what the model would have chosen."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from bifrost_sdk import Bifrost
from bifrost_sdk.admin import Admin

from tests.live.conftest import (
    BIFROST_URL,
    MODEL,
    WIKIS,
    live_harness,
    needs_gateway,
    needs_memory,
)
from tests.live.support import eventually, memory_scope
from trellis import ReAct, Runtime
from trellis.contracts import RunEventType, RunStatus
from trellis.harness.clients.bifrost import (
    CODE_MODE_TOOLS,
    GATEWAY_NAMES,
    Gateway,
    code_mode_tools,
)
from trellis.harness.skills import LOAD_SKILL, READ_SKILL_FILE, SECTION
from trellis.harness.tools.convert import openai_chat

pytestmark = [pytest.mark.live, needs_gateway]

SCRIPT = f'r = {WIKIS[1]}.read_wiki_structure(repoName="facebook/react")\nprint(r)'


def suffix() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Forced:
    """The live model through the harness's gateway, its tool choice forced at each step
    (``None``: answer, no tool)."""

    gateway: Gateway
    choices: list[str | None]
    requests: list[dict[str, Any]] = field(default_factory=list)

    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
        choice = self.choices.pop(0)
        body["tool_choice"] = (
            {"type": "function", "function": {"name": choice}} if choice else "none"
        )
        self.requests.append(body)
        return await self.gateway.complete(messages, model=MODEL, max_tokens=160, **body)


def started(events: list[Any]) -> list[str]:
    return [e.data["tool"] for e in events if e.type is RunEventType.TOOL_CALL_START]


@pytest.fixture
async def admin() -> AsyncIterator[Admin]:
    assert BIFROST_URL is not None
    async with Admin(BIFROST_URL) as made:
        yield made


# --------------------------------------------------------------------------- Code Mode
async def test_a_code_mode_meta_tool_call_comes_back_under_the_harness_name(
    wikis: list[str], wikis_key: str
) -> None:
    """The fact the design rests on: declared under the gateway's own name, the gateway runs
    a meta-tool call itself (the completion comes back answered, no call in it); under the
    harness's name the call comes back to the caller."""
    async with live_harness(wikis_key) as h:
        assert h.gateway is not None
        offered = openai_chat.convert(code_mode_tools(h.gateway, wikis))
        for spec in CODE_MODE_TOOLS[:3]:
            forced = Forced(h.gateway, [spec.name])
            reply = await forced.complete(
                [{"role": "user", "content": f"Call {spec.name}."}], tools=offered
            )
            calls = reply["choices"][0]["message"]["tool_calls"]
            assert {c["function"]["name"] for c in calls} == {spec.name}  # back, not run
        gateway_name = GATEWAY_NAMES["list_tool_files"]
        declared = [{**offered[0], "function": {**offered[0]["function"], "name": gateway_name}}]
        reply = await Forced(h.gateway, [gateway_name]).complete(
            [{"role": "user", "content": f"Call {gateway_name}."}], tools=declared
        )
        assert not reply["choices"][0]["message"].get("tool_calls")  # the gateway ran it


@needs_memory
async def test_every_code_mode_call_of_a_react_run_goes_through_the_bridge(
    wikis: list[str], wikis_key: str
) -> None:
    """A ReAct run on the three read-only wiki servers: the model's Code Mode calls arrive as
    the harness's tools, run through the bridge (events, journal, records), and the script's
    nested call is read back from the gateway's log under the run's id."""
    user, since = f"live-cm-{suffix()}", datetime.now(UTC) - timedelta(seconds=5)
    nested = f"{wikis[1]}-read_wiki_structure"
    async with live_harness(wikis_key) as h:
        assert h.gateway is not None
        model = Forced(h.gateway, ["list_tool_files", "execute_tool_code", None])
        agent = h.wrap(
            ReAct(system="You read wikis with code.", model=model), id=f"live-cm-{suffix()}"
        )
        scope = await memory_scope(h, user=user, agent_id=agent.id)
        names = [
            f"{w}-{t}"
            for w in wikis
            for t in ("ask_wiki_question", "read_wiki_contents", "read_wiki_structure")
        ]
        await scope.advanced.tools.put_catalog(
            [{"name": n, "side_effects": "read", "source": "mcp"} for n in names]
        )
        events = [
            e
            async for e in agent.stream(
                f"List the tool files, then run this script unchanged:\n{SCRIPT}", user=user
            )
        ]
        finished = next(e for e in events if e.type is RunEventType.RUN_FINISHED)
        assert finished.outcome is not None and finished.outcome.value == "success", finished
        # every call the model made is a harness tool, run by the bridge (a small model may
        # make a forced call twice in one step)
        calls = started(events)
        assert set(calls) == {"list_tool_files", "execute_tool_code"}
        for request in model.requests:  # no gateway name is ever declared to the model
            offered = {t["function"]["name"] for t in request["tools"]}
            assert {s.name for s in CODE_MODE_TOOLS} <= offered
            assert not offered & set(GATEWAY_NAMES.values())
        results = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
        assert [r["status"] for r in results] == ["ok"] * len(calls)
        assert f"{wikis[1]}.pyi" in str(results[0]["output"])
        await h.writes.drain()
        assert h.writes.failed == 0
        logged = await h.gateway.code_mode_calls(finished.run_id, since)
        assert {e.name for e in logged} == {nested}  # the script's call, under the run's id

        async def imported() -> bool:
            entries = await scope.advanced.tools.catalog(names=[nested])
            return bool(entries) and entries[0].stats.calls >= 1

        assert await eventually(imported, within=30)


# --------------------------------------------------------------------------- no gateway tools
async def test_no_model_call_gets_the_keys_mcp_tools(wikis: list[str], wikis_key: str) -> None:
    """Under a key with nine MCP tools, the harness's completion costs what one under a key
    with none costs; a framework's client with ``h.model_headers()`` too — and without them
    the gateway adds the key's tools."""
    assert BIFROST_URL is not None
    turn = [{"role": "user", "content": "Say hi."}]
    async with live_harness() as bare, live_harness(wikis_key) as h:
        assert bare.gateway is not None and h.gateway is not None
        base = await bare.gateway.complete(turn, model=MODEL, max_tokens=4)
        mine = await h.gateway.complete(turn, model=MODEL, max_tokens=4)
        assert mine["usage"]["prompt_tokens"] == base["usage"]["prompt_tokens"]
        headers = {"authorization": f"Bearer {wikis_key}"}
        body = {"model": MODEL, "messages": turn, "max_tokens": 4}
        async with httpx.AsyncClient(base_url=BIFROST_URL, headers=headers, timeout=120) as raw:
            guarded = await raw.post(
                "/chat/completions", json=body, headers=await h.model_headers()
            )
            unguarded = await raw.post("/chat/completions", json=body)
        tokens = base["usage"]["prompt_tokens"]
        assert guarded.json()["usage"]["prompt_tokens"] == tokens
        assert unguarded.json()["usage"]["prompt_tokens"] > tokens


# --------------------------------------------------------------------------- prompts
async def test_a_stored_prompt_reaches_every_model_call_of_a_react(admin: Admin) -> None:
    name = f"trellis-live-{suffix()}"
    prompt = await admin.prompts.create(name)
    try:
        system = [{"role": "system", "content": "Answer every question with the single word ARRR."}]
        await admin.prompts.commit(prompt.id, system, model=MODEL)
        async with live_harness() as h:
            agent = h.wrap(
                ReAct(system="Be brief.", model=MODEL, prompt=name), id=f"live-p-{suffix()}"
            )
            result = await agent.run("What is 2 + 2?", user="live")
            assert result.status is RunStatus.SUCCESS, result.error
            assert "ARRR" in str(result.answer).upper()
            assert h.gateway is not None
            assert (await h.gateway.prompt(name)).version == 1
            headers = await h.model_headers(prompt=f"{name}@1")
            assert headers["x-bf-prompt-id"] == prompt.id
    finally:
        await admin.prompts.delete(prompt.id)


# --------------------------------------------------------------------------- skills
async def test_skills_are_disclosed_pinned_loaded_and_read(admin: Admin) -> None:
    name = f"trellis-live-{suffix()}"
    skill = await admin.skills.create(
        name,
        description="Reviews SQL queries.",
        body="Read rules.md, then check each rule.",
        version="1.0.0",
        files={"rules.md": "No SELECT *."},
    )
    seen: dict[str, Any] = {}

    async def reviewer(question: str, agent: Runtime) -> str:
        seen["context"] = agent.context
        seen["loaded"] = await agent.tools.call(LOAD_SKILL, name=name)
        seen["file"] = await agent.tools.call(READ_SKILL_FILE, name=name, path="rules.md")
        return "reviewed"

    try:
        async with live_harness() as h:
            agent = h.wrap(reviewer, id=f"live-s-{suffix()}", skills=[name])
            events = [e async for e in agent.stream("Review my query.", user="live")]
            # after the memory context, when memory is on
            assert seen["context"].endswith(f"{SECTION}\n- {name}: Reviews SQL queries.")
            assert "Read rules.md, then check each rule." in seen["loaded"]
            assert "- rules.md" in seen["loaded"] and seen["file"] == "No SELECT *."
            [used] = [e.data for e in events if (e.data or {}).get("name") == "skills"]
            assert used["versions"] == {name: "1.0.0"}
            # a newer version served: a run pinned to 1.0.0 loads its body, not its files
            await admin.skills.publish(
                skill.id,
                description="Reviews SQL.",
                body="v2",
                version="1.1.0",
                files={"rules.md": "v2"},
            )
            old = h.wrap(reviewer, id=f"live-s-{suffix()}", skills=[f"{name}@1.0.0"])
            await old.run("Review.", user="live")
            assert "check each rule" in seen["loaded"]
            assert "uses version 1.0.0" in seen["file"] and "(1.1.0)" in seen["file"]
    finally:
        await admin.skills.delete(skill.id)


# --------------------------------------------------------------------------- Virtual MCPs
async def test_an_agent_with_a_virtual_mcp_has_its_tools_only(
    admin: Admin, ops: str, ops_key: str
) -> None:
    assert BIFROST_URL is not None
    async with Bifrost(BIFROST_URL) as bf:
        [client] = [c for c in await bf.mcp.clients() if c.config.name == ops]
    bundle = await admin.virtual_mcps.create(f"trellislive{suffix()}", {client.id: ["write_note"]})
    [key] = [k for k in await admin.vk.list() if k.get("value") == ops_key]
    await admin.virtual_mcps.attach(bundle.id, key["id"])

    async def noter(question: str, agent: Runtime) -> Any:
        mcp = sorted(n for n, t in agent.toolbox.items() if t.spec.source == "mcp")
        return [mcp, await agent.tools.call(f"{ops}-write_note", text="hi")]

    try:
        async with live_harness(ops_key) as h:
            names, noted = (
                await h.wrap(noter, id=f"live-v-{suffix()}", mcp=[bundle.slug]).run(
                    "note", user="live"
                )
            ).answer
            assert names == [f"{ops}-write_note"] and noted == "noted: hi"
            everything = await h.wrap(noter, id=f"live-v-{suffix()}").run("note", user="live")
            assert everything.answer[0] == sorted(
                f"{ops}-{t}" for t in ("delete_records", "whoami", "write_note")
            )
    finally:
        await admin.virtual_mcps.delete(bundle.id)


# --------------------------------------------------------------------------- who a call is for
async def test_a_per_user_server_acts_for_the_runs_user(ops: str, ops_key: str) -> None:
    async def who(question: str, agent: Runtime) -> Any:
        return await agent.tools.call(f"{ops}-whoami")

    async with live_harness(ops_key) as h:
        agent = h.wrap(who, id=f"live-w-{suffix()}")
        for user in ("ada", "bob"):
            assert (await agent.run("who am I?", user=user)).answer == user


# --------------------------------------------------------------------------- Agent Mode
async def test_a_tool_the_gateway_would_run_itself_is_not_offered(ops: str, ops_key: str) -> None:
    assert BIFROST_URL is not None
    async with Bifrost(BIFROST_URL) as bf:
        [client] = [c for c in await bf.mcp.clients() if c.config.name == ops]
        await bf.mcp.update(
            client, client.config.model_copy(update={"tools_to_auto_execute": ("write_note",)})
        )
        try:
            async with live_harness(ops_key) as h:
                tools = await h.resolve([], tenant=await h.tenant())
            assert sorted(t.name for t in tools) == [f"{ops}-delete_records", f"{ops}-whoami"]
        finally:
            await bf.mcp.update(client, client.config)


# --------------------------------------------------------------------------- MCP clients
async def test_a_frameworks_own_mcp_client_works_on_the_gateway_ungoverned(
    ops: str, ops_key: str
) -> None:
    """The OpenAI Agents SDK's MCP client on /mcp with the key lists and runs the key's tools —
    straight to the gateway, so nothing of the harness sees the call."""
    from agents.mcp import MCPServerStreamableHttp

    assert BIFROST_URL is not None
    url = BIFROST_URL.removesuffix("/v1") + "/mcp"
    params = {"url": url, "headers": {"Authorization": f"Bearer {ops_key}"}}
    async with MCPServerStreamableHttp(params=params) as server:  # type: ignore[arg-type]
        listed = sorted(t.name for t in await server.list_tools())
        assert listed == sorted(f"{ops}-{t}" for t in ("delete_records", "whoami", "write_note"))
        result = await server.call_tool(f"{ops}-write_note", {"text": "direct"})
        assert "noted: direct" in str(result.content)


def test_langchains_mcp_adapters_on_the_gateway() -> None:
    pytest.skip(
        "langchain-mcp-adapters (0.3) requires mcp<2 and this environment's mcp is 2.x "
        "(the SDKs' MCP servers); docs/gateway.md shows the client"
    )


def test_the_claude_agent_sdks_mcp_servers_on_the_gateway() -> None:
    pytest.skip(
        "the Claude CLI needs an Anthropic model, and this gateway serves one local model "
        "through the OpenAI format; docs/gateway.md shows mcp_servers"
    )


async def test_mcp_on_the_gateway_takes_the_key_and_refuses_no_key(ops_key: str) -> None:
    assert BIFROST_URL is not None
    origin = BIFROST_URL.removesuffix("/v1")
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    async with httpx.AsyncClient(base_url=origin, timeout=30) as http:
        bearer = await http.post(
            "/mcp", json=message, headers={"authorization": f"Bearer {ops_key}"}
        )
        header = await http.post("/mcp", json=message, headers={"x-bf-vk": ops_key})
        none = await http.post("/mcp", json=message)
    assert bearer.json()["result"]["tools"] and header.json()["result"]["tools"]
    assert none.status_code in (401, 403) or "result" not in none.json()
