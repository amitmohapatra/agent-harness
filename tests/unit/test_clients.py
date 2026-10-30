"""The gateway and memory client modules: the only places the harness calls those services."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
import respx
from pydantic import SecretStr

from tests.support.memory import AGENT_TOOLS, FakeMemoryService
from trellis.contracts import Endpoint, ToolCall, ToolError, ToolOutcome
from trellis.harness import runtime as runtime_module
from trellis.harness.clients import bifrost
from trellis.harness.clients.bifrost import Gateway
from trellis.harness.clients.memory import READ_ONLY_TOOLS, Memory, Window
from trellis.harness.history import Called, Said, Speaker, recorded
from trellis.harness.identity import Identity, conversation_key

GATEWAY = "http://gw.test"


@respx.mock
async def test_the_gateway_lists_what_the_virtual_key_allows_by_execution_name() -> None:
    """With admin auth on, ``/api`` refuses a virtual key: the toolbox is the gateway's ``/mcp``
    ``tools/list`` asked with the key — Code Mode clients through their meta-tools."""
    admin = respx.get(f"{GATEWAY}/api/mcp/clients").mock(return_value=httpx.Response(401))

    def rpc(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "tools/list":
            tools = [
                {"name": "erp-get_stock", "annotations": {"readOnlyHint": True}},
                {"name": "listToolFiles", "annotations": {}},
                {"name": "executeToolCode", "annotations": {}},
            ]
            result: dict[str, object] = {"tools": tools}
        elif body["params"]["name"] == "listToolFiles":
            result = {"content": [{"type": "text", "text": "servers/\n  wiki.pyi"}]}
        else:
            text = "def read(repo: str) -> dict:  # Read a wiki.\n"
            result = {"content": [{"type": "text", "text": text}]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": result})

    listing = respx.post(f"{GATEWAY}/mcp").mock(side_effect=rpc)
    gateway = Gateway(f"{GATEWAY}/v1", "vk")
    stock, read = await gateway.tools()
    assert stock.name == "erp-get_stock" and not stock.code_mode
    assert stock.annotations is not None and stock.annotations.read_only_hint is True
    assert (read.name, read.client, read.code_mode) == ("wiki-read", "wiki", True)
    assert all(c.request.headers["authorization"] == "Bearer vk" for c in listing.calls)
    assert not admin.called
    await gateway.aclose()


@respx.mock
async def test_a_call_is_scoped_to_its_server_and_a_failed_tool_raises() -> None:
    route = respx.post(f"{GATEWAY}/v1/mcp/tool/execute").mock(
        side_effect=[
            httpx.Response(
                200, json={"role": "tool", "content": '{"units": 7}', "tool_call_id": "c"}
            ),
            httpx.Response(200, json={"role": "tool", "content": "no such sku", "is_error": True}),
        ]
    )
    gateway = Gateway(f"{GATEWAY}/v1", "vk")
    assert await gateway.execute(
        "erp-get_stock", {"sku": "a"}, clients=["erp"], parent_request_id="run_1"
    ) == {"units": 7}
    request = route.calls[0].request
    assert request.headers["x-bf-mcp-include-clients"] == "erp"
    assert request.headers["x-bf-parent-request-id"] == "run_1"
    assert json.loads(request.content)["function"] == {
        "name": "erp-get_stock",
        "arguments": '{"sku": "a"}',
    }
    with pytest.raises(ToolError, match="no such sku"):
        await gateway.execute("erp-get_stock", {"sku": "?"}, clients=["erp"])
    await gateway.aclose()


@respx.mock
async def test_in_a_run_a_request_waits_what_is_left_of_the_call_and_carries_its_key() -> None:
    executed = respx.post(f"{GATEWAY}/v1/mcp/tool/execute").mock(
        return_value=httpx.Response(200, json={"role": "tool", "content": "ok"})
    )
    message = {"role": "assistant", "content": "hi"}
    completed = respx.post(f"{GATEWAY}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": message}]})
    )
    gateway = Gateway(f"{GATEWAY}/v1", "vk")
    run = SimpleNamespace(
        run_id="run_1", idempotency_key="run_1:k:0", remaining=lambda: 12.5, tenant="t", user="u"
    )
    token = runtime_module._current.set(run)  # type: ignore[arg-type]
    try:
        assert await gateway.execute("erp-ship", {}, clients=["erp"]) == "ok"
        await gateway.complete([{"role": "user", "content": "hi"}], model="m")
    finally:
        runtime_module._current.reset(token)
    request = executed.calls[0].request
    assert json.loads(request.content)["id"] == "run_1:k:0"
    assert request.extensions["timeout"]["read"] == 12.5
    assert completed.calls[0].request.extensions["timeout"]["read"] == 12.5
    await gateway.aclose()


@respx.mock
async def test_code_mode_calls_are_read_back_from_the_log_by_the_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = respx.get(f"{GATEWAY}/api/mcp-logs").mock(
        return_value=httpx.Response(
            200,
            json={
                "logs": [
                    {
                        "id": "l1",
                        "timestamp": "2026-09-30T00:00:00Z",
                        "server_label": "wiki",
                        "tool_name": "search",
                        "status": "success",
                        "llm_request_id": "run_1",
                        "arguments": '{"q": "x"}',
                        "result": "found",
                        "latency": 12,
                    }
                ]
            },
        )
    )
    monkeypatch.setattr(bifrost, "LOG_POLL_SECONDS", 0.0)
    gateway = Gateway(f"{GATEWAY}/v1", None)
    [entry] = await gateway.code_mode_calls("run_1", datetime(2026, 9, 30, tzinfo=UTC))
    assert entry.name == "wiki-search" and entry.arguments == {"q": "x"}
    assert route.calls[0].request.url.params["llm_request_ids"] == "run_1"
    # the log is read again until two reads agree: it is written behind the calls
    assert route.call_count == 2
    await gateway.aclose()


async def test_a_scripts_calls_are_recorded_in_the_order_they_ran_at_the_scripts_step() -> None:
    """B17: a script's calls were recorded after the run's own, numbered by the memory service
    in the order the log came back: a tool the run called after the script came first in a
    learned procedure. Each is now recorded in log-time order at its script's step."""
    from trellis.harness.agent import Agent

    t0 = datetime(2026, 9, 30, tzinfo=UTC)

    def logged(id: str, at: float, name: str, q: str) -> SimpleNamespace:
        at_ = t0 + timedelta(seconds=at)
        return SimpleNamespace(
            id=id,
            timestamp=at_,
            name=name,
            arguments={"q": q},
            result="r",
            error=None,
            latency_ms=1,
        )

    class Log:
        async def code_mode_calls(self, run_id: str, since: datetime) -> list[SimpleNamespace]:
            return [
                logged("l3", 31, "wiki-read", "c"),
                logged("l2", 12, "wiki-read", "b"),
                logged("l1", 11, "wiki-search", "a"),
                logged("l0", 9.5, "wiki-list", "z"),  # the gateway's clock a little behind
            ]

    recorded: list[tuple[int | None, str]] = []

    class Records:
        async def record_tool(self, call: ToolCall, outcome: ToolOutcome) -> None:
            recorded.append((call.step, call.tool))

    class Writes:
        async def submit(self, name: str, work: Any, events: Any = None) -> None:
            await work()

    agent = SimpleNamespace(harness=SimpleNamespace(gateway=Log(), writes=Writes()))
    runtime = SimpleNamespace(
        run_memory=Records(),
        uses=lambda feature: True,
        run_id="run_1",
        started_at=t0,
        task="t",
        events=None,
        # two scripts: at step 1, and at step 4 after the run's own calls at steps 2 and 3
        code_mode_scripts=[(t0 + timedelta(seconds=10), 1), (t0 + timedelta(seconds=30), 4)],
    )
    await Agent.imported_code_mode_calls(cast("Agent", agent), cast("Any", runtime))
    assert recorded == [(1, "wiki-list"), (1, "wiki-search"), (1, "wiki-read"), (4, "wiki-read")]


async def test_identical_calls_of_one_script_are_each_sent_at_the_scripts_step() -> None:
    """Code Mode: a script that called one tool twice with the same arguments sends both, in
    log order (time, then id), at the script's step; the memory service records the pair as
    one call (idempotent on run, step, tool and arguments), so one call backs one approval."""
    from trellis.harness.agent import Agent

    t0 = datetime(2026, 9, 30, tzinfo=UTC)
    at = t0 + timedelta(seconds=5)

    def logged(id: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=id,
            timestamp=at,
            name="erp-get_stock",
            arguments={"sku": "ST-101"},
            result="7",
            error=None,
            latency_ms=1,
        )

    class Log:
        async def code_mode_calls(self, run_id: str, since: datetime) -> list[SimpleNamespace]:
            return [logged("l2"), logged("l1")]

    recorded: list[tuple[int | None, str, dict[str, Any]]] = []

    class Records:
        async def record_tool(self, call: ToolCall, outcome: ToolOutcome) -> None:
            recorded.append((call.step, call.tool, call.args))

    class Writes:
        async def submit(self, name: str, work: Any, events: Any = None) -> None:
            await work()

    agent = SimpleNamespace(harness=SimpleNamespace(gateway=Log(), writes=Writes()))
    runtime = SimpleNamespace(
        run_memory=Records(),
        uses=lambda feature: True,
        run_id="run_1",
        started_at=t0,
        task="t",
        events=None,
        code_mode_scripts=[(t0, 2)],
    )
    await Agent.imported_code_mode_calls(cast("Agent", agent), cast("Any", runtime))
    assert recorded == [(2, "erp-get_stock", {"sku": "ST-101"})] * 2


def test_a_call_logged_as_a_script_starts_is_that_scripts_and_one_before_any_the_firsts() -> None:
    """``script_step`` at its edges: a call logged at the instant a script started is that
    script's; one logged before every script (the gateway's clock behind) is the first's."""
    from trellis.harness.agent import script_step

    t0 = datetime(2026, 9, 30, tzinfo=UTC)
    scripts = [(t0, 1), (t0 + timedelta(seconds=10), 4)]
    assert script_step(scripts, t0 - timedelta(seconds=1)) == 1
    assert script_step(scripts, t0) == 1
    assert script_step(scripts, t0 + timedelta(seconds=9.999)) == 1
    assert script_step(scripts, t0 + timedelta(seconds=10)) == 4
    assert script_step([(t0, None)], t0) is None


# --------------------------------------------------------------------------- memory
def identity() -> Identity:
    return Identity(tenant="t", user="u", agent_id="a", run_id="run_1", thread="th")


def memory(service: FakeMemoryService) -> Memory:
    return Memory(service.client())


async def test_the_key_says_who_the_deployment_is() -> None:
    service = FakeMemoryService(tenant="acme")
    key = await memory(service).key()
    assert (key.tenant_id, key.principal, key.role) == ("acme", "svc:harness", "service")


async def test_a_run_memory_is_bound_to_the_run_scope() -> None:
    service = FakeMemoryService()
    run = memory(service).bind(identity())
    pushed = await run.context("q", tools=["erp-get_stock"], window=Window.NONE, hints=True)
    assert pushed.rendered.startswith(service.context_text)
    assert pushed.bundle_id is not None and pushed.bundle_id.startswith("bnd_")
    [call] = service.named("context")
    assert call.scope == {
        "tenant_id": "t",
        "user_id": "u",
        "agent_id": "a",
        "agent_run_id": "run_1",
        "thread_id": conversation_key("t", "u", "th"),  # u's conversation "th"
        "custom_metadata": {},
    }
    assert call.body["tools"] == {"available": ["erp-get_stock"], "k": 8}
    assert call.body["token_budget"] == 2000 and call.body["window"] is False
    assert "window_format" not in call.body  # the service's default


async def test_the_window_is_asked_for_in_the_form_the_framework_takes() -> None:
    service = FakeMemoryService()
    run = memory(service).bind(identity())
    await run.context("q", tools=None, window=Window.RENDERED, hints=True)
    await run.context("q", tools=None, window=Window.MESSAGES, hints=True)
    rendered, messages = service.named("context")
    assert rendered.body["window"] is True and "window_format" not in rendered.body
    assert messages.body["window"] is True and messages.body["window_format"] == "messages"
    assert service.violations == []


async def test_agent_tools_are_listed_once_and_tiered_read_or_write() -> None:
    service = FakeMemoryService()
    mem = memory(service)
    listed = await mem.bind(identity()).agent_tools()
    again = await mem.bind(identity()).agent_tools()
    assert [t.name for t in listed] == [t["name"] for t in AGENT_TOOLS] == [t.name for t in again]
    assert {t.name for t in listed if t.side_effects == "read"} <= READ_ONLY_TOOLS
    writes = {"memory_remember", "memory_update", "memory_forget", "profile_edit"}
    assert {t.name for t in listed if t.side_effects == "write"} == writes
    search = next(t for t in listed if t.name == "memory_search")
    assert (search.input_schema or {})["required"] == ["query"]  # the service's own schema
    assert len(service.named("agent_tools")) == 1
    # the SDK returns the tool's result itself
    found: Any = await mem.bind(identity()).call_agent_tool("memory_search", {"query": "x"})
    assert [item["kind"] for item in found] == ["memory", "chunk"] and found[0]["id"] == "mem_1"


async def test_records_carry_idempotency() -> None:
    service = FakeMemoryService()
    run = memory(service).bind(identity())
    await run.record_messages(
        [
            Said(Speaker.USER, "hi"),
            Said(Speaker.ASSISTANT, "", (Called("c1", "lookup", {"q": "hi"}),)),
            Said(Speaker.TOOL, "found", tool_call_id="c1"),
            Said(Speaker.ASSISTANT, "hello"),
        ],
        "run_1",
        2,
    )
    [batch] = service.named("messages")
    # each message names itself, so a re-recorded attempt stores it once
    assert [(m["role"], m["source_message_id"]) for m in batch.body["messages"]] == [
        ("USER", "run_1:user:0"),
        ("ASSISTANT", "run_1:2:msg:1"),
        ("TOOL", "run_1:2:msg:2"),
        ("ASSISTANT", "run_1:2:msg:3"),
    ]
    call, result = batch.body["messages"][1:3]
    assert call["tool_calls"] == [{"id": "c1", "name": "lookup", "arguments": {"q": "hi"}}]
    assert result["kind"] == "INTERNAL" and result["tool_call_id"] == "c1"
    assert service.violations == []
    assert {m["source_system"] for m in batch.body["messages"]} == {"trellis-harness"}
    await run.record_tool(
        ToolCall(tool="t", args={"a": 1}, task="q", step=1), ToolOutcome(tool="t", output=2)
    )
    assert service.named("record_tool")[0].body["status"] == "ok"
    await run.run_feedback("confirm", source="system", key="run_1:outcome")
    [feedback] = service.named("feedback")
    assert feedback.body["target_kind"] == "run" and feedback.body["target_id"] == "run_1"
    assert feedback.body["source"] == "system" and feedback.idempotency_key == "run_1:outcome"


async def test_the_model_endpoint_is_registered_in_the_agents_scope() -> None:
    """The whole endpoint, its key in plain (``SecretStr`` masks it); a key rotated or an
    endpoint moved is a new request, the same one repeated is not."""
    service = FakeMemoryService()
    scoped = memory(service).scoped("t", "a")
    endpoint = Endpoint(base_url="http://llm.test/v1", api_key=SecretStr("sk"))
    await scoped.register_endpoint(endpoint)
    await scoped.register_endpoint(endpoint)
    await scoped.register_endpoint(endpoint.model_copy(update={"api_key": SecretStr("sk-2")}))
    await scoped.register_endpoint(Endpoint(base_url="https://login.test/v1", auth="google"))
    first, again, rotated, login = service.named("model_key")
    assert first.body["endpoint"]["api_key"] == "sk" and login.body["endpoint"]["api_key"] is None
    assert first.idempotency_key is not None and first.idempotency_key.startswith("model-key:a:")
    assert first.idempotency_key == again.idempotency_key != rotated.idempotency_key
    assert first.scope == {"tenant_id": "t", "agent_id": "a", "custom_metadata": {}}


async def test_an_answer_that_holds_no_endpoint_is_sent_again_as_a_new_request() -> None:
    """A replay of an earlier registration the service has since dropped (its migration
    deleted the bare keys): the endpoint is registered again, under a fresh key."""
    service = FakeMemoryService(stale_registrations=1)
    await memory(service).scoped("t", "a").register_endpoint(Endpoint(base_url="http://l/v1"))
    replayed, again = service.named("model_key")
    assert replayed.idempotency_key is not None and again.idempotency_key is not None
    assert again.idempotency_key.startswith(replayed.idempotency_key + ":")
    assert again.body == replayed.body


@respx.mock
async def test_a_tool_result_that_is_not_json_is_returned_as_it_came() -> None:
    respx.post(f"{GATEWAY}/v1/mcp/tool/execute").mock(
        side_effect=[
            httpx.Response(200, json={"role": "tool", "content": "plain words"}),
            httpx.Response(200, json={"role": "tool", "content": [{"type": "text", "text": "x"}]}),
        ]
    )
    gateway = Gateway(f"{GATEWAY}/v1", "vk")
    assert await gateway.execute("wiki-read", {}, clients=["wiki"]) == "plain words"
    assert await gateway.execute("wiki-read", {}, clients=["wiki"]) == [
        {"type": "text", "text": "x"}
    ]
    await gateway.aclose()


async def test_a_document_can_be_added_without_waiting_for_it() -> None:
    service = FakeMemoryService()
    info = await memory(service).for_user("t", "u").add_document(b"text", wait=None)
    assert info.status == "STAGED"  # its parse job has not run yet
    assert [c.name for c in service.named("document")] == ["document"]  # read once, not polled


async def test_a_spooled_memory_write_replays_in_its_own_scope() -> None:
    """Each write the harness spools (its ``record``) is the same request again, in the scope
    it was made in, when an empty process replays it."""
    from trellis.contracts import (
        AgentExecutionContext,
        Interrupt,
        InterruptDecision,
        InterruptReason,
        InterruptResolution,
    )

    service = FakeMemoryService()
    memory = Memory(service.client())
    scope = Identity(tenant="acme", user="ada", agent_id="a", run_id="run_1", thread="thr")
    run = memory.bind(scope)
    call, outcome = ToolCall(tool="erp-get", args={"sku": "1"}), ToolOutcome(tool="erp-get")
    asked = Interrupt(
        interrupt_id="run_1.1.1",
        tenant_id="acme",
        run_id="run_1",
        question="Refund?",
        reason=InterruptReason.APPROVAL,
        tool_call=ToolCall(tool="refund", args={"amount": 5}),
    )
    decision = InterruptResolution(
        interrupt_id="run_1.1.1", run_id="run_1", decision=InterruptDecision.APPROVE, reviewer="cfo"
    ).to_feedback(
        asked, AgentExecutionContext.create(tenant_id="acme", agent_id="a", agent_run_id="run_1")
    )
    assert decision is not None
    records = [
        # as spooled before tool calls were kept, and as spooled now
        run.record("record_messages", messages=[["user", "hi"]], run_id="run_1", attempt=1),
        run.record(
            "record_messages",
            messages=[
                recorded(Said(Speaker.ASSISTANT, "", (Called("c1", "erp-get", {"sku": "1"}),))),
                recorded(Said(Speaker.TOOL, "7", tool_call_id="c1")),
            ],
            run_id="run_1",
            attempt=1,
        ),
        run.record(
            "record_tool",
            call=call.model_dump(mode="json"),
            outcome=outcome.model_dump(mode="json"),
        ),
        run.record(
            "run_feedback", verdict="confirm", source="system", comment=None, key="run_1:outcome"
        ),
        run.record("feedback", record=decision.model_dump(mode="json")),
        run.record(
            "publish_catalog", entries=[{"name": "erp-get", "description": "", "source": "local"}]
        ),
    ]
    for record in json.loads(json.dumps(records)):  # through the spool's JSON and back
        work = memory.replay(record)
        assert work is not None
        await work()
    assert [c.name for c in service.calls] == [
        "messages",
        "messages",
        "record_tool",
        "feedback",
        "feedback",
        "put_catalog",
    ]
    assert all(c.scope.get("tenant_id") == "acme" for c in service.calls)
    assert service.named("messages")[0].body["messages"][0]["source_message_id"] == "run_1:user:0"
    call_message, result = service.named("messages")[1].body["messages"]
    assert call_message["tool_calls"] == [
        {"id": "c1", "name": "erp-get", "arguments": {"sku": "1"}}
    ]
    assert result["tool_call_id"] == "c1" and result["kind"] == "INTERNAL"
    assert memory.replay({"op": "unknown", "scope": {}, "args": {}}) is None
    await memory.client.aclose()
