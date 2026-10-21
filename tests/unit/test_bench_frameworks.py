"""The framework head-to-head (``benchmark/cost/frameworks``) without a model: the metering
proxy counts and normalizes what it passes on, the cells plan their runs, the native, wrapped
and ReAct arms run whole cells against a scripted OpenAI-compatible model behind the proxy
(every row's success, coverage and tokens), and the report ranks the arms and gives its
verdict."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from benchmark.cost import catalog
from benchmark.cost.frameworks import __main__ as cli
from benchmark.cost.frameworks import arms as built
from benchmark.cost.frameworks import meter, report, runner, survive
from trellis import tool
from trellis.harness.tools.sources import as_source


# --------------------------------------------------------------------------- a scripted model
def _needed(text: str) -> list[str]:
    for taskset in catalog.task_sets():
        for task in taskset.tasks:
            if task["input"] == text:
                return list(task["metadata"]["needed_tools"])
    return []


def _sample(schema: dict[str, Any]) -> dict[str, Any]:
    values = {"string": "S12", "integer": 2, "number": 1.0, "boolean": True}
    props = schema.get("properties") or {}
    return {k: values.get(props[k].get("type"), "x") for k in schema.get("required") or ()}


class Scripted:
    """An OpenAI-compatible chat model (an ``httpx`` transport): the first call of a task
    calls every tool it needs at once, the next answers. ``refuse`` answers every call as too
    long, the way llama.cpp's server does."""

    def __init__(self, *, refuse: bool = False) -> None:
        self.refuse = refuse
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST":
            return httpx.Response(200, json={"object": "list", "data": []})
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.refuse:
            return httpx.Response(
                400,
                json={
                    "error": {"message": "request (9000 tokens) exceeds the available context size"}
                },
            )
        messages = body["messages"]
        users = [m for m in messages if m["role"] == "user"]
        text = users[-1]["content"] if users else ""
        text = text if isinstance(text, str) else " ".join(p.get("text", "") for p in text)
        tools = {t["function"]["name"]: t["function"] for t in body.get("tools") or []}
        if any(m["role"] == "tool" for m in messages) or not tools:
            message: dict[str, Any] = {"role": "assistant", "content": "Done: ST-205 and ST-101."}
            finish = "stop"
        else:
            wanted = [
                n for n in next((_needed(u) for u in _texts(users) if _needed(u)), []) if n in tools
            ]
            calls = [
                {
                    "id": f"call_{len(self.bodies)}_{n}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(_sample(tools[name]["parameters"])),
                    },
                }
                for n, name in enumerate(wanted)
            ]
            message = {
                "role": "assistant",
                "content": None if calls else "Nothing to call.",
                "tool_calls": calls or None,
            }
            finish = "tool_calls" if calls else "stop"
        usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        if body.get("stream"):
            return httpx.Response(
                200,
                content=_sse(message, finish, usage),
                headers={"content-type": "text/event-stream"},
            )
        reply = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage,
        }
        return httpx.Response(200, json=reply)


def _texts(users: list[dict[str, Any]]) -> list[str]:
    out = []
    for u in users:
        c = u["content"]
        out.append(c if isinstance(c, str) else " ".join(p.get("text", "") for p in c))
    # a framework may wrap the task in its own words: the task is in there
    return [*out] + [
        task["input"]
        for s in catalog.task_sets()
        for task in s.tasks
        if isinstance(task["input"], str) and any(task["input"] in t for t in out)
    ]


def _sse(message: dict[str, Any], finish: str, usage: dict[str, Any]) -> bytes:
    delta: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        delta["tool_calls"] = [{**c, "index": n} for n, c in enumerate(message["tool_calls"])]
    chunks = [
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        },
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
        },
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [],
            "usage": usage,
        },
    ]
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def scripted() -> Scripted:
    return Scripted()


@contextlib.contextmanager
def serving(model: Scripted) -> Iterator[str]:
    """The metering proxy on a free port, in a thread, its upstream ``model``."""
    app = meter.Meter(
        "http://upstream", client=httpx.AsyncClient(transport=httpx.MockTransport(model))
    )
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="off")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def proxy(scripted: Scripted) -> Iterator[str]:
    with serving(scripted) as root:
        yield root


def _options(
    root: str, arm: str, out: Path, cells: tuple[str, ...] = ("short",), n: int = 2
) -> runner.Options:
    return runner.Options(
        arm=built.ARMS[arm],
        setup=built.Setup.on(root, "granite/test", "vk"),
        meter_url=root,
        cells=cells,
        n=n,
        seed=7,
        timeout=60.0,
        out=out,
    )


# --------------------------------------------------------------------------- the proxy
def test_usage_is_read_from_openai_and_anthropic_replies_and_streams() -> None:
    body = json.dumps(
        {
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 8},
            }
        }
    )
    got = meter.usage_of(body.encode())
    assert (got.input_tokens, got.output_tokens, got.cached_tokens) == (12, 3, 8)
    stream = _sse({"content": "hi"}, "stop", {"prompt_tokens": 40, "completion_tokens": 5})
    got = meter.usage_of(stream)
    assert (got.input_tokens, got.output_tokens, got.cached_tokens) == (40, 5, None)
    anthropic = (
        'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":70,"output_tokens":1,"cache_read_input_tokens":0}}}\n\n'
        'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":9}}\n\n'
    )
    got = meter.usage_of(anthropic.encode())
    assert (got.input_tokens, got.output_tokens, got.cached_tokens) == (70, 9, 0)
    plain = meter.usage_of(json.dumps({"usage": {"input_tokens": 5, "output_tokens": 2}}).encode())
    assert (plain.input_tokens, plain.output_tokens) == (5, 2)


@pytest.mark.parametrize(
    "body", [b"", b"not json", b"[1]", b"{bad", b"data: {bad\n\ndata: [1]\n\n"]
)
def test_a_reply_without_usage_counts_nothing(body: bytes) -> None:
    got = meter.usage_of(body)
    assert (got.input_tokens, got.output_tokens, got.cached_tokens) == (0, 0, None)


def test_every_call_is_sent_with_the_same_sampling() -> None:
    sent = json.loads(
        meter.normalized(
            json.dumps(
                {"model": "m", "temperature": 0.9, "max_tokens": 4096, "stream": True}
            ).encode(),
            "/v1/chat/completions",
            11,
        )
    )
    assert sent["seed"] == 11 and sent["temperature"] == meter.TEMPERATURE
    assert sent["max_tokens"] == meter.MAX_TOKENS and sent["stream_options"] == {
        "include_usage": True
    }
    small = json.loads(
        meter.normalized(
            json.dumps({"max_completion_tokens": 50}).encode(), "/v1/chat/completions", None
        )
    )
    assert (
        small["max_completion_tokens"] == 50 and "seed" not in small and "max_tokens" not in small
    )
    unset = json.loads(meter.normalized(b"{}", "/v1/chat/completions", None))
    assert unset["max_tokens"] == meter.MAX_TOKENS
    claude = json.loads(
        meter.normalized(json.dumps({"max_tokens": 32000}).encode(), "/anthropic/v1/messages", 3)
    )
    assert claude == {"max_tokens": meter.MAX_TOKENS, "temperature": meter.TEMPERATURE}
    assert meter.normalized(b"not json", "/v1/chat/completions", 1) == b"not json"
    assert meter.normalized(b"[1]", "/v1/chat/completions", 1) == b"[1]"


def test_only_completions_and_messages_are_model_calls() -> None:
    assert meter.is_call("POST", "/v1/chat/completions")
    assert meter.is_call("POST", "/anthropic/v1/messages")
    assert not meter.is_call("POST", "/anthropic/v1/messages/count_tokens")
    assert not meter.is_call("GET", "/v1/chat/completions")
    assert not meter.is_call("GET", "/v1/models")


def test_a_refused_call_counts_its_prompt_and_a_failure() -> None:
    m = meter.Meter("http://x")
    m.record(400, b'{"error": "request (9123 tokens) exceeds the available context size"}', 0.1)
    m.record(500, b"boom", 0.2)
    m.record(200, json.dumps({"usage": {"prompt_tokens": 3, "completion_tokens": 1}}).encode(), 0.3)
    c = m.counts
    assert (c.calls, c.failed, c.refused_tokens, c.input_tokens, c.cache_reported) == (
        3,
        2,
        9123,
        3,
        False,
    )
    assert c.seconds == [0.1, 0.2, 0.3]


async def test_the_proxy_passes_calls_on_counts_them_and_sets_the_seed() -> None:
    model = Scripted()
    app = meter.Meter(
        "http://upstream", client=httpx.AsyncClient(transport=httpx.MockTransport(model))
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://proxy"
    ) as client:
        set_seed = await client.post("/__meter", params={"seed": "5"})
        assert set_seed.json()["seed"] == 5
        reply = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert reply.status_code == 200 and reply.json()["choices"][0]["message"]["content"]
        listed = await client.get("/v1/models")  # passed on, not a model call
        assert listed.status_code == 200
        counts = (await client.get("/__meter")).json()
        unset = await client.post("/__meter", params={"seed": "x"})
    assert model.bodies[0]["seed"] == 5 and model.bodies[0]["temperature"] == meter.TEMPERATURE
    assert (counts["calls"], counts["input_tokens"], counts["output_tokens"]) == (1, 100, 10)
    assert unset.json()["seed"] is None


async def test_an_upstream_that_cannot_be_reached_is_a_failed_call() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    app = meter.Meter(
        "http://upstream", client=httpx.AsyncClient(transport=httpx.MockTransport(down))
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://proxy"
    ) as client:
        reply = await client.post("/v1/chat/completions", json={})
        assert reply.status_code == 502
        assert (await client.get("/v1/models")).status_code == 502  # not a call: not counted
    assert (app.counts.calls, app.counts.failed) == (1, 1)


async def test_the_proxy_answers_its_lifespan_and_closes_its_client() -> None:
    app = meter.Meter("http://upstream")
    assert app.client is app.client  # made once
    messages = iter([{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}])
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return next(messages)

    async def send(message: Any) -> None:
        sent.append(message)

    await app({"type": "lifespan"}, receive, send)
    assert [m["type"] for m in sent] == ["lifespan.startup.complete", "lifespan.shutdown.complete"]


def test_used_is_the_difference_of_two_snapshots() -> None:
    before = {
        "calls": 1,
        "failed": 0,
        "input_tokens": 10,
        "output_tokens": 1,
        "cached_tokens": 0,
        "cache_reported": False,
        "refused_tokens": 0,
        "seconds": [0.5],
    }
    after = {
        "calls": 3,
        "failed": 1,
        "input_tokens": 40,
        "output_tokens": 5,
        "cached_tokens": 8,
        "cache_reported": True,
        "refused_tokens": 900,
        "seconds": [0.5, 1.0, 2.0],
    }
    assert meter.used(before, after) == {
        "calls": 2, "failed": 1, "input_tokens": 30, "output_tokens": 4, "cached_tokens": 8,
        "cache_reported": True, "refused_tokens": 900, "seconds": [1.0, 2.0],
    }  # fmt: skip


# --------------------------------------------------------------------------- arms and cells
def test_the_arms_cover_every_framework_and_every_removed_feature() -> None:
    arms = built.ARMS
    assert {a.framework for a in arms.values()} == set(built.FRAMEWORKS)
    assert {a.framework for a in arms.values() if a.kind == "B"} == {
        "langchain",
        "deepagents",
        "openai_agents",
        "claude_agent_sdk",
    }
    assert {a.feature for a in arms.values() if a.kind == "C"} == {None, *built.REMOVED}
    assert [a.name for a in arms.values() if a.framework == "crewai"] == ["crewai/A"]
    assert arms["crewai/A"].venv == "crewai" and not arms["crewai/A"].wrapped
    assert {a.feature for a in arms.values() if a.kind == "A" and a.feature} == {
        "summarization",
        "planning",
        "session",
        "preload",
    }
    assert {a.name for a in arms.values() if a.framework == "trueforge"} == {
        "trueforge/A",
        "trueforge/A+preload",
    }
    assert all(arms[n].wrapped for n in ("langchain/B", "react/C-memory"))


def test_a_cell_goes_through_its_tasks_with_a_seed_per_pass() -> None:
    planned = runner.plan("long", 7, 10)
    assert [p.seed for p in planned] == [10, 10, 10, 11, 11, 11, 12]
    assert [p.task["metadata"]["length"] for p in planned] == ["long"] * 7
    assert planned[0].task is planned[3].task and planned[0].size == 10
    many = runner.plan("many", 6, 0)
    assert {p.size for p in many} == {100} and len({p.task["metadata"]["task"] for p in many}) == 6
    assert runner.plan("many50", 1, 0)[0].size == 50


@pytest.mark.parametrize(
    ("cell", "n", "said"), [("huge", 1, "no cell"), ("short", 0, "at least one")]
)
def test_a_cell_that_cannot_be_planned_is_refused(cell: str, n: int, said: str) -> None:
    with pytest.raises(ValueError, match=said):
        runner.plan(cell, n, 0)


def test_a_cell_with_no_task_of_its_length_is_refused() -> None:
    sets = [catalog.TaskSet("x", "s", [], [])]
    with pytest.raises(ValueError, match="no short task"):
        runner.plan("short", 1, 0, sets)


def test_a_change_task_needs_an_approval_and_a_read_does_not() -> None:
    store = catalog.task_sets()[0]
    change, read = store.tasks[1], store.tasks[0]
    assert runner.needs_approval(store, change) and not runner.needs_approval(store, read)


async def test_a_native_tool_runs_its_own_code_and_records_the_call() -> None:
    @tool(side_effects="read")
    def stock(sku: str) -> dict[str, Any]:
        """Units of a SKU in stock.

        Args:
            sku: The SKU.
        """
        if sku == "bad":
            raise ValueError("no such SKU")
        return {"sku": sku, "units": 4}

    [made] = await as_source(stock).resolve()
    calls = built.Calls()
    call = built.invoker(made, calls)
    assert json.loads(await call({"sku": "a-1"})) == {"sku": "a-1", "units": 4}
    assert (await call({"sku": "bad"})).startswith("Error: ValueError: no such SKU")
    assert calls.names == ["stock", "stock"]
    model = built.arguments_model(made)
    assert model(sku="x").sku == "x"  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        model()
    assert [o.status for _, o in calls.made] == ["ok", "error"]
    calls.reset()
    assert calls.names == []
    assert built.schema_of(made)["required"] == ["sku"]


async def test_crewai_arguments_take_the_schema_types_and_leave_optional_ones_out() -> None:
    [rule] = await as_source(catalog.scenario.create_stock_rule).resolve()
    fields = built.arguments_model(rule).model_fields
    assert set(fields) == set(built.schema_of(rule)["properties"])
    assert all(f.description for f in fields.values())
    for distractor in await catalog.Distractors(20).resolve():
        schema = built.schema_of(distractor)
        optional = [n for n in schema["properties"] if n not in (schema.get("required") or [])]
        if optional:
            made = built.arguments_model(distractor)(**_sample(schema))
            assert getattr(made, optional[0]) is None
            break
    else:
        pytest.fail("no distractor with an optional argument")


def test_a_wrapped_arm_is_no_native_one() -> None:
    with pytest.raises(ValueError, match="not a native arm"):
        built.native(
            built.ARMS["langchain/B"], built.Setup.on("http://x", "m", "k"), "s", [], built.Calls()
        )


async def test_an_unknown_wrapped_arm_is_refused() -> None:
    with pytest.raises(ValueError, match="is not wrapped"):
        await built.wrapped(
            built.ARMS["langchain/A"], built.Setup.on("http://x", "m", "k"), None, "s", []
        )
    odd = built.Arm("other/B", "other", "B", "none")
    with pytest.raises(ValueError, match="no wrapped arm"):
        await built.wrapped(odd, built.Setup.on("http://x", "m", "k"), None, "s", [])


def test_a_graphs_answer_is_its_last_ai_message() -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    assert built.answer_of_messages([HumanMessage("q"), AIMessage("a"), HumanMessage("x")]) == "a"
    parts = AIMessage(content=[{"type": "text", "text": "x"}, "y"])
    assert built.answer_of_messages([parts]) == "xy"
    assert built.answer_of_messages([]) == ""


def test_the_claude_cli_reaches_only_the_proxy() -> None:
    env = built.claude_env(built.Setup.on("http://p:1/", "m", "k"))
    assert env["ANTHROPIC_BASE_URL"] == "http://p:1/anthropic" and env["ANTHROPIC_API_KEY"] == "k"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


async def test_native_tool_formats_hold_the_same_tools() -> None:
    tools = [
        t
        for s in catalog.catalogue(catalog.task_sets()[0].tools, 10)
        for t in await as_source(s).resolve()
    ]
    calls = built.Calls()
    lc = built.as_langchain(tools, calls)
    oa = built.as_openai_agents(tools, calls)
    server = built.as_claude(tools, calls)
    assert [t.name for t in lc] == [t.name for t in oa] == [t.name for t in tools]
    assert server["name"] == built.SERVER
    bad = await oa[0].on_invoke_tool(None, "{not json")
    assert bad.startswith("Error: the arguments are not JSON")
    listed = await oa[0].on_invoke_tool(None, "[]")  # not an object: no arguments
    assert isinstance(listed, str)


# --------------------------------------------------------------------------- whole cells
@pytest.mark.parametrize(
    "arm",
    [
        "langchain/A",
        "langchain/A+summarization",
        "deepagents/A+planning",
        "openai_agents/A",
        "openai_agents/A+session",
    ],
)
async def test_a_native_arm_runs_its_cell_and_keeps_every_run(
    proxy: str, scripted: Scripted, tmp_path: Path, arm: str
) -> None:
    out = tmp_path / "rows.jsonl"
    seen: list[str] = []
    rows = await runner.run_arm(_options(proxy, arm, out), progress=seen.append)
    assert len(rows) == 2 == len(seen) == len(out.read_text().splitlines())
    first = rows[0]
    assert first["arm"] == arm and first["kind"] == "A" and first["cell"] == "short"
    assert first["success"] is True and first["tool_coverage"] == 1.0
    assert sorted(first["called"]) == ["excess_stock", "lost_sales"]
    assert first["model_calls"] >= 2 and first["input_tokens"] >= 200
    assert first["seed"] == 7 and first["cpu_seconds"]["arm"] is not None
    assert any(b.get("seed") == 7 for b in scripted.bodies)
    assert rows[1]["task"] == "store: a change behind approval" and rows[1]["success"] is True
    # resumed: nothing left to run
    assert await runner.run_arm(_options(proxy, arm, out)) == []


async def test_a_refused_native_run_is_an_error_with_its_prompt_counted(tmp_path: Path) -> None:
    with serving(Scripted(refuse=True)) as root:
        rows = await runner.run_arm(_options(root, "openai_agents/A", tmp_path / "r.jsonl", n=1))
    [row] = rows
    assert row["status"] == "error" and row["success"] is False
    assert row["refused_prompt_tokens"] >= 9000 and row["failed_model_calls"] >= 1


@pytest.mark.parametrize(
    "arm", ["langchain/B", "deepagents/B", "openai_agents/B", "react/C", "react/C-planning"]
)
async def test_a_wrapped_arm_runs_its_cell_through_the_harness(
    proxy: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    for name in ("MEMORY_URL", "RUNS_URL", "TRELLIS_API_KEY", "OTEL_EXPORTER_OTLP_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BIFROST_URL", f"{proxy}/v1")
    monkeypatch.setenv("BIFROST_VIRTUAL_KEY", "vk")
    out = tmp_path / "rows.jsonl"
    rows = await runner.run_arm(_options(proxy, arm, out), progress=lambda _: None)
    assert [r["kind"] for r in rows] == [arm.split("/")[1][0]] * 2
    read, change = rows
    assert read["success"] is True and sorted(read["called"]) == ["excess_stock", "lost_sales"]
    assert read["model_calls"] >= 2 and read["input_tokens"] >= 200
    # the change pauses at its approval: done, as far as the task goes
    assert change["status"] == "interrupted" and change["success"] is True


async def test_the_load_check_measures_runs_at_once(
    proxy: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("MEMORY_URL", "RUNS_URL", "TRELLIS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BIFROST_URL", f"{proxy}/v1")
    out = tmp_path / "rows.jsonl"
    native = await runner.load_check(
        _options(proxy, "langchain/A", out), 3, 3, progress=lambda _: None
    )
    wrapped = await runner.load_check(
        _options(proxy, "react/C", out), 2, 2, progress=lambda _: None
    )
    assert (native["runs"], native["successes"], native["errors"]) == (3, 3, 0)
    assert native["model_calls"] == 6 and native["p95"] >= native["p50"] > 0
    assert (wrapped["successes"], wrapped["concurrency"]) == (2, 2)
    loads = [r for r in report.read_rows(out) if r.get("load")]
    assert len(loads) == 2


def test_a_run_keeps_its_usage_and_resets_its_peak() -> None:
    with runner.measured() as usage:
        sum(range(10_000))
    assert usage["seconds"] >= 0 and usage["cpu_seconds"] >= 0
    assert usage["peak_rss_mb"] > 0 and isinstance(usage["peak_rss_per_run"], bool)


def test_resetting_the_peak_where_proc_refuses_it(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(self: Path, text: str) -> int:
        raise PermissionError("no")

    monkeypatch.setattr(Path, "write_text", refuse)
    assert runner.reset_peak() is False


def test_done_keys_are_the_arms_own_runs(tmp_path: Path) -> None:
    out = tmp_path / "rows.jsonl"
    assert runner.done_keys(out, "a") == set()
    out.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"arm": "a", "cell": "short", "index": 0},
                {"arm": "b", "cell": "short", "index": 1},
                {"arm": "a", "load": True},
            ]
        )
        + "\n\n"
    )
    assert runner.done_keys(out, "a") == {("short", 0)}


def test_agent_ids_are_one_per_arm_set_and_size() -> None:
    assert (
        runner.agent_id("deepagents/A+planning", "help desk", 10)
        == "fwbench-deepagents-A-planning-help-desk-10"
    )


def test_the_survival_worker_is_started_with_the_arm() -> None:
    line = survive.worker_command("react/C", "fws-1", built.Setup.on("http://p", "m", "k"), 60.0)
    assert line[1:4] == ["-m", "benchmark.cost.frameworks.survive", "worker"]
    assert "react/C" in line and "fws-1" in line and "k" not in line


async def test_the_first_call_is_waited_for_within_its_time(proxy: str) -> None:
    assert await survive._first_call(proxy, 0, 0.3) is False
    async with httpx.AsyncClient() as client:
        await client.post(f"{proxy}/v1/chat/completions", json={"model": "m", "messages": []})
    assert await survive._first_call(proxy, 0, 5.0) is True


# --------------------------------------------------------------------------- the report
def _row(
    arm: str, cell: str, i: int, *, success: bool, tokens: int, seconds: float = 10.0
) -> dict[str, Any]:
    known = built.ARMS[arm]
    return {
        "arm": arm, "framework": known.framework, "kind": known.kind, "cell": cell, "index": i,
        "success": success, "tool_coverage": 1.0 if success else 0.5, "input_tokens": tokens,
        "output_tokens": 10, "model_calls": 2, "errored": False, "seconds": seconds,
        "cpu_seconds": {"arm": 1.0}, "peak_rss_mb": 300.0, "date": "2026-10-21",
    }  # fmt: skip


def _rows(
    scores: dict[str, tuple[float, int]], cells: tuple[str, ...] = ("short", "long")
) -> list[dict[str, Any]]:
    rows = []
    for arm, (rate, tokens) in scores.items():
        for cell in cells:
            for i in range(6):
                rows.append(_row(arm, cell, i, success=i < round(rate * 6), tokens=tokens + i))
    return rows


def test_react_is_best_when_it_wins_on_accuracy_and_is_no_dearer() -> None:
    rows = _rows(
        {"react/C": (1.0, 1000), "langchain/A": (0.5, 900), "openai_agents/B": (1.0, 1200)}
    )
    decided = report.verdict(report.summarize(rows))
    assert decided["react_best"] is True and decided["rival"] == "openai_agents/B"
    assert decided["text"].startswith("**ReAct best.**")


def test_react_is_not_best_when_out_scored_and_the_winner_is_named() -> None:
    rows = _rows({"react/C": (0.5, 1000), "deepagents/A+planning": (1.0, 900)})
    decided = report.verdict(report.summarize(rows))
    assert decided["react_best"] is False and decided["lost"] == ["short", "long"]
    assert "**deepagents**" in decided["text"] and "write_todos planning" in decided["text"]


def test_react_is_not_best_when_it_costs_more_at_the_same_accuracy() -> None:
    rows = _rows({"react/C": (1.0, 5000), "langchain/A": (1.0, 1000)})
    decided = report.verdict(report.summarize(rows))
    assert decided["react_best"] is False and decided["lost"] == []
    assert "Its tokens per run" in decided["text"] and "**langchain**" in decided["text"]


def test_no_verdict_without_react_or_a_rival() -> None:
    assert report.verdict(report.summarize(_rows({"langchain/A": (1.0, 1)})))["react_best"] is False
    assert "Not decided" in report.verdict(report.summarize(_rows({"react/C": (1.0, 1)})))["text"]


@pytest.mark.parametrize(
    ("arm", "said"),
    [
        ("langchain/A", "langchain's own agent loop"),
        ("langchain/A+summarization", "SummarizationMiddleware"),
        ("openai_agents/B", "the harness around it"),
        ("react/C-files", "ReAct without files"),
        ("react/C", "ReAct as built"),
        ("unknown/X", "its loop"),
    ],
)
def test_the_winning_feature_is_named(arm: str, said: str) -> None:
    assert said in report.feature_of(arm)


def test_the_ranking_puts_success_first_then_fewer_tokens() -> None:
    rows = _rows(
        {"react/C": (1.0, 2000), "langchain/A": (1.0, 1000), "openai_agents/A": (0.5, 10)},
        ("many",),
    )
    ranked = [arm for arm, _ in report.ranking(report.summarize(rows), "many")]
    assert ranked == ["langchain/A", "react/C", "openai_agents/A"]


def test_the_markdown_has_every_section(tmp_path: Path) -> None:
    rows = _rows({"react/C": (1.0, 1000), "langchain/A": (0.5, 900)})
    rows += [
        {"arm": "react/C", "framework": "react", "kind": "C", "load": True, "concurrency": 1, "runs": 5, "successes": 5, "errors": 0, "p50": 1.0, "p95": 2.0, "wall_seconds": 9.0, "throughput_per_min": 33.3, "model_calls": 10, "cpu_seconds": 1.0, "peak_rss_mb": 300.0},
        {"arm": "react/C", "framework": "react", "kind": "C", "survive": True, "trial": 0, "survived": True, "attempts": 2, "seconds": 80.0, "error": None},
        {"arm": "claude_agent_sdk/A", "skipped": "needs an Anthropic key"},
    ]  # fmt: skip
    text = report.markdown(rows, {"intro": "Intro.", "notes": ["A note."]})
    for heading in (
        "## All task types",
        "## Ranking: short (10 tools)",
        "## Ranking: long (10 tools)",
        "## Load check",
        "## Kill mid-run",
        "## Skipped",
        "## Verdict",
    ):
        assert heading in text
    assert "**claude_agent_sdk/A** — needs an Anthropic key" in text and "A note." in text
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in [*rows, rows[0]]) + "\n")
    assert len(report.read_rows(path)) == len(rows)
    assert report.read_rows(tmp_path / "none.jsonl") == []
    assert report.title("odd") == "odd"


# --------------------------------------------------------------------------- the command
def test_the_command_line_is_checked() -> None:
    args = cli.arguments(["--model", "m", "--load", "1:5,10:10"])
    assert args.command == "run" and args.load == [(1, 5), (10, 10)] and args.n == 6
    for bad in (["--arms", "nope/A"], ["--n", "0"], ["--cells", "huge"], ["--load", "0:1"], []):
        with pytest.raises(SystemExit):
            cli.arguments(["--model", "m", *bad] if bad else ["run"])
    assert cli.arguments(["report"]).command == "report"


def test_the_arm_order_is_fixed_by_the_seed_and_shuffled() -> None:
    names = list(built.ARMS)
    assert cli.shuffled(names, 7) == cli.shuffled(names, 7) != names
    assert sorted(cli.shuffled(names, 8)) == sorted(names)


def test_a_native_child_has_no_harness_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_URL", "http://m")
    monkeypatch.setenv("RUNS_URL", "http://r")
    native = cli.child_env(built.ARMS["langchain/A"], "http://p", "k")
    wrapped = cli.child_env(built.ARMS["langchain/B"], "http://p", "k")
    assert "MEMORY_URL" not in native and "RUNS_URL" not in native
    assert wrapped["MEMORY_URL"] == "http://m" and wrapped["BIFROST_URL"] == "http://p/v1"
    assert (
        cli.upstream_of("http://g:8180/v1/") == "http://g:8180"
        and cli.upstream_of("http://g") == "http://g"
    )


def test_crewai_without_its_interpreter_is_skipped_saying_why() -> None:
    assert "no CrewAI interpreter" in (cli.crewai_problem(None) or "")
    assert "cannot run the arm" in (cli.crewai_problem("/bin/false") or "")


def test_an_unreachable_anthropic_route_says_why() -> None:
    assert "did not answer" in (cli.anthropic_route("http://127.0.0.1:9", "k", "m") or "")


async def test_an_anthropic_route_that_refuses_says_so(proxy: str) -> None:
    # the scripted model answers OpenAI's shape: not an Anthropic message
    said = await asyncio.to_thread(cli.anthropic_route, proxy, "k", "m")
    assert said is not None and "answered 200" in said


def test_the_report_command_writes_the_doc(tmp_path: Path) -> None:
    out = tmp_path / "bench"
    out.mkdir()
    (out / "rows.jsonl").write_text(
        "".join(
            json.dumps(r) + "\n" for r in _rows({"react/C": (1.0, 9), "langchain/A": (1.0, 900)})
        )
    )
    doc = tmp_path / "frameworks.md"
    assert cli.main(["report", "--out", str(out), "--doc", str(doc)]) == 0
    assert "**ReAct best.**" in doc.read_text() and (out / "summary.json").exists()


# --------------------------------------------------------------------------- TrueForge
def test_trueforge_names_are_resource_names() -> None:
    from benchmark.cost.frameworks import trueforge as tf

    assert tf.resource_name("Help Desk: 100 tools!") == "help-desk-100-tools"
    assert tf.resource_name("100 tools") == "s-100-tools"
    assert tf.resource_name("---") == "s"
    assert len(tf.resource_name("x" * 200)) <= 63


def test_trueforge_runs_with_no_person_in_the_loop_and_says_what_to_preload() -> None:
    from benchmark.cost.frameworks import trueforge as tf

    agent = tf.spec("bench/granite", "Be brief.", "tools-1", preload=True)
    [server] = agent["mcp_servers"]
    assert server["require_approval_for_tools"] == [] and server["preload"] is True
    assert agent["config"]["ask_user_questions"] == {"enabled": False}
    assert agent["config"]["sandbox"] == {"enabled": False}
    assert tf.spec("m", "s", "n", preload=False)["mcp_servers"][0]["preload"] is False
    assert tf.text_of([{"text": "a"}, {"text": "b"}, "x"]) == "a\nb" and tf.text_of(None) == ""
    assert tf.text_of("plain") == "plain"


def test_trueforge_is_skipped_without_its_install(tmp_path: Path) -> None:
    from benchmark.cost.frameworks import trueforge as tf

    assert "no TrueForge install" in (tf.install_problem({}) or "")
    assert "no TrueForge CLI" in (tf.install_problem({tf.DIR: str(tmp_path)}) or "")
    cli_path = tmp_path / tf.CLI
    cli_path.parent.mkdir(parents=True)
    cli_path.write_text("")
    assert tf.install_problem({tf.DIR: str(tmp_path)}) is None


async def test_the_tools_are_served_to_trueforge_as_mcp_with_their_own_schemas() -> None:
    from benchmark.cost.frameworks import trueforge as tf

    tools = [
        t
        for s in catalog.catalogue(catalog.task_sets()[0].tools, 10)
        for t in await as_source(s).resolve()
    ]
    calls = built.Calls()
    by_name = {t.name: built.invoker(t, calls) for t in tools}

    async def call(name: str, args: dict[str, Any]) -> str:
        return await by_name[name](args)

    served = tf.Served(tf.mcp_app(tools, call))
    try:
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with (
            streamable_http_client(served.url) as (read, write, *_),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            listed = await session.list_tools()
            assert [t.name for t in listed.tools] == [t.name for t in tools]
            assert listed.tools[0].input_schema["type"] == "object"
            done = await session.call_tool("excess_stock", {"store": "S12"})
            failed = await session.call_tool("excess_stock", {})
    finally:
        served.server.should_exit = True
        served.thread.join(timeout=10)
    text = getattr(done.content[0], "text", "")
    assert "ST-205" in text and not done.is_error
    assert failed.is_error and calls.names == ["excess_stock", "excess_stock"]


@pytest.mark.skipif(
    not os.environ.get("FWBENCH_TRUEFORGE_DIR"),
    reason="needs a TrueForge install (FWBENCH_TRUEFORGE_DIR)",
)
@pytest.mark.parametrize(("arm", "covered"), [("trueforge/A+preload", 1.0), ("trueforge/A", 0.0)])
async def test_a_trueforge_arm_runs_its_cell(
    proxy: str, tmp_path: Path, arm: str, covered: float
) -> None:
    rows = await runner.run_arm(
        _options(proxy, arm, tmp_path / "rows.jsonl"), progress=lambda _: None
    )
    # the scripted model calls the task's tools by name: preloaded, they are there; deferred,
    # it is offered TrueForge's meta tools instead and answers without them
    assert [r["tool_coverage"] for r in rows] == [covered, covered]
    assert all(r["status"] == "success" and r["model_calls"] >= 1 for r in rows)
