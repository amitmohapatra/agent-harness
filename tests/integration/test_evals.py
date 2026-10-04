"""Evaluation: ``h.evaluate`` over a local list and a Langfuse dataset (a fake Langfuse), the
built-in evaluators, the judge's model and virtual key, and the online judges."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import pytest
import respx

from tests.support.memory import FakeMemoryService
from trellis import (
    EvalCase,
    EvalItem,
    EvalScore,
    Harness,
    ReAct,
    Runtime,
    Settings,
    contains,
    exact_match,
    grounding,
    llm_judge,
)
from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness import telemetry
from trellis.harness.clients.memory import Memory

LF = "https://lf.test"
#: Langfuse's credentials as the OTLP headers carry them (what reaches its public API too).
OTLP = {"authorization": "Basic cGs6c2s=", "x-langfuse-host": LF}
GW = "http://gw.test/v1"


def judge_reply(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class JudgeOrAnswer:
    """A scripted chat model that answers as the agent, or — asked by the judge — with the
    next scripted verdict."""

    def __init__(self, verdicts: list[str], answer: str = "Paris") -> None:
        self.verdicts = list(verdicts)
        self.answer = answer
        self.judged: list[list[dict[str, Any]]] = []
        self.bodies: list[dict[str, Any]] = []

    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
        if "strict evaluator" in str(messages[0].get("content")):
            self.judged.append([dict(m) for m in messages])
            self.bodies.append(body)
            return judge_reply(self.verdicts.pop(0))
        return judge_reply(self.answer)


def langfuse_harness(service: FakeMemoryService | None = None, **settings: Any) -> Harness:
    config = {"otlp_headers": OTLP, **settings}
    if service is not None:
        config.update(memory_url="http://m", api_key="test")
    h = Harness(config=Settings(**config))
    if service is not None:
        h.memory = Memory("http://m", None, client=service.client())
    return h


async def capital(input: str, agent: Runtime) -> str:
    return {"France": "Paris", "Italy": "Rome"}.get(input, "I don't know")


# --------------------------------------------------------------------------- offline, local
async def test_a_local_dataset_is_run_scored_and_summarised(harness: Harness) -> None:
    agent = harness.wrap(capital, id="capitals")
    report = await harness.evaluate(
        agent,
        [
            {"input": "France", "expected": "paris"},
            EvalItem(input="Italy", expected="Rome", metadata={"region": "south"}),
            {"input": "Spain", "expected": ["Madrid"]},
        ],
        [exact_match(), contains()],
        run_name="nightly",
    )
    assert report.run_name == "nightly" and report.dataset is None
    assert [i.input for i in report.items] == ["France", "Italy", "Spain"]
    assert [i.output for i in report.items] == ["Paris", "Rome", "I don't know"]
    assert all(i.status == "success" and i.trace_url is None for i in report.items)
    exact = report.summary["exact_match"]
    assert (exact.mean, exact.count, exact.failures) == (0.6667, 3, 0)
    assert report.items[2].scores[1] == EvalScore("contains", False, "missing: Madrid")
    assert report.statuses == {"success": 3}
    printed = str(report)
    assert printed.startswith("nightly: 3 items (3 success)") and "exact_match" in printed
    assert "0.667" in str(report.summary)
    record = await harness.runs.get(report.items[0].run_id or "")
    assert record is not None and record.user_id == "trellis-evaluate"


async def test_a_failing_item_a_pausing_item_and_a_failing_evaluator_stop_nothing(
    harness: Harness,
) -> None:
    async def moody(input: str, agent: Runtime) -> str:
        if input == "boom":
            raise RuntimeError("the agent broke")
        if input == "ask":
            return await agent.ask("Really?")
        return "fine"

    async def broken(case: EvalCase) -> EvalScore | None:
        raise ValueError("bad evaluator")

    async def silent(case: EvalCase) -> EvalScore | None:
        return None

    agent = harness.wrap(moody, id="moody")
    report = await harness.evaluate(
        agent, [{"input": i} for i in ("boom", "ask", "ok")], [broken, silent], user="qa"
    )
    boom, ask, ok = report.items
    assert (boom.status, boom.error) == ("error", "the agent broke")
    assert ask.status == "interrupted" and ask.scores == []
    paused = await harness.runs.get(ask.run_id or "")
    assert paused is not None and paused.status is RunStatus.CANCELLED  # not left in an inbox
    assert await harness.inbox() == []
    assert ok.status == "success" and ok.failed == {"broken": "ValueError: bad evaluator"}
    assert report.summary["broken"].failures == 1 and report.summary["silent"].count == 0
    assert report.summary["silent"].mean is None


async def test_an_item_the_harness_cannot_even_start_is_an_error(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trellis.harness.clients.runs import RunStoreError

    async def refused(*args: Any, **kwargs: Any) -> Any:
        raise RunStoreError("agent-runs is down")

    monkeypatch.setattr(harness.runs, "started", refused)
    agent = harness.wrap(capital, id="capitals")
    report = await harness.evaluate(agent, [{"input": "France"}], [exact_match()])
    [item] = report.items
    assert item.status == "error" and item.run_id is None
    assert item.error == "RunStoreError: agent-runs is down"


async def test_a_cancelled_run_is_reported_cancelled(harness: Harness) -> None:
    async def cancelled(input: str, agent: Runtime) -> str:
        from trellis.harness.runtime import RunCancelled

        raise RunCancelled("no")

    report = await harness.evaluate(harness.wrap(cancelled, id="c"), [{"input": "x"}], [])
    assert report.items[0].status == "cancelled"


async def test_items_run_at_most_concurrency_at_a_time(harness: Harness) -> None:
    running, most = 0, 0

    async def slow(input: str, agent: Runtime) -> str:
        nonlocal running, most
        running += 1
        most = max(most, running)
        await asyncio.sleep(0.01)
        running -= 1
        return input

    agent = harness.wrap(slow, id="slow")
    report = await harness.evaluate(
        agent, [{"input": str(n)} for n in range(9)], [], concurrency=2, limit=7
    )
    assert most == 2 and [i.output for i in report.items] == [str(n) for n in range(7)]
    with pytest.raises(ConfigurationError, match="at least one"):
        await harness.evaluate(agent, [], [], concurrency=0)


async def test_a_dataset_name_needs_langfuse(harness: Harness) -> None:
    with pytest.raises(ConfigurationError, match="read from Langfuse"):
        await harness.evaluate(harness.wrap(capital, id="c"), "capitals", [])


# --------------------------------------------------------------------------- offline, Langfuse
def langfuse_items(routes: respx.MockRouter) -> tuple[respx.Route, respx.Route, respx.Route]:
    items = [
        {
            "id": "item-1",
            "status": "ACTIVE",
            "input": "France",
            "expectedOutput": "Paris",
            "metadata": {"level": "easy"},
        },
        {
            "id": "item-0",
            "status": "ARCHIVED",
            "input": "Peru",
            "expectedOutput": "Lima",
            "metadata": None,
        },
        {
            "id": "item-2",
            "status": "ACTIVE",
            "input": "Italy",
            "expectedOutput": "Milan",
            "metadata": None,
        },
    ]
    dataset = routes.get(f"{LF}/api/public/v2/datasets/geo%2Fcapitals").mock(
        return_value=httpx.Response(200, json={"id": "ds-1", "name": "geo/capitals"})
    )

    def page(request: httpx.Request) -> httpx.Response:
        number = int(request.url.params["page"])
        data = items[:2] if number == 1 else items[2:]
        meta = {"page": number, "limit": 50, "totalItems": 3, "totalPages": 2}
        return httpx.Response(200, json={"data": data, "meta": meta})

    pages = routes.get(f"{LF}/api/public/dataset-items").mock(side_effect=page)
    linked = routes.post(f"{LF}/api/public/dataset-run-items").mock(
        return_value=httpx.Response(200, json={"id": "dri"})
    )
    return dataset, pages, linked


@respx.mock
async def test_a_langfuse_dataset_is_read_page_by_page_scored_and_linked() -> None:
    dataset, pages, linked = langfuse_items(respx.mock)
    scores = respx.post(f"{LF}/api/public/scores").mock(
        return_value=httpx.Response(200, json={"id": "s"})
    )
    async with langfuse_harness() as h:
        agent = h.wrap(capital, id="capitals")
        report = await h.evaluate(agent, "geo/capitals", [exact_match()], run_name="r1")
    assert dataset.call_count == 1
    assert [dict(c.request.url.params) for c in pages.calls] == [
        {"datasetName": "geo/capitals", "page": "1", "limit": "50"},
        {"datasetName": "geo/capitals", "page": "2", "limit": "50"},
    ]
    assert report.dataset == "geo/capitals"
    assert [(i.input, i.expected) for i in report.items] == [
        ("France", "Paris"),
        ("Italy", "Milan"),
    ]
    first, second = report.items
    trace = telemetry.trace_hex(first.run_id or "")
    assert first.trace_url == f"{LF}/trace/{trace}"
    bodies = [json.loads(c.request.content) for c in linked.calls]
    assert sorted(bodies, key=lambda b: b["datasetItemId"]) == [
        {
            "runName": "r1",
            "datasetItemId": "item-1",
            "traceId": trace,
            "metadata": {"agent_id": "capitals"},
        },
        {
            "runName": "r1",
            "datasetItemId": "item-2",
            "traceId": telemetry.trace_hex(second.run_id or ""),
            "metadata": {"agent_id": "capitals"},
        },
    ]
    posted = sorted((json.loads(c.request.content) for c in scores.calls), key=lambda b: b["value"])
    assert posted == [
        {
            "id": f"{second.run_id}:exact_match",
            "traceId": telemetry.trace_hex(second.run_id or ""),
            "name": "exact_match",
            "value": 0.0,
            "dataType": "BOOLEAN",
        },
        {
            "id": f"{first.run_id}:exact_match",
            "traceId": trace,
            "name": "exact_match",
            "value": 1.0,
            "dataType": "BOOLEAN",
        },
    ]


@respx.mock
async def test_a_score_langfuse_refuses_is_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    respx.post(f"{LF}/api/public/scores").mock(return_value=httpx.Response(503))
    async with langfuse_harness() as h:
        report = await h.evaluate(
            h.wrap(capital, id="c"), [{"input": "France", "expected": "Paris"}], [exact_match()]
        )
    assert report.summary["exact_match"].mean == 1.0
    assert "score exact_match of run" in caplog.text and "was not posted" in caplog.text


@respx.mock
async def test_a_dataset_langfuse_does_not_have_is_a_configuration_error() -> None:
    respx.get(f"{LF}/api/public/v2/datasets/nope").mock(return_value=httpx.Response(404))
    async with langfuse_harness() as h:
        with pytest.raises(ConfigurationError, match="no dataset 'nope'"):
            await h.evaluate(h.wrap(capital, id="c"), "nope", [])


@respx.mock
async def test_a_link_langfuse_refuses_is_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    langfuse_items(respx.mock)
    respx.post(f"{LF}/api/public/dataset-run-items").mock(return_value=httpx.Response(500))
    async with langfuse_harness() as h:
        report = await h.evaluate(h.wrap(capital, id="c"), "geo/capitals", [], limit=1)
    assert report.items[0].status == "success"
    assert "was not linked to dataset run" in caplog.text


@respx.mock
async def test_a_react_agent_with_memory_is_graded_for_grounding_and_by_a_judge(
    memory_service: FakeMemoryService,
) -> None:
    langfuse_items(respx.mock)
    scores = respx.post(f"{LF}/api/public/scores").mock(
        return_value=httpx.Response(200, json={"id": "s"})
    )
    model = JudgeOrAnswer(['{"score": 0.9, "reasoning": "names the capital"}'] * 2)
    async with langfuse_harness(memory_service, grounding_sample=0.0) as h:
        h._judge_model = model
        agent = h.wrap(ReAct(system="You know capitals.", model=model), id="geo")
        report = await h.evaluate(
            agent, "geo/capitals", [grounding(), llm_judge("Names the right capital city.")]
        )
    assert report.summary["grounding"].mean == 0.8 and report.summary["grounding"].count == 2
    assert report.summary["llm_judge"].mean == 0.9
    verified = memory_service.named("verify")
    assert len(verified) == 2 and all(v.body["answer"] == "Paris" for v in verified)
    asked = model.judged[0][1]["content"]
    assert "## Criteria\nNames the right capital city." in asked
    assert "## Expected answer\n" in asked and memory_service.context_text in asked
    assert model.bodies[0] == {"temperature": 0}
    judged = [
        json.loads(c.request.content) for c in scores.calls if b"llm_judge" in c.request.content
    ]
    assert judged[0]["comment"] == "names the capital" and judged[0]["dataType"] == "NUMERIC"


async def test_grounding_gives_no_score_without_memory_or_a_text_answer(harness: Harness) -> None:
    agent = harness.wrap(capital, id="c")
    judge = grounding()
    assert await judge(EvalCase(input="q", output="a")) is None
    assert (
        await judge(EvalCase(input="q", output=None, run_id="r", bundle_id="b", agent=agent))
        is None
    )
    assert (
        await judge(EvalCase(input="q", output="a", run_id="r", bundle_id="b", agent=agent)) is None
    )


async def test_grounding_gives_no_score_for_an_answer_with_no_claim(
    memory_service: FakeMemoryService,
) -> None:
    memory_service.claims = 0
    memory_service.unsupported = 0
    async with langfuse_harness(memory_service, otlp_headers={}, grounding_sample=0.0) as h:
        report = await h.evaluate(h.wrap(capital, id="c"), [{"input": "France"}], [grounding()])
    assert report.items[0].scores == [] and len(memory_service.named("verify")) == 1


async def test_a_score_of_a_case_with_no_run_is_kept_off_any_trace(harness: Harness) -> None:
    from trellis.harness.evals import scored

    async def category(case: EvalCase) -> EvalScore | None:
        return EvalScore("tone", "polite")

    scores, failed = await scored(harness, EvalCase("q", "a", "a"), [exact_match(), category])
    assert [s.value for s in scores] == [True, "polite"] and failed == {}


@respx.mock
async def test_a_categorical_score_goes_to_langfuse_as_one() -> None:
    scores = respx.post(f"{LF}/api/public/scores").mock(
        return_value=httpx.Response(200, json={"id": "s"})
    )

    async def tone(case: EvalCase) -> EvalScore | None:
        return EvalScore("tone", "polite", "said please")

    async with langfuse_harness() as h:
        report = await h.evaluate(h.wrap(capital, id="c"), [{"input": "France"}], [tone])
    [body] = [json.loads(c.request.content) for c in scores.calls]
    assert (body["value"], body["dataType"], body["comment"]) == (
        "polite",
        "CATEGORICAL",
        "said please",
    )
    assert report.summary["tone"].mean is None and report.summary["tone"].count == 1


async def test_exact_match_and_contains_read_expected_values() -> None:
    assert await exact_match()(EvalCase(input="q", output="x")) is None
    assert await contains()(EvalCase(input="q", output="x")) is None
    assert (await exact_match(case_sensitive=True)(EvalCase("q", "Paris", "paris"))).value is False  # type: ignore[union-attr]
    assert (await exact_match()(EvalCase("q", {"a": 1}, {"a": 1}))).value is True  # type: ignore[union-attr]
    found = await contains(case_sensitive=True)(EvalCase("q", {"city": "Rome"}, ["Rome", "rome"]))
    assert found == EvalScore("contains", False, "missing: rome")


# --------------------------------------------------------------------------- the judge
@pytest.mark.parametrize(
    ("verdicts", "value", "asked"),
    [
        (['{"score": 1, "reasoning": "yes"}'], 1.0, 1),
        (["I think it is fine.", '```json\n{"score": 0.25, "reasoning": "meh"}\n```'], 0.25, 2),
        (['{"score": 7}', '{"score": 0.5}'], 0.5, 2),
        (["", "nope"], None, 2),
        (['{"score": "high"', '{"score": 0.5,}'], None, 2),
        (['{"score": true}', '{"reasoning": "no score"}'], None, 2),
    ],
    ids=[
        "valid",
        "prose-then-fenced",
        "out-of-range-then-valid",
        "empty-then-prose",
        "broken-then-not-an-object",
        "bool-then-missing",
    ],
)
async def test_the_judge_reads_strict_json_and_asks_once_more(
    harness: Harness,
    caplog: pytest.LogCaptureFixture,
    verdicts: list[str],
    value: float | None,
    asked: int,
) -> None:
    model = JudgeOrAnswer(verdicts)
    harness._judge_model = model
    agent = harness.wrap(capital, id="c")
    score = await llm_judge("Is it right?", name="right")(
        EvalCase(input="France", output="Paris", run_id="run_x", agent=agent)
    )
    assert len(model.judged) == asked
    if value is None:
        assert score is None and "gave no score for run run_x" in caplog.text
    else:
        assert score is not None and (score.name, score.value) == ("right", value)
    if asked == 2:
        retry = model.judged[1][-1]["content"]
        assert retry.startswith("That was not the JSON object asked for (")


async def test_the_judge_judges_harness_runs_only() -> None:
    with pytest.raises(ConfigurationError, match="harness agent"):
        await llm_judge("x")(EvalCase(input="q", output="a"))


@respx.mock
async def test_the_judge_asks_the_judge_model_through_bifrost_with_the_judge_key() -> None:
    respx.post(f"{GW.rsplit('/v1', 1)[0]}/mcp").mock(
        return_value=httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})
    )
    chat = respx.post(f"{GW}/chat/completions").mock(
        return_value=httpx.Response(200, json=judge_reply('{"score": 0.7, "reasoning": "ok"}'))
    )
    settings = Settings(
        bifrost_url=GW,
        bifrost_virtual_key="agent-key",
        judge_model="judges/strong",
        judge_virtual_key="eval-key",
    )
    async with Harness(config=settings) as h:
        assert h.judge_gateway is not h.gateway
        report = await h.evaluate(
            h.wrap(capital, id="c"), [{"input": "France"}], [llm_judge("ok?")]
        )
    assert report.summary["llm_judge"].mean == 0.7
    [call] = chat.calls
    assert call.request.headers["authorization"] == "Bearer eval-key"
    body = json.loads(call.request.content)
    assert body["model"] == "judges/strong" and body["temperature"] == 0


@respx.mock
async def test_without_a_judge_key_or_model_the_agents_are_used(
    caplog: pytest.LogCaptureFixture,
) -> None:
    respx.post(f"{GW.rsplit('/v1', 1)[0]}/mcp").mock(
        return_value=httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})
    )

    def answer(request: httpx.Request) -> httpx.Response:
        judging = b"strict evaluator" in request.content
        return httpx.Response(
            200, json=judge_reply('{"score": 0.4, "reasoning": "r"}' if judging else "Paris")
        )

    chat = respx.post(f"{GW}/chat/completions").mock(side_effect=answer)
    settings = Settings(
        bifrost_url=GW, bifrost_virtual_key="agent-key", judge_virtual_key="agent-key"
    )
    with caplog.at_level(logging.WARNING, logger="trellis.evals"):
        async with Harness(config=settings) as h:
            assert h.judge_gateway is h.gateway
            agent = h.wrap(ReAct(system="s", model="agents/small"), id="geo")
            await h.evaluate(agent, [{"input": "France"}] * 2, [llm_judge("ok?")])
    judge_calls = [c for c in chat.calls if b"strict evaluator" in c.request.content]
    assert len(judge_calls) == 2
    assert all(json.loads(c.request.content)["model"] == "agents/small" for c in judge_calls)
    assert all(c.request.headers["authorization"] == "Bearer agent-key" for c in judge_calls)
    assert caplog.text.count("the judge shares the model of agent geo") == 1


async def test_a_judge_with_no_model_to_ask_is_an_evaluator_failure(harness: Harness) -> None:
    report = await harness.evaluate(
        harness.wrap(capital, id="c"), [{"input": "France"}], [llm_judge("ok?")]
    )
    assert "TRELLIS_JUDGE_MODEL" in report.items[0].failed["llm_judge"]
    async with Harness(config=Settings(judge_model="judges/strong")) as h:
        report = await h.evaluate(
            h.wrap(capital, id="c"), [{"input": "France"}], [llm_judge("ok?")]
        )
    assert "BIFROST_URL" in report.items[0].failed["llm_judge"]


async def test_a_judge_falls_back_to_a_react_agents_own_model_object(harness: Harness) -> None:
    model = JudgeOrAnswer(['{"score": 0.6, "reasoning": "fine"}'])
    agent = harness.wrap(ReAct(system="s", model=model), id="geo")
    report = await harness.evaluate(agent, [{"input": "France"}], [llm_judge("ok?")])
    assert report.summary["llm_judge"].mean == 0.6


# --------------------------------------------------------------------------- online
@pytest.mark.parametrize(("sample", "judged"), [(1.0, 1), (0.0, 0)])
async def test_online_judges_score_sampled_runs_in_the_background(
    sample: float, judged: int
) -> None:
    model = JudgeOrAnswer(['{"score": 0.8, "reasoning": "good"}'])
    scored: list[tuple[Any, ...]] = []
    async with Harness(
        config=Settings(judge_sample=sample), judges=[llm_judge("Helpful?", name="helpful")]
    ) as h:
        h._judge_model = model
        original = h.score

        async def score(*args: Any, **kwargs: Any) -> None:
            scored.append(args)
            await original(*args, **kwargs)

        h.score = score  # type: ignore[method-assign]
        result = await h.wrap(capital, id="c").run("France", user="u")
        await h.writes.drain()
    assert result.status is RunStatus.SUCCESS and len(model.judged) == judged
    assert scored == ([(result.run_id, "helpful", 0.8)] if judged else [])


async def test_judges_default_to_a_tenth_of_runs_and_none_without_judges() -> None:
    async with Harness(config=Settings(), judges=[contains()]) as h:
        assert h.judge_sample == 0.1
    async with Harness(config=Settings(judge_sample=0.5)) as h:
        assert h.judge_sample == 0.5 and h.judges == []


async def test_an_online_judge_that_fails_is_a_warning_not_a_failed_run() -> None:
    async def broken(case: EvalCase) -> EvalScore | None:
        raise RuntimeError("judge down")

    async def no_answer(input: str, agent: Runtime) -> dict[str, str]:
        return {"not": "text"}

    async with Harness(config=Settings(judge_sample=1.0), judges=[broken]) as h:
        events = [e async for e in h.wrap(capital, id="c").stream("France", user="u")]
        await h.writes.drain()
        assert events[-1].type.value == "RUN_FINISHED"
        structured = await h.wrap(no_answer, id="s").run("x", user="u")
        assert structured.status is RunStatus.SUCCESS


async def test_an_online_judge_failure_reaches_the_runs_listeners() -> None:
    from trellis.harness import pipeline

    async def broken(case: EvalCase) -> EvalScore | None:
        raise RuntimeError("judge down")

    seen: list[Any] = []
    async with Harness(config=Settings(judge_sample=1.0), judges=[broken]) as h:
        agent = h.wrap(capital, id="c")
        identity = await agent._opened("France", user="u", thread=None, tenant=None)
        await pipeline.attempt(agent, identity, "France", listener=seen.append)
        await h.writes.drain()
    warnings = [e for e in seen if e.data.get("code") == "judge_failed"]
    assert warnings and "judge broken: RuntimeError: judge down" in warnings[0].data["message"]
