"""An evaluated run is an item of a Langfuse experiment, as Langfuse's own SDK makes one: the
dataset run linked through ``dataset-run-items`` (Langfuse v3) *and* the experiment attributes
on the run's spans (``langfuse.experiment.*``, what Langfuse v4 reads) — for a wrapped agent's
runs and for a callable's — checked on the spans an in-memory exporter receives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Settings, tool
from trellis.harness import telemetry
from trellis.harness.evals import EvalItem, EvalServices, evaluate, exact_match

LF = "https://lf.test"
OTLP = {"authorization": "Basic cGs6c2s=", "x-langfuse-host": LF}
PREFIX = "langfuse.experiment."


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("t"))
    yield exporter


@tool(side_effects="read")
def capital_of(country: str) -> str:
    """The capital of a country."""
    return {"France": "Paris", "Italy": "Rome"}.get(country, "unknown")


def geo_agent(h: Harness, turns: int) -> Any:
    script: list[Any] = []
    for country, city in [("France", "Paris"), ("Italy", "Rome")][:turns]:
        script += [("capital_of", {"country": country}), city]
    model = ScriptedChat(script)
    return h.wrap(ReAct(system="Capitals.", model=model), id="geo", tools=[capital_of])


def experimental(span: ReadableSpan) -> dict[str, Any]:
    attributes = dict(span.attributes or {})
    return {
        k: v for k, v in attributes.items() if k.startswith(PREFIX) or k == "langfuse.environment"
    }


def of_run(exporter: InMemorySpanExporter, run_id: str) -> list[ReadableSpan]:
    trace_id = telemetry.trace_id_of(run_id)
    return [
        s for s in exporter.get_finished_spans() if s.context and s.context.trace_id == trace_id
    ]


def root_of(found: list[ReadableSpan]) -> ReadableSpan:
    [root] = [s for s in found if s.name.startswith("invoke_agent")]
    return root


def langfuse_dataset(routes: respx.MockRouter, *, link: httpx.Response) -> respx.Route:
    routes.get(f"{LF}/api/public/v2/datasets/capitals").mock(
        return_value=httpx.Response(
            200, json={"id": "ds-1", "name": "capitals", "projectId": "proj-1"}
        )
    )
    items = [
        {
            "id": "item-1",
            "status": "ACTIVE",
            "input": "France",
            "expectedOutput": "Paris",
            "metadata": {"level": "easy", "tags": {"region": "west", "rank": 1}},
        },
        {
            "id": "item-2",
            "status": "ACTIVE",
            "input": "Italy",
            "expectedOutput": {"city": "Rome"},
            "metadata": {"long": "x" * 201},
        },
    ]
    meta = {"page": 1, "limit": 50, "totalItems": 2, "totalPages": 1}
    routes.get(f"{LF}/api/public/dataset-items").mock(
        return_value=httpx.Response(200, json={"data": items, "meta": meta})
    )
    routes.post(f"{LF}/api/public/scores").mock(return_value=httpx.Response(200, json={}))
    return routes.post(f"{LF}/api/public/dataset-run-items").mock(return_value=link)


@respx.mock
async def test_a_langfuse_dataset_run_is_linked_and_on_every_span_of_its_runs(
    spans: InMemorySpanExporter,
) -> None:
    linked = langfuse_dataset(
        respx.mock,
        link=httpx.Response(200, json={"id": "dri-1", "datasetRunId": "dsr-1"}),
    )
    async with Harness(config=Settings(otlp_headers=OTLP)) as h:
        report = await h.evaluate(
            geo_agent(h, 2),
            "capitals",
            [exact_match()],
            run_name="nightly",
            description="the nightly capitals run",
            metadata={"model": "small", "params": {"temperature": 0}},
            concurrency=1,
        )
    assert report.experiment_id == "dsr-1"
    assert report.dataset_run_url == f"{LF}/project/proj-1/datasets/ds-1/runs/dsr-1"
    assert str(report).startswith(f"nightly: 2 items (2 success) — {report.dataset_run_url}")
    bodies = [json.loads(c.request.content) for c in linked.calls]
    assert bodies[0] == {
        "runName": "nightly",
        "datasetItemId": "item-1",
        "traceId": telemetry.trace_hex(report.items[0].run_id or ""),
        "metadata": {"agent_id": "geo", "model": "small", "params": {"temperature": 0}},
        "runDescription": "the nightly capitals run",
    }

    first, second = (of_run(spans, i.run_id or "") for i in report.items)
    root = root_of(first)
    root_id = format(root.context.span_id, "016x")  # type: ignore[union-attr]
    shared = {
        "langfuse.experiment.id": "dsr-1",
        "langfuse.experiment.name": "nightly",
        "langfuse.experiment.dataset.id": "ds-1",
        "langfuse.experiment.item.id": "item-1",
        "langfuse.experiment.item.root_observation_id": root_id,
        "langfuse.experiment.metadata.agent_id": "geo",
        "langfuse.experiment.metadata.model": "small",
        "langfuse.experiment.metadata.params.temperature": "0",
        "langfuse.experiment.item.metadata.level": "easy",
        "langfuse.experiment.item.metadata.tags.region": "west",
        "langfuse.experiment.item.metadata.tags.rank": "1",
        "langfuse.environment": "sdk-experiment",
    }
    assert experimental(root) == {
        **shared,
        "langfuse.experiment.description": "the nightly capitals run",
        "langfuse.experiment.item.expected_output": "Paris",
    }
    children = [s for s in first if s is not root]
    names = {s.name.split(" ")[0] for s in children}
    assert {"chat", "execute_tool", "score"} <= names
    assert all(experimental(s) == shared for s in children)

    second_root = root_of(second)
    attributes = experimental(second_root)
    assert attributes["langfuse.experiment.item.id"] == "item-2"
    assert attributes["langfuse.experiment.item.expected_output"] == '{"city": "Rome"}'
    assert "langfuse.experiment.item.metadata.long" not in attributes  # over 200: dropped
    assert attributes["langfuse.experiment.item.root_observation_id"] == format(
        second_root.context.span_id,  # type: ignore[union-attr]
        "016x",
    )


@respx.mock
async def test_a_refused_link_still_marks_the_spans_with_one_fallback_id(
    spans: InMemorySpanExporter, caplog: pytest.LogCaptureFixture
) -> None:
    langfuse_dataset(respx.mock, link=httpx.Response(404, json={"message": "gone in v4"}))
    async with Harness(config=Settings(otlp_headers=OTLP)) as h:
        report = await h.evaluate(geo_agent(h, 2), "capitals", [], run_name="v4", concurrency=1)
    assert "was not linked to dataset run v4" in caplog.text
    assert report.dataset_run_url is None
    fallback = report.experiment_id
    assert fallback is not None and len(fallback) == 16
    roots = [experimental(root_of(of_run(spans, i.run_id or ""))) for i in report.items]
    assert [r["langfuse.experiment.id"] for r in roots] == [fallback] * 2  # one per evaluation
    assert all(r["langfuse.experiment.dataset.id"] == "ds-1" for r in roots)
    assert all("langfuse.experiment.description" not in r for r in roots)


@respx.mock
async def test_a_callables_items_are_linked_and_on_every_span_of_their_traces(
    spans: InMemorySpanExporter,
) -> None:
    linked = langfuse_dataset(
        respx.mock, link=httpx.Response(200, json={"id": "dri-1", "datasetRunId": "dsr-1"})
    )

    async def capitals(country: str) -> str:
        with telemetry._tracer.start_as_current_span("my framework's step"):
            return {"France": "Paris"}.get(country, "Rome")

    langfuse = telemetry.Langfuse.of(Settings(otlp_headers=OTLP))
    async with EvalServices(langfuse=langfuse) as services:
        report = await evaluate(
            capitals,
            "capitals",
            [exact_match()],
            services=services,
            user="qa",
            run_name="plain",
            concurrency=1,
        )
    assert report.experiment_id == "dsr-1" and report.statuses == {"success": 2}
    first = report.items[0]
    assert first.trace_url == f"{LF}/trace/{telemetry.trace_hex(first.run_id or '')}"
    assert json.loads(linked.calls[0].request.content)["metadata"] == {"agent_id": "capitals"}

    found = of_run(spans, first.run_id or "")
    root = root_of(found)
    attributes = dict(root.attributes or {})
    assert root.name == "invoke_agent capitals"
    assert attributes["user.id"] == "qa" and attributes["trellis.run_id"] == first.run_id
    assert attributes["langfuse.observation.input"] == "France"
    assert attributes["langfuse.observation.output"] == "Paris"
    assert experimental(root)["langfuse.experiment.item.root_observation_id"] == format(
        root.context.span_id,  # type: ignore[union-attr]
        "016x",
    )
    assert experimental(root)["langfuse.experiment.item.expected_output"] == "Paris"
    names = {s.name for s in found if s is not root}
    assert names == {"my framework's step", "score exact_match"}
    score = next(s for s in found if s.name == "score exact_match")
    assert experimental(score)["langfuse.experiment.id"] == "dsr-1"  # its own: experimental
    step = next(s for s in found if s.name == "my framework's step")
    assert step.parent is not None and step.parent.span_id == root.context.span_id  # type: ignore[union-attr]
    assert experimental(step) == {}  # a framework's own span is its child, unmarked


async def test_a_local_dataset_is_an_experiment_with_hashed_item_ids(
    spans: InMemorySpanExporter,
) -> None:
    async with Harness(config=Settings()) as h:
        report = await h.evaluate(
            geo_agent(h, 1),
            [
                {"input": "France", "expected": ["Paris"]},
                EvalItem(input={"country": "Italy"}, id="mine"),
            ],
            [],
            run_name="local",
            concurrency=1,
        )
    assert report.dataset_run_url is None and report.dataset is None
    first, second = (experimental(root_of(of_run(spans, i.run_id or ""))) for i in report.items)
    assert first["langfuse.experiment.id"] == second["langfuse.experiment.id"]
    assert first["langfuse.experiment.id"] == report.experiment_id
    assert first["langfuse.experiment.item.id"] == hashlib.sha256(b"France").hexdigest()[:16]
    assert first["langfuse.experiment.item.expected_output"] == '["Paris"]'
    assert "langfuse.experiment.dataset.id" not in first
    assert second["langfuse.experiment.item.id"] == "mine"
    assert "langfuse.experiment.item.expected_output" not in second


def test_before_its_root_span_an_item_names_no_root_and_no_environment() -> None:
    item = telemetry.Experiment(id="e", name="n", item_id="i", description="d")
    assert item.propagated() == {
        "langfuse.experiment.id": "e",
        "langfuse.experiment.name": "n",
        "langfuse.experiment.item.id": "i",
    }
    assert item.root()["langfuse.experiment.description"] == "d"


def test_an_item_id_hashes_the_input_as_the_sdk_serializes_it() -> None:
    from trellis.harness.evals import item_id_of

    digest = hashlib.sha256(json.dumps({"a": [1, 2]}).encode()).hexdigest()[:16]
    assert item_id_of(EvalItem(input={"a": [1, 2]})) == digest
    assert item_id_of(EvalItem(input=None)) == hashlib.sha256(b"null").hexdigest()[:16]
    assert telemetry.flattened(None) == {}
    assert telemetry.flattened({"a": None, "b": {"c": "d"}}) == {"b.c": "d"}


async def test_a_run_outside_an_evaluation_carries_no_experiment(
    spans: InMemorySpanExporter,
) -> None:
    async with Harness(config=Settings()) as h:
        result = await geo_agent(h, 1).run("France", user="u")
    assert all(experimental(s) == {} for s in of_run(spans, result.run_id))
