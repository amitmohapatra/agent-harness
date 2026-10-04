"""Evaluation against the real services: ``h.evaluate`` over a dataset read from Langfuse,
graded for grounding by the real memory service, and — with a gateway — by a real judge model.
Langfuse is a local stand-in that serves the dataset and records every score and dataset-run
link the harness sends (what reaches Langfuse is the wire format, checked in the offline
suite against its API definitions)."""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from tests.live.conftest import (
    MODEL,
    _env,
    live_harness,
    needs_gateway,
    needs_memory,
)
from tests.live.support import memory_scope
from trellis import Runtime, contains, grounding, llm_judge
from trellis.harness import telemetry

pytestmark = [pytest.mark.live, needs_memory]

#: Each test waits on the memory service (the SDK's live timeout, 60 s) and, for the judge, on
#: a model: well above both.
TIMEOUT_SECONDS = 240


class StubLangfuse:
    """A local Langfuse: one dataset, and every POST recorded."""

    def __init__(self, dataset: str, items: list[dict[str, Any]]) -> None:
        self.received: list[tuple[str, Any]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                url = urlparse(self.path)
                if url.path == f"/api/public/v2/datasets/{dataset}":
                    self._answer({"id": "ds-live", "name": dataset})
                elif url.path == "/api/public/dataset-items":
                    assert parse_qs(url.query)["datasetName"] == [dataset]
                    meta = {"page": 1, "limit": 50, "totalItems": len(items), "totalPages": 1}
                    self._answer({"data": items, "meta": meta})
                else:
                    self._answer({"message": "not found"}, status=404)

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("content-length", 0)))
                stub.received.append((self.path, json.loads(body)))
                self._answer({"id": "ok"})

            def _answer(self, body: Any, status: int = 200) -> None:
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def log_message(self, format: str, *args: Any) -> None:
                return None

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def posted(self, path: str) -> list[Any]:
        return [body for p, body in self.received if p == path]

    def settings(self) -> dict[str, Any]:
        """Langfuse reached through the OTLP headers (no exporter: spans stay in process)."""
        headers = {"authorization": "Basic cGs6c2s=", "x-langfuse-host": self.url}
        return {"otlp_endpoint": None, "otlp_headers": headers}


@pytest.fixture
def langfuse() -> Iterator[StubLangfuse]:
    stub = StubLangfuse(
        "live-capitals",
        [
            {
                "id": "i-1",
                "status": "ACTIVE",
                "input": "Who supplies steel to the Berlin office?",
                "expectedOutput": "Acme Steel",
                "metadata": {},
            },
            {
                "id": "i-2",
                "status": "ACTIVE",
                "input": "What is Acme Steel's supplier id?",
                "expectedOutput": "SUP-40",
                "metadata": {},
            },
        ],
    )
    thread = threading.Thread(target=stub.server.serve_forever, daemon=True)
    thread.start()
    yield stub
    stub.server.shutdown()


async def answer(question: str, agent: Runtime) -> str:
    return "Acme Steel supplies steel to the Berlin office; its supplier id is SUP-40."


@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_a_langfuse_dataset_is_graded_for_grounding_by_the_memory_service(
    langfuse: StubLangfuse,
) -> None:
    suffix = uuid.uuid4().hex[:8]
    user = f"live-eval-{suffix}"
    async with live_harness(grounding_sample=0.0, **langfuse.settings()) as h:
        agent = h.wrap(answer, id=f"live-eval-{suffix}")
        scope = await memory_scope(h, user=user, agent_id=agent.id)
        await scope.remember(
            "The Berlin office reorders steel from Acme Steel, supplier id SUP-40.",
            visibility="USER",
        )
        report = await h.evaluate(
            agent, "live-capitals", [grounding(), contains()], run_name=f"live-{suffix}", user=user
        )
    assert [i.status for i in report.items] == ["success", "success"], report
    assert report.summary["contains"].mean == 1.0
    grounded = report.summary["grounding"]
    assert grounded.count == 2 and grounded.mean is not None and grounded.mean > 0.5, report
    links = langfuse.posted("/api/public/dataset-run-items")
    assert sorted(link["datasetItemId"] for link in links) == ["i-1", "i-2"]
    traces = {telemetry.trace_hex(i.run_id or "") for i in report.items}
    assert {link["traceId"] for link in links} == traces
    scores = langfuse.posted("/api/public/scores")
    assert {(s["name"], s["traceId"]) for s in scores} >= {("grounding", t) for t in traces}


@needs_gateway
@pytest.mark.skipif(
    not (_env("BIFROST_URL") and _env("BIFROST_VIRTUAL_KEY")),
    reason="the judge's model needs BIFROST_URL and BIFROST_VIRTUAL_KEY",
)
@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_a_real_judge_model_grades_the_answers(langfuse: StubLangfuse) -> None:
    suffix = uuid.uuid4().hex[:8]
    judge = llm_judge("The answer names the steel supplier or its id, as the question asks.")
    async with live_harness(
        judge_model=_env("TRELLIS_JUDGE_MODEL") or MODEL, **langfuse.settings()
    ) as h:
        agent = h.wrap(answer, id=f"live-judged-{suffix}")
        report = await h.evaluate(agent, "live-capitals", [judge], user=f"live-judge-{suffix}")
    stats = report.summary["llm_judge"]
    assert stats.failures == 0 and stats.count >= 1, report
    assert stats.mean is not None and 0.0 <= stats.mean <= 1.0
    posted = [s for s in langfuse.posted("/api/public/scores") if s["name"] == "llm_judge"]
    assert posted and all(0.0 <= s["value"] <= 1.0 and s.get("comment") for s in posted)
