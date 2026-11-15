"""Live check: ReAct and Claude Code (Haiku) through the harness, their spans exported over
OTLP/HTTP to an OpenTelemetry collector whose file exporter writes them (a local Langfuse
needs ClickHouse, Redis, MinIO and Postgres beside it: too heavy for this machine's disk).
Reads the file back and checks every step is a span, as Langfuse would ingest it.

    OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:43180 ANTHROPIC_API_KEY=... \\
        python docs/research/langfuse-2026-11-15/live_check.py <collector out dir> <work dir>
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions

from tests.support.arm import arm
from trellis import Harness, ReAct, tool
from trellis.harness import telemetry

MODEL = "claude-haiku-5-5"
#: US dollars per million tokens (benchmark.cost.prices)
PRICE = {"input": 0.10, "cached": 0.01, "write": 0.125, "output": 0.50}
OUT, WORK = Path(sys.argv[1]), Path(sys.argv[2])


@tool(side_effects="read")
def order_status(order: str) -> str:
    """The status of an order."""
    return f"{order}: shipped on 2026-11-02"


def spans_of(trace_hex: str) -> list[dict[str, Any]]:
    found = []
    for line in (OUT / "traces.jsonl").read_text().splitlines():
        for resource in json.loads(line).get("resourceSpans", []):
            for scope in resource.get("scopeSpans", []):
                for span in scope.get("spans", []):
                    if span["traceId"] == trace_hex:
                        attrs = {a["key"]: next(iter(a["value"].values())) for a in span.get("attributes", [])}
                        found.append({**span, "attrs": attrs})
    return found


def show(title: str, spans: list[dict[str, Any]]) -> None:
    by_id = {s["spanId"]: s for s in spans}
    print(f"\n## {title}: {len(spans)} spans")
    for s in sorted(spans, key=lambda s: int(s["startTimeUnixNano"])):
        parent = by_id.get(s.get("parentSpanId", ""), {}).get("name", "-")
        a = s["attrs"]
        keep = {
            k: a[k]
            for k in (
                "langfuse.observation.type",
                "trellis.governance.action",
                "gen_ai.usage.input_tokens",
                "gen_ai.usage.output_tokens",
                "gen_ai.usage.cache_read.input_tokens",
                "gen_ai.usage.cost",
                "trellis.run.status",
                "langfuse.trace.metadata.spent_cost",
            )
            if k in a
        }
        print(f"- {s['name']}  (under {parent})  {keep}")


async def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "notes.txt").write_text("The order to chase is O-4471.\n")
    async with Harness() as h:
        target = ReAct(system_prompt="Answer briefly.", model=MODEL)
        react = arm(h.wrap(target, id="live-react"), [order_status])
        one = await react.run("What is the status of order O-4471?", user="live")
        claude = h.wrap(
            ClaudeAgentOptions(
                model=MODEL,
                setting_sources=[],
                cwd=str(WORK),
                allowed_tools=["Read", "Grep", "Glob"],
                env={"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
            ),
            id="live-claude",
        )
        two = await claude.run(
            "Use the Read tool to read notes.txt and tell me the order id it names.", user="live"
        )
    await asyncio.sleep(3)  # the collector's batch
    for title, result in (("ReAct", one), ("Claude Code", two)):
        print(f"\n{title}: {result.status.value} answer={str(result.answer)[:120]!r}")
        spans = spans_of(telemetry.trace_hex(result.run_id))
        show(title, spans)
        names = [s["name"] for s in spans]
        assert any(n.startswith("invoke_agent ") for n in names), names
        assert any(n.startswith("execute_tool ") for n in names), names
        assert any(n.startswith("chat ") for n in names), names
        ended = [s["attrs"] for s in spans if s["name"].startswith("invoke_agent live-")][-1]
        print("spent:", {k: v for k, v in ended.items() if k.startswith("trellis.spent.")})
    claude_spans = spans_of(telemetry.trace_hex(two.run_id))
    assert "execute_tool Read" in [s["name"] for s in claude_spans]


asyncio.run(main())
