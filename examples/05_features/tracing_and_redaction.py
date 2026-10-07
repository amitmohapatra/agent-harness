"""Tracing and redaction: the OTel GenAI spans every run makes, and what never leaves the process.

Every attempt of a run is an ``invoke_agent`` span in one trace per run, with an
``execute_tool`` span per tool call (and ``chat`` per model call of a ``ReAct``). Span
attributes pass the redactor first: a value under a secret-looking name is dropped, a value
that looks like a credential is dropped, an e-mail address is masked, a long value is cut.

The harness uses the OTel API only. ``OTEL_EXPORTER_OTLP_ENDPOINT`` (a collector's), or
Langfuse's own keys alone, install an exporter; here the example installs its own in-memory
one first — the harness then leaves it alone — and prints what was exported.

    python -m examples.05_features.tracing_and_redaction
"""

from __future__ import annotations

import asyncio

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from trellis import Harness, Runtime, tool

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)  # yours: the harness exports to it


@tool(side_effects="write")
def notify(email: str, api_key: str, text: str) -> str:
    """Send a notification through the mail provider."""
    return f"sent to {email}"


async def mailer(input: str, agent: Runtime) -> str:
    return await agent.tools.call(
        "notify", email="ada@example.com", api_key="example-not-a-real-key", text=input
    )


async def main() -> None:
    async with Harness() as h:
        agent = h.wrap(mailer, id="mailer", tools=[notify])
        result = await agent.run("Your order shipped.", user="ada", thread="orders")
        print(result.status.value, result.answer)
    for span in exporter.get_finished_spans():
        attributes = span.attributes or {}
        shown = {k: v for k, v in attributes.items() if k.startswith(("gen_ai.tool", "langfuse"))}
        print(span.name, shown)


if __name__ == "__main__":
    asyncio.run(main())
