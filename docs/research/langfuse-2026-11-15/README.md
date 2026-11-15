# Live check: every step of a run as Langfuse ingests it (2026-11-15)

`live_check.py` ran `ReAct` and Claude Code (Claude Agent SDK 0.2.182, the real CLI), both on
`claude-haiku-5-5` with the provider's key, through the harness on branch
`claude/cool-maxwell-w5avo8-lfuse`. Spans went over OTLP/HTTP (`OTEL_EXPORTER_OTLP_ENDPOINT`)
to an OpenTelemetry collector (`otel/opentelemetry-collector-contrib:0.115.1`) whose file
exporter wrote them; the script read them back. A local Langfuse was not used: v3 needs
ClickHouse, Redis, MinIO and Postgres beside it, more than this machine's shared disk had
(under 3 GB). The collector receives exactly what Langfuse's `/api/public/otel` would.

Spend: Claude $0.0041 (as the CLI reported it), ReAct about $0.0002 (3,832 tokens, 1,841
cached, at `benchmark.cost.prices`): $0.0043 in all, cap $1.

What the collector got (span, its parent, key attributes):

```
ReAct: SUCCESS answer='Order O-4471 shipped on November 2, 2026.'

## ReAct: 4 spans
- invoke_agent live-react  (under -)  {'langfuse.observation.type': 'agent', 'trellis.run.status': 'SUCCESS', 'langfuse.trace.metadata.spent_cost': 0}
- chat claude-haiku-5-5  (under invoke_agent live-react)  {'langfuse.observation.type': 'generation', 'gen_ai.usage.input_tokens': '1845', 'gen_ai.usage.output_tokens': '50'}
- execute_tool order_status  (under invoke_agent live-react)  {'langfuse.observation.type': 'tool', 'trellis.governance.action': 'run'}
- chat claude-haiku-5-5  (under invoke_agent live-react)  {'langfuse.observation.type': 'generation', 'gen_ai.usage.input_tokens': '1916', 'gen_ai.usage.output_tokens': '21', 'gen_ai.usage.cache_read.input_tokens': '1841'}
spent: {'trellis.spent.tokens': '3832', 'trellis.spent.cached_tokens': '1841', 'trellis.spent.cache_written_tokens': '0', 'trellis.spent.cost': 0, 'trellis.spent.model_calls': '2', 'trellis.spent.tool_calls': '1'}

Claude Code: SUCCESS answer='The order id named in notes.txt is **O-4471**.'

## Claude Code: 6 spans
- invoke_agent live-claude  (under -)  {'langfuse.observation.type': 'agent', 'trellis.run.status': 'SUCCESS', 'langfuse.trace.metadata.spent_cost': 0.00406743}
- initialize  (under invoke_agent live-claude)  {}
- notifications/initialized  (under invoke_agent live-claude)  {}
- tools/list  (under invoke_agent live-claude)  {}
- execute_tool Read  (under invoke_agent live-claude)  {'langfuse.observation.type': 'tool', 'trellis.governance.action': 'run'}
- chat claude-haiku-5-5  (under invoke_agent live-claude)  {'langfuse.observation.type': 'generation', 'gen_ai.usage.input_tokens': '38188', 'gen_ai.usage.output_tokens': '93', 'gen_ai.usage.cache_read.input_tokens': '19033', 'gen_ai.usage.cost': 0.00406743}
spent: {'trellis.spent.tokens': '38281', 'trellis.spent.cached_tokens': '19033', 'trellis.spent.cache_written_tokens': '19151', 'trellis.spent.cost': 0.00406743, 'trellis.spent.model_calls': '2', 'trellis.spent.tool_calls': '0'}
```

Both traces have the attempt span (its outcome and spend, Claude's cost on the trace's
metadata), a `chat` generation per model call with usage, cache tokens and cost (Claude's one
per query result, `trellis.model_calls` 2), and an `execute_tool` span for every tool call —
for Claude, its built-in `Read`, which no span showed before this branch. The `initialize`,
`tools/list` spans are the MCP SDK's own instrumentation of the harness's in-process server.
