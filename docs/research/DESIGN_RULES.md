# Design rules

The standing rules for every harness and `ReAct` change. They sit with the
[improvement backlog](IMPROVEMENT_BACKLOG.md) and the [evaluation protocol](EVAL_PROTOCOL.md).

1. **The framework's own mechanism first.** Each adapter's framework (LangGraph, Deep Agents,
   OpenAI Agents, Claude Agent SDK, Kimi) is used for what it already does, through its own slot.
2. **The harness adds only what is missing,** and adds it at the harness level so every adapter
   gets it. `ReAct` adds in-loop intelligence on top, so it does better than every adapter.
3. **Research-backed, library-first.** Every addition cites its evidence and uses the best
   proven library before any custom code.
4. **Check the code before every change.** Never redo what exists. Reuse the pieces already
   there.
5. **Keep only what passes [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md).** A big-model judge checks the
   answer and the sufficiency of the context. Accuracy comes first, then latency and cost.

## Who does what

The paths are this repository's. *Measuring* marks a row whose choice is still being measured
against the protocol.

| Concern | Adapter-native mechanism | Library we use | What the harness / `ReAct` adds |
|---|---|---|---|
| History, sessions | LangGraph checkpointer `thread_id`; OpenAI Agents `Session`; Claude SDK session resume | the frameworks' own; the memory service's window | the conversation key per tenant, user and thread; the memory window replayed where the framework keeps none (`history.py`, `agent.py`). Memory-off history and session continuity: #64 |
| Compaction, summarization | LangChain `ContextEditingMiddleware` (`ClearToolUsesEdit`); Deep Agents summarization | LangChain, Deep Agents | `FitWindow` fits each request to the window (`middleware.py`); clearing at half the window (`react.py`). *Measuring*: a window sized to the request (`-explain2`) |
| Large results | Deep Agents `FilesystemMiddleware` eviction | Deep Agents | results kept in the journal and paged with `read_file` for OpenAI Agents, Claude and MCP (`tools/results.py`). *Measuring*: every adapter, and a cut that says where the rest is (`-data`) |
| Compute (count, sum, rank, math) | provider code execution (a framework's hosted tools, governed as its own); Deep Agents `execute` | Bifrost Code Mode (Starlark over read-only MCP tools); `sandbox()` (Docker or yours) | Code Mode chosen for read-only servers (`tools/toolbox.py`). *Measuring*, in this order: tool-side aggregation arguments, a read-only query engine over saved results (e.g. DuckDB with external access off), then a sandbox or Code Mode (`-data`, `--classes aggregate`) |
| Tool choice | LangChain `LLMToolSelectorMiddleware` (opt-in) | the memory service's tool hints (hybrid search, learned plans) | narrowing on confident hints (`agent.py` `NARROW_FLOOR`); a lint of the tool set (`tools/lint.py`) |
| Tool chains | none | pydantic `TypeAdapter`, the MCP `outputSchema`; the memory service's learned procedures | links from the tools' schemas, said in the descriptions, a hinted tool's chain kept offered (`tools/links.py`, #65). *Measuring*: a proxy only so far |
| Freshness of live values | none | — | *measuring*: `tool(freshness=, max_age=)`, dated snapshots, a stale tool kept offered (`-data`) |
| Explain record | none (each framework keeps its own trace) | the run journal; the memory window | *measuring*: the run's record in context on later turns (`-explain`, `-explain2`, arms C/D) |
| Steer, interrupt | LangGraph `interrupt()`; Claude SDK interrupt | the frameworks' own | pause and resume in any worker (`RunCheckpointer`); busy-conversation modes (`run(mode=)`, #64) |
| Approvals | `HumanInTheLoopMiddleware`; OpenAI Agents `needs_approval`; Claude permissions | the frameworks' own | one governance for every adapter: risk tiers, the catalog's `approve_when`, failing closed (`governance/`) |
| Retries, limits | LangChain model and tool call limits and retries | tenacity (`tools/base.py`) | read-only retries within the call's time, `StepLimit`, `StallGuard` (`middleware.py`). *Measuring*: LangChain's own limits in `ReAct` (`-reactnative`, #58) |
| Streaming | each framework's stream | AG-UI, A2A | one event stream for every adapter (`events.py`, `agui/`) |
