# 0006. The intelligent harness: what the model gets, so a small model gives a big model's answers

**Status:** Accepted as the direction (2026-10-30). The parts marked *proposed* below ship only
after they pass the [evaluation protocol](../research/EVAL_PROTOCOL.md), each tracked in the
[improvement backlog](../research/IMPROVEMENT_BACKLOG.md).

## Context

What a model needs to answer is spread over the memory context, the history, results the run
saved and tools that can fetch more. A small model cannot search for what it has not been told
exists. A large one wastes its window when everything is pushed at it. The
[scenario coverage audit](../research/SCENARIO_COVERAGE_AUDIT.md) found the gaps (follow-ups,
undated results, large results, aggregation, explaining earlier answers). The
[design rules](../research/DESIGN_RULES.md) say how each gap is closed.

## Decision

**The goal.** On business questions (retrieve, compute, chain tools, converse, explain), a small
model with the harness matches a big-model reference within a small margin. The gap is measured
per category: a big-model judge writes the reference answer from the same inputs.

**The layers.** Each layer has one owner, and the adapters' own mechanisms come first
([DESIGN_RULES.md](../research/DESIGN_RULES.md)).

| Layer | What it is | Status |
|---|---|---|
| Models | `LLM_*` routes, and a model profile (window, size) that drives density and behaviour | exists ([ADR 0004](0004-provider-neutral-models.md)) |
| Context engine | §1 below | partly exists, see §1 |
| Tools | §2 below | partly exists, see §2 |
| Reasoning scaffold (`ReAct`, per call) | a plan for a long task (`write_todos`); errors told in the harness's own words (`tools.base.not_run`); the stall guard and the step limit | exists. *Proposed*: decomposing multi-part questions (Plan-and-Solve, least-to-most); selective self-consistency, used only when the evidence conflicts or a number is unverified, and capped in cost (backlog: ReAct) |
| Verification | deterministic and cheap: every number traced to a tool or query result, and every tool and source named present in the run record (CRITIC-style checking without an extra model call) | *proposed* (backlog: in-run verification). The model-based answer check stays removed: it measured worse (#56) |
| Record and explain | the journal and one id per message (#64); steps lists and fetch times | the journal and the message ids exist. Steps lists are in flight (`-explain2`); see §4 |
| Learning | memory facts and preferences, tool statistics, learned plans (successful paths only) | exists (memory service, its ADR 0018) |
| Evaluation | the big-model judge (is the answer correct, is the context sufficient) and the keep rule | the protocol exists ([EVAL_PROTOCOL.md](../research/EVAL_PROTOCOL.md)). *Proposed*: the complex suite with its edge cases (backlog) |

**Libraries.**

- LangChain `create_agent` and its middleware, the LangGraph checkpointer, Deep Agents (files,
  summarization, sub-agents);
- the memory service (retrieval, facts, plans);
- bifrost-sdk (the OpenAI-compatible client, auth strategies);
- pydantic for schemas;
- *proposed*: DuckDB for compute, or stdlib `sqlite3` if it measures as sufficient (`-data`);
  llama.cpp grammars or XGrammar for local models' tool calls (backlog).

**Not doing.**

- our own reasoning engine;
- a second history store;
- model-based answer checks;
- rules keyed on how a question is worded;
- escalating to a bigger model by default (only as an opt-in capability the user enables).

**The honest limit.** The harness closes gaps of information, computation, procedure and memory.
Deep novel reasoning is bound by the model. Those cases are reported as model gaps, not hidden.

**What the model gets.** The context holds a map, and the details come on demand.

The harness never guesses what kind of question it has. What it offers follows from three things, all of them data:

- what the run actually holds;
- what the deployment can do (capabilities, such as `gateway.has()`);
- the model's profile (its window and size).

A small window or a small model gets more pushed facts, steps lists and structured table
operations. A large one gets the map and the cards, fetches more on demand, and uses SQL or code.

### 1. Context: always, fitted to a budget computed from the model's window

| Block | Content | Status (checked against the code) |
|---|---|---|
| Fixed prefix (cached) | the system prompt and the tools, in a stable order | exists (tools offered in a stable order, `react.py`, [tools.md](../tools.md#tool-hints-what-the-model-is-offered)). The memory context still sits before the history, which breaks the cross-turn cache (audit K1, backlog) |
| Data map | one short block: this chat (N turns, steps available), saved results (names), memory (facts and preferences present), past chats (searchable), tool areas | new (small) |
| Memory context | facts and preferences for this question, evidence status, tool hints | exists (`agent.py` `remembered`). Expanding a follow-up query with recent turns is new (audit A1). Tool chains are said in the tools' descriptions, not in this block (`tools/links.py`, #65) |
| History | recent turns whole; older turns as steps lists (tool, arguments, source, fetch time, status); the chat summary | the summary exists (memory service). Steps lists are in flight, not on main (`-explain2`, arm D) |
| Data cards | for each saved result: source and fetch time, rows, column types, sample rows, ranges, likely join keys | new (`-data`) |
| Freshness notes | "snapshot from <time>; call the live tool again" | in flight (`-data`) |

### 2. Built-in tools: offered only when their condition holds, never narrowed away

| Tool | Offered when | Status (checked against the code) |
|---|---|---|
| user and MCP tools | narrowed by the hints (confidence 0.5 or more), else all of them with `tool_search` | exists (`agent.py` `NARROW_FLOOR`); a hinted tool's chain stays offered (#65) |
| `tool_search` | many tools and no confident hint | exists (memory pull tool; the harness answers it, `clients/memory.py`) |
| `memory_search` (kinds memory, chunk, summary, episode, message; a time range) | memory is on: facts, this chat, past chats | exists (memory service `agent_tools/service.py`) |
| `memory_remember`, `memory_update`, `memory_forget` | memory is on | exists. `memory_forget` forgets only what the agent wrote. It is a write, announced and not asked by default. Forgetting what the user said, asked in chat, is in the backlog (audit H4) |
| read a saved result (by page) | the run holds a saved large result | exists for OpenAI Agents, Claude and MCP (`tools/results.py`), and for `ReAct` with a backend (Deep Agents). Plain `ReAct` is in flight (`-data`) |
| query saved results (read-only SQL; structured table operations for small models later) | the run holds saved results | in flight (`-data`), still being measured |
| `execute`, Code Mode | a sandbox or Code Mode is configured | exists (`sandbox()`; `tools/toolbox.py`) |
| provider code execution | the provider offers it and the data policy allows it | backlog |
| `write_todos` | a long-task signal and a window of 32k or more | exists (`automatic.py`, `TodoListMiddleware`) |
| `ask_user` | the request is ambiguous (the model decides) | exists as an opt-in tool (`trellis.ask_user()`: options, a date, a date range, a number); offering it by default is in flight (#58, being measured) |

### 3. Per call (`ReAct` only, in the loop)

Exists:

- budget fitting and `FitWindow`;
- clearing old results;
- the stall guard;
- a call repeated within one step runs once (#52).

Backlog:

- answering a call repeated in a later step from a cache;
- grammar-constrained tool calls for local models;
- a cheap check of the tool and source names an answer claims against the run record (measured
  first).

### 4. Explain

Answers about an earlier answer come from the run record (steps, sources, fetch times,
approvals, message ids), never from the model's own account. "Why" and "how sure" are derived
from the evidence (backlog; [SELF_EXPLANATION.md](../research/SELF_EXPLANATION.md)).

## Consequences

- A new block or tool ships only when [EVAL_PROTOCOL.md](../research/EVAL_PROTOCOL.md) passes on
  the complex suite, with a large model as judge, and with a small local model when one is
  available:
  - the context-sufficiency rate goes up;
  - accuracy does not drop;
  - the made-up rate does not rise;
  - cost is not materially up.
- Each part is added at the harness level, so every adapter gets it. Per-call logic stays in
  `ReAct` ([DESIGN_RULES.md](../research/DESIGN_RULES.md)).
- The rows marked new, in flight or backlog are tracked in the
  [improvement backlog](../research/IMPROVEMENT_BACKLOG.md).
