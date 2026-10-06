# Examples

Simple to complex. Every example runs **offline** — a scripted model, a scripted memory service
and gateway, runs kept in process (`examples/_support`) — and against the real services when
their variables are set (`BIFROST_URL`, `MEMORY_URL` with `TRELLIS_API_KEY`, `RUNS_URL`):
the harness, the frameworks and the tools are always real; offline only the model's choices
are written down.

```bash
make examples                                     # all of them, offline, in parallel (~30 s)
make examples-live                                # all of them, with the environment as it is
.venv/bin/python -m examples.02_way1_react.agent  # one (from the repository root)
```

## 01_start: the first ten minutes

| Example | Shows |
|---|---|
| [hello.py](01_start/hello.py) | Level 0: `Harness()`, `h.wrap(fn, id=)`, `agent.run` — memory on as soon as `MEMORY_URL` is set |
| [first_agent.py](01_start/first_agent.py) | a `ReAct` agent with a read tool and an irreversible one: the approval, `agent.resume` |

## 02_way1: one framework each, wrapped

Each `agent.py` shows every piece that applies to its framework: memory (pushed and searched),
local and MCP tools, governance and a person, a hook, `without=`, `timeout=` and
`framework_options=`.

| Example | Shows |
|---|---|
| [02_way1_react/agent.py](02_way1_react/agent.py) | `ReAct`: memory, a local and two MCP tools, a destructive call approved and resumed in place, a `Rewrite` hook, `timeout`, `framework_options`, a run `without={"memory"}` |
| [02_way1_react/structured_output.py](02_way1_react/structured_output.py) | `ReAct(output=Forecast)`: a pydantic answer |
| [02_way1_react/hooks_guardrails.py](02_way1_react/hooks_guardrails.py) | a guardrail (`Deny`, `Ask` on a role, `Rewrite`) and card numbers masked in tool results and before the model reads them |
| [02_way1_function/agent.py](02_way1_function/agent.py) | an async function: `agent.context`, `agent.memory`, an MCP tool and a local one through `agent.tools.call`, `agent.ask` with options, a `Deny` hook, `without=`, `cancel` (`framework_options` does not apply) |
| [02_way1_function/memory_documents_feedback.py](02_way1_function/memory_documents_feedback.py) | `h.add_document` (the next context cites it), `agent.memory.search`, `h.feedback` (pending the tenant administrator) |
| [02_way1_function/queue_worker_inbox.py](02_way1_function/queue_worker_inbox.py) | `agent.start` → a worker → `ask` with a diff → `h.inbox("role:legal")` → `resume` → the worker again |
| [02_way1_function/reliability.py](02_way1_function/reliability.py) | a read retried, a write past its `timeout` (unknown effect, its idempotency key), a run past its `timeout`, `cancel` |
| [02_way1_function/schedule.py](02_way1_function/schedule.py) | `agent.schedule("weekdays", ...)`, fired now (`h.runs.schedules.fire`), run by a worker |
| [02_way1_langgraph/agent.py](02_way1_langgraph/agent.py) | `create_agent` built with `h.tools` (local, MCP and memory tools), no checkpointer: the approval re-run against the journal; a hook, `timeout`, `framework_options`, `without=` |
| [02_way1_langgraph/stategraph.py](02_way1_langgraph/stategraph.py) | a hand-built `StateGraph` with a checkpointer: a harness approval in the tool node and the graph's own `interrupt()`, both resumed in place |
| [02_way1_langgraph/hitl_middleware.py](02_way1_langgraph/hitl_middleware.py) | LangChain's `HumanInTheLoopMiddleware`: an edit, then a reject with a reason |
| [02_way1_deepagents/agent.py](02_way1_deepagents/agent.py) | Deep Agents with `TodoListMiddleware`: a plan, an MCP lookup, an approval resumed in place; a `Deny` hook, `timeout`, `framework_options`, `without=` |
| [02_way1_openai_agents/agent.py](02_way1_openai_agents/agent.py) | an OpenAI Agents `Agent`: harness tools added per run, an MCP lookup replayed on the resume, an `after_tool` audit hook, `max_turns` through `framework_options`, `without={"hints"}` |
| [02_way1_openai_agents/handoff.py](02_way1_openai_agents/handoff.py) | a handoff to a specialist built with `h.tools(framework="openai_agents")` |
| [02_way1_claude/agent.py](02_way1_claude/agent.py) | `ClaudeAgentOptions`: the harness's tools as the `trellis` MCP server, an MCP lookup, an approval, a `Rewrite` hook, `max_turns`, `without=` (a scripted CLI offline) |

## 03_way2: one block each, no wrapping

| Example | Shows |
|---|---|
| [03_way2_memory/memory_block.py](03_way2_memory/memory_block.py) | `trellis.memory`: `bind(...).agent(...)`, `context`, `remember`, `search`, `record_tool`, `history.add`, `feedback` |
| [03_way2_runs/worker_and_webhooks.py](03_way2_runs/worker_and_webhooks.py) | `trellis.runs.Worker` with a handler of your own: queued runs, a pause, the inbox, the resumed run claimed again; a `run.paused` webhook checked with `verify_signature` |
| [03_way2_runs/schedule.py](03_way2_runs/schedule.py) | `runs.schedules.create(ScheduleSpec(...))` with metadata, `fire`, a `Worker` |
| [03_way2_governance/governed_tools.py](03_way2_governance/governed_tools.py) | `Governance.check`, an administrator's `approve_when` in the catalog, `governed(fn, gov, on_ask=, hooks=)`, `Rejected`, `Denied`, `decided` |
| [03_way2_governance/langgraph_recipe.py](03_way2_governance/langgraph_recipe.py) | recipe: a plain LangGraph graph with memory, `governed` tools asking through `interrupt`, the pause in agent-runs answered from the inbox, `Command(resume=)`, a judge |
| [03_way2_governance/openai_agents_recipe.py](03_way2_governance/openai_agents_recipe.py) | recipe: a plain OpenAI Agents `Agent`: `needs_approval` from governance, its `RunState` as the run's checkpoint, the inbox, a judge |
| [03_way2_governance/claude_recipe.py](03_way2_governance/claude_recipe.py) | recipe: a plain Claude Agent SDK `query()`: `can_use_tool` from governance, the session as the checkpoint, the inbox, a judge |
| [03_way2_evals/evaluate_and_judge.py](03_way2_evals/evaluate_and_judge.py) | `evaluate` on a plain function over a dataset, and `judge` on one run of your own |
| [03_way2_a2a/remote.py](03_way2_a2a/remote.py) | `remote(url)` from plain code: a question answered by `on_input`, then by `reply` |
| [03_way2_contracts/records.py](03_way2_contracts/records.py) | the records: `RunEvent`, `RunRecord`, `Interrupt`, `InterruptResolution.resolves`, `to_feedback`, `AgentError`, `classify` |
| [03_way2_gateway/mcp_and_model_headers.py](03_way2_gateway/mcp_and_model_headers.py) | `bifrost_sdk`: the MCP tools with their annotations, `execute_tool` scoped to a server, `NO_GATEWAY_TOOLS` for your own model client |

## 04_no_harness: only the SDKs

Nothing here imports `trellis.harness`: `trellis.memory`, `trellis.runs` and
`trellis.contracts` stitched into your own graph.

| Example | Shows |
|---|---|
| [04_no_harness/langgraph_sdks.py](04_no_harness/langgraph_sdks.py) | your `StateGraph`: memory context and records, your own approval rule (`trellis.memory.approval.evaluate`) through `interrupt`, the run and its pause in agent-runs, the inbox, `Command(resume=)` |
| [04_no_harness/deepagents_sdks.py](04_no_harness/deepagents_sdks.py) | your Deep Agent: memory context in its instructions, its own `interrupt_on` approval kept in agent-runs, resumed with the middleware's decisions |

## 05_features: one feature each

| Example | Shows |
|---|---|
| [hitl_forms_options_own_screen.py](05_features/hitl_forms_options_own_screen.py) | an approval rule in a hook, labelled options with several picks, a form (`form=`, `ui_schema=`), your own screen (`component=`, `props=`), a result from outside the run, `trellis.testing.Reviewer` |
| [langgraph_interrupt_fields.py](05_features/langgraph_interrupt_fields.py) | a graph's own `interrupt({...})` with options, `multiple`, `component`, `props`, `assignee`; the inbox; resumed in place |
| [subagents.py](05_features/subagents.py) | `agent.as_tool()`: a planner calling two agents at once, one asking a person through the planner |
| [skills_and_prompts.py](05_features/skills_and_prompts.py) | prompts from a folder and from code, skills from a `SKILL.md` folder and from code, on a `ReAct` and on a function |
| [sandbox.py](05_features/sandbox.py) | `sandbox()`: a script written and run in the run's own sandbox (Docker with `SANDBOX=docker`, else a provider of the example's own); then the provider governed with no harness |
| [serve_agui_and_a2a.py](05_features/serve_agui_and_a2a.py) | `serve_chat` (AG-UI) and `serve_a2a` on one FastAPI app; another agent calling it with `a2a(url)`, its question asked through the caller |
| [trajectory_evals.py](05_features/trajectory_evals.py) | `h.evaluate`: `exact_match`, `contains`, `llm_judge`, `called`, `tool_sequence` and an evaluator of your own reading the trajectory |
| [online_judges.py](05_features/online_judges.py) | `Harness(judges=[...])`: judges on live runs, in the background |
| [tracing_and_redaction.py](05_features/tracing_and_redaction.py) | the OTel spans of a run, exported to your own provider; a secret dropped and an e-mail masked in their attributes |
| [own_scheduler.py](05_features/own_scheduler.py) | `ReAct` on blocks of your own: your run store, your scheduler loop (`agent.execute`), your governance, memory off |
| [priority_and_concurrency.py](05_features/priority_and_concurrency.py) | `start(priority=)` claimed first; a busy conversation's second message waiting (`concurrency_key`) while others overtake it |

## 06_scenarios: end to end

| Example | Shows |
|---|---|
| [parallel_subagents_mixed_frameworks.py](06_scenarios/parallel_subagents_mixed_frameworks.py) | a `ReAct` planner calling a LangGraph, an OpenAI Agents and a function sub-agent at once; the function's question answered through the planner; the child runs |
| [sql_approval_own_screen.py](06_scenarios/sql_approval_own_screen.py) | SQL that changes data asks `role:dba` on a `sql-review` screen: the inbox, an `edit` that adds a `WHERE`, then an approval remembered for the run |
| [worker_crash_resume.py](06_scenarios/worker_crash_resume.py) | a worker dies after a payment; the lease lapses; another worker finishes the run and the payment is not made twice |
| [scheduled_run_selection_limit.py](06_scenarios/scheduled_run_selection_limit.py) | a schedule with `without=`, `timeout=`, `framework_options=` and `priority=`, kept in its metadata and applied to the run it fires |
| [gateway_code_mode_governed_evals.py](06_scenarios/gateway_code_mode_governed_evals.py) | read-only MCP servers behind Code Mode, a refund behind the catalog's `approve_when`, then `h.evaluate` on the answers and the trajectories |
| [planning_todolist.py](06_scenarios/planning_todolist.py) | opt-in planning: `ReAct(middleware=[TodoListMiddleware()])`, the plan shown by an `after_model` hook |

## What the examples share

`_support/offline.py` picks each piece: the real one when its variable is set, else a scripted
one — `langchain_model`, `openai_agents_model` and `react_model` (the model), `offline_blocks`
(a scripted memory service and gateway for `Harness(**offline_blocks())`), `memory_client`,
`runs_store`, `judge_services`, `claude_cli` (a scripted Claude Code CLI). `_support/memory.py`
and `_support/gateway.py` are the scripted memory service and gateway: in process, in the real
services' shapes, behind the real SDKs.
