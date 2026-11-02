# 0008. Background sub-agents: Deep Agents' five task tools, backed by the run store, the step hooks and the cancel the harness already has

**Status:** Proposed (2026-11-02).

## Context

A sub-agent today is a blocking tool ([subagents.md](../subagents.md)): `ReAct(subagents=[a])`
runs the child inside the parent's call, and the parent's answer waits for it. The goal is what
Claude Code does with its background agents: the model starts sub-agents on its own and the
user keeps chatting; it reports how they are going when asked; it gives one a new instruction,
read at its next step; it stops one on request; their results reach the chat when they are
ready; and only the user's own sub-agents, in their workspace, are reachable.

What existed at `main` (checked by `tests/integration/test_background.py` and the matrix rows
`B01`–`B06`, written first):

| Capability | What `main` had | What was missing |
|---|---|---|
| Start in the background | `agent.start()` inside a run queues a child run (`parent_run_id`, the parent's tenant, user and thread) that a worker works; the parent does not wait | a tool the model calls; nothing tells a task from a blocking child |
| Status | the run store's record (`runs.get`, `RunHandle.status`) | a tool the model calls, scoped to the conversation |
| Steer | `mode="steer"` reaches the run holding a *conversation's* turn at its framework's next step (`Adapter.steers`: LangChain `before_model` on `HarnessTools`, OpenAI Agents `call_model_input_filter`, Claude `PostToolUse`, ACP) | a way to reach a run by its id: a queued child holds no turn |
| Stop | `agent.cancel` (a worker's run at its next heartbeat; in process at once) and `cancel_children` | a tool; and `cancel_children` also killed a task when the turn that started it was cancelled |
| Results into the chat | the child's record and its event log | nothing told the conversation |
| Isolation | the run store is tenant-scoped; AG-UI checks the run's user | nothing user- or conversation-scoped for a model-driven tool; a paused queued child could not be answered at all (`resume` refused any child run) |

The frameworks' own mechanisms, checked against the installed versions (their code, and the
bundled CLIs' own tool definitions):

| Framework (version) | Native mechanism | Of the six | Under the harness |
|---|---|---|---|
| Claude Agent SDK 0.2.160, its CLI 2.1.283 | `Agent` tool with `run_in_background`, `TaskOutput`, `TaskStop`, peer messages; `AgentDefinition(background=True)`; `TaskStarted/Progress/Updated/NotificationMessage`; `ClaudeSDKClient.stop_task` | all six, inside one live CLI session | no: the adapter runs one `query()` per attempt, so the CLI process (and every task it holds) ends with the attempt; `stop_task` "only works with streaming mode"; the CLI says "Only synchronous subagents are supported" where it cannot keep them. Its synchronous subagents stay native (`bridges.helpers`) |
| Deep Agents 0.7.19 | `task` (synchronous); `AsyncSubAgentMiddleware`: `start/check/update/cancel/list_async_task` over a LangGraph Agent Protocol server (`langgraph_sdk`) | start, status, steer (interrupt and re-run on the task's thread), stop, list; results only when asked; isolation is the server's | only with a LangGraph server, which the harness does not run (no `langgraph_api` installed; agent-runs is the run store). A graph that brings its own middleware and server keeps it |
| LangGraph 1.2.11 (library), `langgraph_sdk` 0.4.4 (Platform) | library: none. Platform: background runs (`runs.create`), double-texting (`multitask_strategy`: reject, rollback, interrupt, enqueue), `runs.cancel`, `runs.join` | Platform: start, status, stop, steer as interrupt-and-restart | the Platform is a server the harness does not run; the library has the step hooks the harness steers through (`HarnessTools.before_model`) |
| OpenAI Agents 0.22.3 | `Agent.as_tool`, handoffs, `Runner.run_streamed` | none in the background: a tool call blocks; a handoff continues the same run | agents-as-tools stay native child runs (`bridges.helpers`); `call_model_input_filter` is the step hook the harness steers through |
| Kimi Code 2.1.1 over ACP | `Agent` with `run_in_background`, `TaskOutput`, `TaskStop`, background-task listing, `WaitFor` | start, status, stop, results within one session | no: one `kimi acp` process per attempt (`adapters/acp.py`), so its background tasks end with the attempt. Its synchronous `Agent` and `AgentSwarm` stay native child runs |

So no framework's background mechanism outlives a harness run on the frameworks the harness
runs in process per attempt, and the two that keep tasks durably (Deep Agents' async
sub-agents, LangGraph Platform) need a server that is not the harness's run store. What every
framework does have, and the harness already uses, is a hook between two of its steps.

## Decision

1. **The model-facing part is Deep Agents' own.** `background(*agents)` is a tool source with
   Deep Agents' five tools: the same names, the same arguments, its usage notes as the start
   tool's description. A model trained or prompted on that vocabulary reads it unchanged. It
   is opt-in: an agent without it is offered nothing new and runs exactly as before.
2. **Every tool is backed by what the harness already has; nothing new is stored.**
   * `start_async_task` queues the sub-agent's run in the run store as the calling run's
     child (`Agent._start(parent=...)`, the existing path of `agent.start()` inside a run),
     its id derived from the call (a re-run after a crash finds it again). A worker works it,
     so a worker that dies under it is replaced by the lease, as for any queued run.
   * `check_async_task` and `list_async_tasks` read the run store.
   * `update_async_task` gives the instruction to the task's run by its id
     (`Turns.give`); the framework's step hook that `mode="steer"` already uses takes it
     (`Runtime.steered`). Nothing new per adapter.
   * `cancel_async_task` is `agent.cancel`.
3. **A task is its user's, in their tenant, and outlives the turn.** A tool given a task's id
   finds it only when the task is the calling run's user's (the run store reads only the
   tenant's runs); another user's — even on a thread named alike — or another tenant's is "no
   task". Its record keeps the conversation key that started it (`metadata["background"]`),
   which lists it and is told of it. Cancelling the turn that started a task leaves it working
   (`cancel_children` skips it). A task that asks a
   person is answered on its own run (nobody relays it): `resume` no longer refuses it.
4. **Its result reaches the conversation twice over.** When it ends in a process where a turn
   of its conversation is working, that turn reads it at its next step (the same step hook).
   Every later turn is told, as it starts, of its conversation's tasks: each one's status, and
   each ended one's answer (`## Background tasks of this conversation`), read from the run
   store beside the memory context (`asyncio.gather`, so the turn does not wait longer).

## Consequences

* **Latency.** An agent without `background(...)` takes the old path (no extra call, no
  gather). With it, a turn's start reads the run store once per sub-agent type, beside the
  memory context; a start returns after one run-store write.
* **Tokens.** The five tool definitions, and the note while tasks exist (each answer cut at
  2,000 characters; `check_async_task` gives it whole).
* **Workers.** Tasks are queued runs: a deployment runs a worker for the sub-agents
  (`h.worker([research])`), as for any `start()`.
* **Limits.** An instruction reaches a task that works, or is queued, in this process; for a
  task held by a worker in another process the tool says it cannot reach it (cancel and start
  again). Reaching it there needs a channel in agent-runs (a run's pending messages, read at
  each heartbeat): a follow-up, not done here. A target with no step the harness reaches (a
  function, a graph without the harness's middleware) takes no instruction, and the tool says
  so. An idle chat is told a result at its next message, not pushed one: AG-UI streams a run,
  not a thread.
