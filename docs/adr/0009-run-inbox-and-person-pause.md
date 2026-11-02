# 0009. A run's inbox, read at each step: instructions on any worker, and a person's pause

**Status:** Proposed (2026-11-02). Amends [0008](0008-background-subagents.md) ("Limits").

## Context

[ADR 0008](0008-background-subagents.md) gave the model Deep Agents' five task tools. Two of
its limits are lifted here:

- An instruction for a task held by a worker in another process was refused.
- Nobody could pause a run and resume it later.

The goal is Claude Code's behaviour with its agents, which a UI can show and control.

What existed, checked against the code:

| Need | What `main` had | Missing |
|---|---|---|
| An instruction reaching a run | `Turns.give` reaches a run in this process at its framework's step hook (`Adapter.steers`) | a channel between processes |
| A notify mechanism in agent-runs | none: no LISTEN/NOTIFY, no Redis; cancel travels on the heartbeat's lease (`cancel_requested`); event streams poll | — |
| Pause and resume | `Runtime.interrupt`, the one pause: LangGraph's own `interrupt` where the framework suspends; else the attempt ends, and Claude's and Kimi's sessions are continued on the resume (`runs.md`, "Continuing where a run stopped") | a pause a person asks for |
| Kimi between steps | its calls end one by one (`Attempt._seen`); `followup` already stops a turn and goes on in the same session with a new prompt (a rejection's reason) | `steers()` was `False` |

## Decision

1. **The inbox is agent-runs'.** `POST/GET /v1/runs/{id}/messages` stores `RunMessage`
   (contracts 0.6.4): an `INSTRUCTION` or a `PAUSE`, at the next position. `LocalRuns` does the
   same in process.
   - The lease tells a worker the newest position (`Lease.inbox`): the existing channel, not a
     new queue.
   - Only keys that may act for the run's user reach the inbox.
2. **A run reads its inbox at each step** through the hook steering already uses
   (`Runtime.step`):
   - LangChain `abefore_model` (`HarnessTools`);
   - OpenAI Agents' `call_model_input_filter`;
   - Claude's `PostToolUse`;
   - ACP, as each call ends.

   A remote inbox is read when the lease says something came, else at most every 2 s.
   `LocalRuns` is read at every step, which is a list slice. The position taken is in the
   journal, so a resume on any worker reads past it.
3. **An instruction is a steered message** (`Turns.give`, then `Runtime.steered`), so each
   framework takes it as `mode="steer"` does. `update_async_task` now sends it to the inbox, so
   a task on any worker receives it.
4. **A pause is `Runtime.interrupt` with `InterruptReason.PAUSE`**, taken only once answered (a
   resume replays its answer):
   - LangGraph (`ReAct`, `middleware()`): its own `interrupt` in the `before_model` node. The
     checkpoint is kept and the resume is `Command(resume=…)`.
   - Claude and Kimi: the session is stopped and continued on resume.
   - OpenAI Agents: the attempt ends and the run continues from its journal.
   - Several pauses read at once pause the run once. A pause sent to a paused run is a 409.
   - Instructions sent while the run waits are read as it goes on.
5. **Kimi takes instructions now.** When a call ends and something was steered, the turn is
   stopped (`session/cancel`) and the same session goes on with the instruction (`followup`).
   `ACPAdapter.steers` is `True`.
6. **AG-UI controls each run and task:** `POST {path}/runs/{id}/messages`, `/pause`, `/resume`.
   - They reach the user's own runs of the surface's agent and of its background agents;
     anyone else gets 404.
   - A target with no step (a function, a graph without the harness's middleware) is refused
     with 409, saying so.

## Consequences

- **Latency.**
  - A run with no message: one list slice per step in process. With agent-runs, at most one
    `GET` every 2 s per running run at its steps.
  - A run that gets an instruction: no extra model call.
- **Tokens.** None, unless a message is sent.
- **Limits, follow-ups.**
  - A graph without the harness's middleware (`create_agent` or `create_deep_agent` with no
    `middleware()`, a hand-built `StateGraph`) is not paused yet. Pausing it between supersteps,
    with `update_state` on the resume, is a follow-up.
  - Deep Agents' native async sub-agents (a LangGraph server) are not wired.
