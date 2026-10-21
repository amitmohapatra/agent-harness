# 0003. ReAct turns native features on from signals, never from new settings

**Status:** Accepted (2026-10-21)

## Context

LangChain, LangGraph and Deep Agents ship the pieces a long or large agent run needs —
planning (`TodoListMiddleware`), scratch files and helpers (Deep Agents' `FilesystemMiddleware`
and `SubAgentMiddleware`), tool selection, fallbacks, retries, call limits, prompt caching —
each off unless a team adds it. A team that adds them all pays for them on every call: the
planning tool alone is about 1,500 tokens of instructions and definition, sent with a
one-question task as with a twenty-step one. The cost benchmark
([evaluation.md](../evaluation.md#cost-benchmark)) measured what `ReAct` sends on short and
long tasks; a short task must keep sending exactly that.

## Decision

1. **Reuse, wrap thinly.** Each feature is the framework's own middleware, used as it is; the
   harness adds only the trigger and the glue that keeps it governed, journaled and resumable
   (`trellis.harness.automatic`). A helper's call is a child run of the same agent — the
   machinery of `agent.as_tool()` — not Deep Agents' in-graph sub-agent.
2. **On from signals.** A feature is offered when the run shows it needs it: a long task (the
   model's third call of an invocation, or a request that asks for steps) on a window with room
   for it (32,000 tokens or more). Until then the model is sent neither its tools nor its
   instructions — the request is unchanged, so a prompt cache holds and a short task costs what
   it did (a golden snapshot of the requests proves it, `tests/integration/test_react_golden.py`).
3. **Off with `without=`, by one name each** (`planning`, `files`, `subagent_task`, ...): not
   offered, and a call of it answers that it is not available. No new setting: the thresholds
   are constants, tuned on the benchmark.
4. **A user's own middleware wins.** Each automatic piece is named as the framework's class, so
   the same class given to `ReAct(middleware=)` replaces it (always on, as the team set it).
5. **The journal holds what a feature keeps.** Files live in a plain state channel the run's
   checkpoint keeps whole (Deep Agents' `DeltaChannel` needs every earlier checkpoint, and the
   run keeps only the latest), so they survive a pause, a crash and another worker.

## Consequences

* A long task on a small window (an 8k local model) is not offered these tools: their cost
  would crowd out the task.
* Each feature is measured on the benchmark as one variant on and one off before its default
  changes; a feature that measures worse stays off and is documented as such.
* A team that wants a feature always on adds the framework's middleware itself, as before.
