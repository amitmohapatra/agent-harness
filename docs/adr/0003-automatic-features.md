# 0003. ReAct turns native features on from signals, never from new settings

**Status:** Accepted (2026-10-21)

## Context

LangChain, LangGraph and Deep Agents ship the pieces a long or large agent run needs —
planning (`TodoListMiddleware`), scratch files (Deep Agents' `FilesystemMiddleware`), tool selection, fallbacks, retries, call limits, prompt caching —
each off unless a team adds it. A team that adds them all pays for them on every call: the
planning tool alone is about 1,500 tokens of instructions and definition, sent with a
one-question task as with a twenty-step one. The cost benchmark
([evaluation.md](../evaluation.md#cost-benchmark)) measured what `ReAct` sends on short and
long tasks; a short task must keep sending exactly that.

## Decision

1. **Reuse, wrap thinly.** Each feature is the framework's own middleware, used as it is; the
   harness adds only the trigger and the glue that keeps it governed, journaled and resumable
   (`trellis.harness.automatic`). Delegation stays the team's explicit code
   (`agent.as_tool()`): no automatic helper.
2. **On from signals.** A feature is offered when the run shows it needs it: a long task (the
   model's third call of an invocation, or a request that asks for steps) on a known window with
   room for it (32,000 tokens or more); an unknown window: off. Until then the model is sent neither its tools nor its
   instructions — the request is unchanged, so a prompt cache holds and a short task costs what
   it did (a golden snapshot of the requests proves it, `tests/integration/test_react_golden.py`).
3. **Off with `without=`, by one name each** (`planning`, `files`): not
   offered, and a call of it answers that it is not available. No new setting: the thresholds
   are constants, tuned on the benchmark.
4. **A user's own middleware wins.** Each automatic piece is named as the framework's class, so
   the same class given to `ReAct(middleware=)` replaces it (always on, as the team set it).
5. **The journal holds what a feature keeps.** Files live in a plain state channel the run's
   checkpoint keeps whole (Deep Agents' `DeltaChannel` needs every earlier checkpoint, and the
   run keeps only the latest), so they survive a pause, a crash and another worker.

6. **One owner per concern, no fallback chains.** When the owner is not there, the feature is
   off (or the user's explicit setting applies), said once in the log:
   * tool narrowing — the memory service's tool hints and `tool_search`; without memory, every
     tool is offered (the toolbox's check warns from 20 tools). `LLMToolSelectorMiddleware` is
     the team's own opt-in, not a fallback;
   * a model's context window — `context_window=`, else the chat model's public LangChain
     profile; unknown, the window-scaled features are off;
   * reasoning effort — a fixed `ReAct(effort=)` passed through to the gateway, no per-step
     policy;
   * dollars — the gateway's per-call cost (Bifrost's `usage.cost`); the price table is the
     benchmark's own (`benchmark/cost/prices.py`);
   * model retries, fallbacks and prompt caching — the gateway (Bifrost's provider retries, the
     virtual key's fallbacks, `prompt_cache.auto_inject`); the client makes one attempt;
   * structured output — LangChain's `response_format`;
   * a run's cost cap — the harness's journal counter, only when `budget=` is given; a tenant's
     or a team's budget is Bifrost's;
   * in-run history — the journal (Deep Agents' offload); long-term facts — memory's records.
7. **Deleted after the architecture review (2026-10-21):** the vendor-native client, the
   per-step effort policy, the in-graph prompt cache, fallbacks from code, the structured-output
   repair, the `task` helper, compaction saved to memory, `compact_conversation`, rewind and
   fork, and the default budget. Each had an owner elsewhere or no evidenced need; reopen one
   only with a measurement that shows a non-inferior gain.

## Consequences

* A long task on a small window (an 8k local model) — or one whose window nobody gave — is not
  offered these tools: their cost would crowd out the task.
* Nothing here is on by default on main until the benchmark shows it non-inferior on a model
  with a window of 32k or more (none of it has been measured yet).
* Each feature is measured on the benchmark as one variant on and one off before its default
  changes; a feature that measures worse stays off and is documented as such.
* A team that wants a feature always on adds the framework's middleware itself, as before.
