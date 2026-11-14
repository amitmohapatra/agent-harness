# Architecture and code-quality review: Trellis stack, this round

Read-only review. No product code changed, no commits, no model calls, no test suite run.

## Scope, refs and method

| Short | Repo and ref | Head |
|---|---|---|
| **M** | agent-harness `origin/main` | 9e60203 |
| **F** | agent-harness `origin/claude/cool-maxwell-w5avo8-final` | d2bf754 |
| **B** | agent-harness `…-react-background` (on top of F) | 983923b |
| **C** | agent-harness `…-chat-order` (on top of F) | 1968640 |
| **RM / RF / RC** | agent-runs `origin/main` / `…-final` / `…-chat-order` | 9ecc7f8 / 422ef68 / 087259c |

Harness paths are relative to `src/trellis/harness/`, agent-runs paths to `src/agent_runs/`. A citation reads `file:line @ref`.

What I did: read the new modules in full (claude_live.py, helpers.py, holding.py, background.py, async_tasks.py, background_tasks.py, line.py, outbound.py), the full diffs of pipeline.py, middleware.py, journal.py, tools/bridge.py, tools/base.py, runs.py, runtime.py, conversation.py, agent.py, harness.py, adapters/claude.py (lines 60–1360), adapters/openai_agents.py, and both agent-runs diffs. Tools ran on `git archive` exports under `/var/tmp/cons/review/arch-scratch/` (raw outputs are there: `ruff-*.txt`, `vulture-*.txt`, `dup-*.txt`, `jscpd-*/`).

Limits you should know:
- Every finding is from reading code; **nothing was executed**. Where a finding predicts runtime behaviour I give a confidence.
- adapters/acp.py, tools/convert/openai_local.py and tools/convert/openai_agents.py were skimmed, not read line by line.
- The repo's `/review` skill was loaded as CLAUDE.md requires; its fix-first, specialist-dispatch and QA steps do not apply to a read-only, three-ref review and were not run.

---

## 1. How a turn, a child run and a background task flow today

**A turn.** `Agent.run/stream` (or a surface) records the run `RUNNING` in the run store (`Agent._opened`, agent.py:1448 @C), then calls `pipeline.attempt` (pipeline.py:168 @F). `attempt` takes a *hold* on the run (`Holds.holding`, holding.py:108 @F: a heartbeat every 20 s naming this process), then `_turned` takes the conversation's turn: at F an in-process lock (`conversation.Turns`); at C first the store's *line* (`line.Line.turn`, line.py:132 @C), then the same lock. `_attempt` builds the `Runtime`, restores per-conversation state kept in this process (`Agent.sessions`, `Agent.turned`, `Agent.results`) and, for LangGraph graphs, from the file store (`.carried/…`, pipeline.py:544–648 @F), then the adapter runs the library. Tool calls reach `tools/bridge.py` (decide, journal, record). `_concluded` (pipeline.py:814 @F) writes the ending.

**A child run** exists in three forms:
1. *A sub-agent's run* (`subagents.child_run`): a nested `pipeline.attempt(parent=…)` inside the parent's tool call. It ends itself.
2. *A helper* (`bridges/helpers.Helpers`): a run record the adapter opens when the library spawns something (Deep Agents `task`, OpenAI agent-as-tool, Claude `Agent`). It is bookkeeping only: started, held, ended with its call, recorded to memory. No pipeline attempt.
3. *A detached helper* (`Helpers.detached`, helpers.py:261 @F): handed to a *keeper* that outlives the attempt; its id goes into `Journal.handed_on`.

**A background task** has three keepers, one per library:
- **Claude** (`claude_live.Background`, F): the conversation's CLI process stays alive (`LiveSession`). A backgrounded subagent's helper is detached and from then on driven as a **full pipeline run** of the parent agent (`Background._run` → `pipeline.attempt`, claude_live.py:302–327 @F); the adapter switches to `_attended` when it sees the run id in `LiveSessions.attended` (adapters/claude.py:462 @F). CLI requests are relayed to whichever attempt is under way.
- **Deep Agents remote** (`bridges/async_tasks.AsyncTasks`, B): the helper is detached and held; a 10 s loop calls Deep Agents' own `check_async_task` and reads the run store (async_tasks.py:173 @B).
- **Harness-native** (`background.Tasks`, B): not a helper. A run started with `Agent._start(parent=…)` on its own thread, attempts driven in a detached asyncio task (`Tasks._work` → `pipeline.attempt`, background.py:329 @B), with a 10 s run-store poll (`_watch`, :362).

Outcomes return to the chat by three routes: Claude's CLI injects its own turn; B's `background.delivered` → `notify` starts a *notification run* in the chat thread (background.py:502–618 @B); or the next turn pulls (`BackgroundTasks.abefore_agent`). A paused background child's question is asked through the conversation's next turn (`LiveSession.asks`, `background.ask`).

**Liveness and endings.** agent-runs ends a held run whose hold lapsed (`end_abandoned`, store/runs.py:908 @RF). Children are ended with their parent by client-side cascades in the harness (two of them, see ARCH-3). At C a refused heartbeat calls `Harness._lost` (harness.py:472 @C), which stops a run cancelled elsewhere.

**Per-conversation state held in process memory**, each with its own bound and fallback: `Agent.sessions`, `Agent.turned`, `Agent.results` (texts, turns, graph state), `ClaudeAdapter.live`, `.costs`, `.seen`, `Turns` (five dicts and two caches at C), `LangGraphAdapter.tasks` (B), `Harness.background` (B), `Holds`.

---

## 2. Mechanism count: verdict per pair

Mechanisms: **A** Claude `Background`; **B′** Deep Agents `AsyncTasks`; **C′** harness `Tasks`; **D** `Helpers`; **E** `Holds`; **F′** `Line`; **G** `Turns`.

| Pair | Verdict | Overlap | Minimal shared abstraction | Worth unifying? |
|---|---|---|---|---|
| A ↔ C′ | **Parallel drivers of the same thing** (a child run whose attempts run in a detached task) | Attempt loop through `pipeline.attempt`; ask-through-the-next-turn (near-identical code, DUP-1); stop-the-attempt snippet (DUP-2); store poll for "cancelled elsewhere" | `Tasks.run(agent, record, then=, journal=, resolution=)` already is the driver. `Background._run` should use it; keep only the CLI relay Claude-specific | **Yes** for ask, stop and the registry (small). The relay stays |
| A ↔ B′ | **Parallel keepers of detached helpers** | Registry by run id (`LiveSessions.attended` / `AsyncTasks._kept`); re-hold after `detached`; "cancel with the parent" predicate written twice (claude_live.py:1018–1022 @F, async_tasks.py:256–263 @B) | One keeper registry keyed by run id with `stop(reason)`; simplest home is the hold itself (`Holds.hold(run_id, tenant, on_lost=…)`) | **Yes**: ARCH-3 and ARCH-4 follow from it |
| B′ ↔ C′ | **One interface, two transports — legitimately two implementations**, but the shared vocabulary is copied | Five tool names in four tables with divergent values (ARCH-14); `"async_subagent"` twice; both poll every 10 s; both end in `background.delivered` (good reuse) | One table of the five tools (names, side effects, "result is a record"); `delivered` already shared | Table: **yes**. Transports: no |
| A ↔ D | **One mechanism plus a hand-off** | `Background._finished` re-emits `helper("finished")` (claude_live.py:329–344 @F vs helpers.py:372–379 @F); `detached` lets go of the hold and the keeper takes it again one line later | `detached` returns the helper *with its hold* | Small tidy-up only |
| B′ ↔ D | **Same mechanism; B′ has a second, weaker end path** | `_ended` goes through `Helpers.finished` (events, memory record, outcome); `_closed` writes the store directly with none of those (async_tasks.py:221–235 @B) | One "end a helper's run" function usable without a `Runtime` | **Yes** (ARCH-12) |
| C′ ↔ D | **Different mechanisms, correctly** | Only metadata conventions (`BACKGROUND` vs `HELPER`), bridged by `origin_of` (background.py:529 @B) | none needed | No |
| E ↔ A/B′/C′ | **E is the shared base; the watchers re-implement what E already learns** | A refused heartbeat already says "this run no longer runs here". Three loops poll `runs.get` every 10 s to learn the same (claude_live.py:353–360 @F, background.py:362–370 @B, async_tasks.py:173–190 @B) | `Holds(lost=…)` — exists at C (holding.py:76–88 @C) | **Yes, at the fold** (ARCH-4) |
| E ↔ agent-runs lease | **Parallel mechanism, converging feature by feature** | `_hold` vs `_lease`; `end_abandoned` vs `requeue_lapsed`; at RC cancel-request, non-extension and release-ends-cancelled are re-added for holds | One liveness lease whose lapse policy (requeue or end) derives from `queued_at` | Should (ARCH-10) |
| F′ ↔ G | **Layered; G's lock is now almost subsumed** | With a line, the run ahead in this process is awaited by `_leaving` events, which is what the lock did. The lock still serves runs with no `concurrency_key` | Keep `Turns.taken(line=…)` as the single entry (it is) | No removal now; tidy `Turns` (ARCH-19) |
| F′ ↔ E | **Complementary** | The line relies on holds to remove a dead process's run. One gap: a run recorded but never held (ARCH-5) | — | Fix the gap |
| G ↔ A | **Not duplicates** | `LiveSession.owed/settled` serialises CLI prompts because a CLI turn can outlive the harness turn | — | No |
| G ↔ C′ | **Reuse, good** | Follow-ups use `conversation.Mode` (background.py:433 @B) | — | No |
| D ↔ E, D ↔ F′, D ↔ G, B′ ↔ F′/G, A ↔ F′, C′ ↔ F′ | No overlap beyond "is held" | — | — | No |

Two more pairs that are not on your list but are the same kind of problem:
- **Scheduled runs ↔ notification runs** ("a turn the chat started itself"): parallel (ARCH-11).
- **`subagents.cancel_children` ↔ `pipeline._children_ended`**: two cascades for one job (ARCH-3).

---

## 3. Findings, ranked

### Must fix

**ARCH-1 — Children are left `RUNNING` and un-held when a run ends `ERROR` after a stall.** Confidence 8/10 (read, not run).
- Evidence: `Helpers.closed` treats every non-`Exception` error as "paused or crashed" and *releases* each open helper's hold in the store (helpers.py:396–402 @F). A stalled model call cancels the attempt task with `STALLED` (runtime.py:577 @F), so the adapter's `finally` calls `closed(CancelledError)`. The pipeline then turns the stall into the run's error (pipeline.py:421, 710, 768–778 @F) and takes the `elif error is not None` branch (pipeline.py:867 @F), which does **not** call `_children_ended`; only the `TIMEOUT` branch (:855–866) and `_settle_cancelled` (:1443) do. agent-runs has no server-side cascade (`cancel`, store/runs.py:529–562 @RF) and never ends a run with no `held_until` (`end_abandoned` docstring, :908 @RF). A helper's record has no timeout, so only the parent's `deadline`, if any, ends it.
- Cost: run records `RUNNING` for ever — the exact thing `holding.py` was added to prevent — and at C/B anything that lists a thread's unfinished runs sees them.
- Smallest change: call `_children_ended` once, for every final status, at the end of `_concluded` instead of in two branches; or stop inferring "paused" from the exception type and pass it to `Helpers.closed` explicitly.
- Files: pipeline.py, bridges/helpers.py.

**ARCH-2 — A cancel from another process is not honoured by the holder, and the in-process stores do not model it.** Confidence 7/10.
- Evidence: at RC a cancel on a held in-process run no longer ends it; it sets `cancel_requested_at` and the hold "answers `cancel_requested: true`" (store/runs.py:609, 860–880 @RC). The harness ignores the heartbeat's return value and reacts only to `LeaseLostError` (holding.py:143–158 @C); `cancel_requested` is read only for worker jobs (pipeline.py:767, 843 @C). `Harness._lost` acts only when the record is already `CANCELLED` (harness.py:472–490 @C). So `mode="interrupt"` from another replica (`_elsewhere`, pipeline.py:335 @C) keeps the interrupted run working — model calls and tool side effects included — until the hold lapses and the ticker cancels it (up to about 60 s + a tick + 20 s).
- `LocalRuns.cancel` still ends such a run at once (runs.py:450–463 @C; only `_leases` defers) and `LocalRuns._hold` never reports `cancel_requested`. `FileRuns` (tests/support/shared_runs.py:25 @C) inherits that. Tests built on these two stores therefore pass against a stand-in whose cancel semantics the real store no longer has (I did not check which tests, if any, run the line against a real agent-runs); the module docstring's "Both behave the same way" (runs.py:4 @C) is false at C.
- Cost: the user-visible promise of interrupt is broken across replicas, and tests cannot see it.
- Smallest change: in `Holds._beating`, treat a lease with `cancel_requested` like a refusal (call `lost`), and let `_lost` stop the attempt on "cancel requested" as well as `CANCELLED`; mirror RC's deferred cancel in `LocalRuns.cancel/_hold`; add one store-contract test parametrised over `LocalRuns`, `FileRuns` and the agent-runs client.
- Files: holding.py, harness.py, runs.py, tests/support/shared_runs.py.

**ARCH-3 — Two cascades end a run's children, with different rules, and every keeper is wired into them separately.**
- Evidence: `subagents.cancel_children` (subagents.py:303–311 @F) cancels every non-final child; `pipeline._children_ended` (pipeline.py:740–765 @F) ends only `RUNNING` children and spares `Journal.handed_on`. `Agent.cancel` on a running turn triggers **both** (agent.py:606–618 @B: the attempt's `_settle_cancelled` spares handed-on children, then `cancel_children` cancels them anyway). At B each cascade gains `harness.background.stop` (pipeline.py:776, subagents.py:317 @B), only `cancel_children` calls the adapter's `child_cancelled` (subagents.py:309 @B), only the LangGraph adapter implements it (adapters/langgraph.py:186 @B), and Claude relies on a third path (`LiveSession._detach`, claude_live.py:1018–1022 @F). The timeout branch adds a second "spare these" list (`background.of_turn`, pipeline.py:877 @B).
- Cost: "children end with their parent" is enforced in five places with three exemption lists. B had to edit both cascades; the next background kind will too, and missing one is silent. The brief calls two mechanisms for one job a rejection.
- Smallest change: one `end_children(agent, run_id, tenant, status, *, kept, reason, error)` used by `Agent.cancel`, `Agent._continue`, `_settle_cancelled` and `_concluded`, with one keeper-stop call inside (see ARCH-4). The clean end state is a cascade inside agent-runs' `_ended`, but that is a contract change.
- Files: subagents.py, pipeline.py, agent.py.

**ARCH-4 — Four detectors of "a run I keep was ended elsewhere" and four copies of "stop the attempt"; B and C will collide when folded.**
- Evidence, detectors: `Holds(lost=…)` + `RunLog(ended=…)` (holding.py:76–88, runlog.py:114–119 @C); `Tasks._watch` (background.py:362–370 @B); `AsyncTasks._watch` (async_tasks.py:173–190 @B); `Background._answered` (claude_live.py:353–360 @F). Three are fixed 10 s polls with no backoff or jitter, which the brief forbids on hot paths.
- Evidence, stop: `Agent.cancel` (agent.py:606–609 @B), `Background.cancel` (claude_live.py:392–395 @F), `Tasks.stop` (background.py:376–381 @B), `Harness._lost` (harness.py:487–490 @C). Each sets `runtime.cancelled` and cancels `running_in`, with a different guard (overwrite / keep first / only if unset).
- Cost: after the fold a harness background task is watched twice, and a reason overwritten or kept depending on which path fired first.
- Smallest change: keep C's `Holds(lost=…)` as the only detector; delete `Tasks._watch`; reduce `AsyncTasks._watch` to the remote read; add one `Agent.stop_attempt(run_id, reason) -> bool` and call it from the four sites.
- Files: holding.py, harness.py, background.py, bridges/async_tasks.py, bridges/claude_live.py, agent.py.

**ARCH-5 — A turn recorded `RUNNING` but never held blocks its conversation's line.** Confidence 6/10.
- Evidence: at C a conversation's turn is recorded with the conversation's `concurrency_key` in `_opened` (agent.py:1474–1505 @C) and held only when `pipeline.attempt` starts (callers: agent.py:479–492, 518–535; agui/__init__.py:241; a2a/executor.py:107 @C). A process killed between the two leaves a `RUNNING` run with no `held_until`; `end_abandoned` never ends it ("A run never held is never ended here", store/runs.py:1037 @RC), and `Line.ahead` counts every `RUNNING` run of the key (line.py:103–130 @C). `resume` already closes the same gap for answered runs with `RESUME_GRACE` (store/runs.py:513 @RF); `start` does not.
- Cost: low probability, but in a multi-replica deployment with restarts it will happen, and then every later message of that conversation waits behind a dead run until its timeout — for ever when none is set (`run_timeout` defaults to `None`, agent.py:316 @C).
- Smallest change: in agent-runs `start`, give a non-queued run with a `concurrency_key` the same grace `resume` gives (`held_until = now + RESUME_GRACE`), so it is born "to be held".
- Files: agent-runs store/runs.py; harness runs.py (`LocalRuns` twin).

**ARCH-6 — New process-wide monkeypatch and private LangGraph imports, to retry the graph's own tools from the harness.**
- Evidence: `ToolNode._arun_one = called` (middleware.py:1140 @F) is new this round (`_execute_tool_async` at :1139 was already patched on M). `_node_retries` reads LangGraph's private task object through `CONFIG_KEY_CALL` and a weak reference (middleware.py:1180–1189 @F), importing `langgraph._internal._constants` and `langgraph.pregel._retry._should_retry_on` (:103, :118). All of it exists so the harness can retry a framework-owned read "around the graph's own handling", which needs the eight-field `_Calling` state (:592–632), the module-global `_BESIDE` (:638), the one-implementation `Deferring` protocol (tools/base.py:262–276 @F) and an `again` flag threaded through `bridge._called` (tools/bridge.py:367–390 @F).
- Cost: breaks brief §1 ("Never change, patch or monkeypatch a library's behaviour for the library's own tools") and "prefer the library's implementation": LangGraph has `RetryPolicy` and `ToolRetryMiddleware`. The `getattr(..., None)` fallbacks in `_node_retries` fail *silently* on a LangGraph update: the harness would then retry on top of the node's own policy.
- Smallest change: do not retry framework-owned tools in the harness (the library's retry is the retry). That deletes `Deferring`, `_Calling.owed/entries/tried`, `_again`, `_node_retries`, both private imports and the `_arun_one` patch's retry loop. If harness retries there are a product decision, record the exception in DECISIONS.md and make `_node_retries` fail loudly, as `handed_to` does (adapters/openai_agents.py:671–684 @F).
- Files: middleware.py, tools/base.py, tools/bridge.py.

### Should fix

**ARCH-7 — `Tasks._done` forgets the wrong attempt; its `task` argument is unused.** Confidence 6/10 that it bites.
- Evidence: `task.add_done_callback(lambda _: self._done(record.run_id, task))` and `_done` pops by run id without comparing (background.py:322–327 @B; ruff ARG002 at :324). The next attempt of the same run is registered under the same key by `answered` (:765); if it registers before the earlier attempt's callback fires (the earlier `_work` is still inside `then`, :351–360), the callback removes the new entry. Then `at_work` is false, `stop` cannot find it and `aclose` does not cut it off.
- Smallest change: `if self._at_work.get(run_id) is task:` before popping.
- Files: background.py.

**ARCH-8 — Persistence and a hand-rolled merge protocol live in `pipeline.py`, for one LangGraph channel.**
- Evidence: `.carried` folder layout, file naming, read-merge-write-then-delete and its concurrency argument are in pipeline.py:544–648 @F. The channel list has one entry (`CARRIED = ("async_tasks",)`, adapters/langgraph.py:138 @F). The pipeline asks the LangGraph adapter by name whether to read (`carries(agent.target)`, pipeline.py:520 @F, import at :65) but writes whenever `runtime.carried` is set (:532–534). The "generic" merge knows Deep Agents' task record (`_later`, `_FURTHER`, tools/results.py:402–409 @B), and the state is cached in a class named for tool results (`Conversations.state/left`, tools/results.py @F).
- Cost: the pipeline has a second reason to change (storage layout), shared code imports an adapter, and a list plus N reads of the file store happen at each turn's start and end for such graphs. Three modules must agree on one record shape.
- Smallest change: move `_carried_at/_kept_name/_read_carried/_keep_carried` into one store class (next to `files`), have the pipeline call `store.load(conversation)` / `store.keep(conversation, state)`, and replace `carries(target)` with a capability on the `Adapter` protocol.
- Files: pipeline.py, tools/results.py, adapters/base.py, adapters/langgraph.py.

**ARCH-9 — The stated layering does not describe the tree.**
- `bridges/` holds four single-library modules: claude.py, claude_live.py (Claude only; imported only by adapters/claude.py:110 @F), async_tasks.py (Deep Agents only), background_tasks.py (a LangChain middleware). Only helpers.py and store.py are shared.
- Shared code imports an adapter: pipeline.py:65 @F (`FOREIGN, HITL, carries, holds, is_hitl`; four of the five predate this round), background.py:65 @B (`ASKED`) plus `langchain_core` inside `notification` (:262–274), agent.py:68 @B, bridges/async_tasks.py:358–359 @B (adapter and middleware).
- Cross-module private access: `agent._continue` (claude_live.py:319 @F), `agent._resumed` and `agent._start` (background.py:466, 599, 758 @B), `agent.harness._lost` (pipeline.py:998 @C), `found._settled.set()` from another class (claude_live.py:1087 @F).
- Cost: reviewers cannot use the layering rule as a check, because it is already false in 6+ places.
- Smallest change: either rename the rule ("`bridges/` = non-adapter code for one or more libraries") or move the four modules beside their adapter; make the three `Agent` methods public (`continue_run`, `resumed`, `start_record`) since three modules depend on them; move `ASKED` and the notification message builder into the LangGraph side and have `background.py` return plain data.

**ARCH-10 — agent-runs keeps two liveness mechanisms that are converging.**
- Evidence: RF adds `holder/held_until`, `_hold`, `end_abandoned` and an index beside `lease_owner/lease_expires_at`, `_lease`, `requeue_lapsed` (store/runs.py:739, 790, 908; tables.py:102–103, 141–143 @RF). RC then re-adds lease features to holds one by one: deferred cancel (:609), `cancel_requested` and non-extension (:860–880, compare `_lease` :919–932), release-ends-cancelled (:883), cancelled-instead in the sweep (:1037 @RC). The "kept in its caller's process" test is written twice (:726, :772 @RF) and twice more in `LocalRuns` (runs.py:407, 425 @F).
- Cost: every lease feature is implemented four times (lease and hold, agent-runs and `LocalRuns`); ARCH-2 is the first drift.
- Smallest change now: one helper for the predicate, and one function computing expiry under a cancel request used by both `_hold` and `_lease`. Decide separately whether a hold is a lease with "end on lapse, do not fence" (open question 2).
- Files: agent-runs store/runs.py; harness runs.py.

**ARCH-11 — "A turn the chat started itself" is implemented twice.**
- Evidence: schedules have `Fired`, `Runtime.scheduled`, `schedules.stale/quiet`; B adds `NOTIFIED`, `Runtime.notification`, `background.stale/quiet` beside each use: pipeline.py:382, 691, 1255 @B, agui/threads.py:144 @B (`Fired.of(...) is not None or NOTIFIED in record.metadata`), conversation.py:235 @B, and `background.quiet` reuses `schedules.NO_REPORT` (background.py:640–646 @B).
- Cost: a third kind (the brief names "a new background kind") needs edits in five places.
- Smallest change: one `Runtime.own_turn` object with `stale()` and `quiet(answer)`, built from the record's metadata by a small registry of two; pipeline and agui branch once.
- Files: pipeline.py, runtime.py, schedules.py, background.py, agui/threads.py.

**ARCH-12 — A detached helper's run ends by four paths that record different things.**
- Evidence: `Helpers.finished` writes the store, emits `helper_finished`, records turns and outcome to memory (helpers.py:388–430 @B). `AsyncTasks._closed` writes the store only (async_tasks.py:221–235 @B). `Helpers._left_open` writes the store and memory but emits no event (helpers.py:412–441 @F). Claude's `Background` ends through the pipeline and emits `helper("finished")` itself (claude_live.py:329–344 @F).
- Cost: whether a Deep Agents task's end reaches memory and the task tree depends on who noticed it (a turn's check, or the watcher).
- Smallest change: let `Helpers.finished` take the pieces it needs (runs, holds, writes, optional events) instead of `self.runtime`, and call it from `_closed` and `_left_open`.
- Files: bridges/helpers.py, bridges/async_tasks.py.

**ARCH-13 — Claude session cost is tracked in four places kept equal by assignment.**
- Evidence: `Spending.billed` (adapters/claude.py:986 @F), `LiveSession.billed` (claude_live.py:635 @F), `Journal.session_cost` (journal.py:302 @F), `ClaudeAdapter.costs[session]` (adapters/claude.py:332 @F); synchronised by hand at adapters/claude.py:727, 1003–1009 and adjusted inside `charged` (claude_live.py:1099–1105). Cost shares are computed in three functions (`Spending.ended`, `LiveSession.charged`, `Background.spent`).
- Cost: a missed assignment double-charges or zero-charges a turn; the invariant "one running total per session" has no owner.
- Smallest change: one small `SessionCost` object (total last charged, `grew_by(total) -> float`) that the journal persists and the live session and `Spending` share by reference.
- Files: adapters/claude.py, bridges/claude_live.py.

**ARCH-14 — The five async-task tools are described in four tables, and two disagree.**
- Evidence: `middleware.ASYNC_TASKS` says start, update and cancel are **writes** (middleware.py:261–267 @F); `background.SIDE_EFFECTS` says start and update are **reads** and only cancel a write (background.py:833–843 @B). Names also in `background.TOOLS` (:773), `async_tasks.START/UPDATE/CHECK/CANCEL` (async_tasks.py:58–61 @B), `background_tasks.SCHEMAS` (background_tasks.py:52–58 @B); the notification source is defined twice (`background.SOURCE` :86, `middleware.ASYNC_SUBAGENT` :2119 @B).
- Cost: the model is promised "one interface whether a task runs on a server or here" (background.py:5–6 @B) but governance treats `start_async_task` differently in the two cases, with nothing saying it is intended.
- Smallest change: one table in one module (name → side effect per transport, "result is a record"), imported by the other three; say in its comment why remote start is a write and local start a hand-over, or make them equal.
- Files: background.py, middleware.py, bridges/async_tasks.py, bridges/background_tasks.py.

**ARCH-15 — The line's ordering key is computed on both sides of the API, and `created_at` now doubles as the place.**
- Evidence: server `_place` = `greatest(created_at, max(resolution.recorded_at))` (store/runs.py:307–316 @RC); client `placed()` recomputes it by paging every resolution of every run in the line (line.py:75–86, 184–195 @C). `_placed` rewrites `created_at`, `running_since` and `queued_at` to enforce order (store/runs.py:772–830, :813 @RC). `IN_LINE` is defined in both repos (line.py:54 @C, store/runs.py:115 @RC).
- Cost: two definitions that must stay equal; one `get` plus N resolution pages per unseen run on a path the brief lists as hot; "created" no longer means created.
- Smallest change: return the place on `RunRecord`/`RunSummary` (a `placed_at` field); the harness sorts by it and `line.placed` goes away. This is a contract addition, so it needs approval.
- Files: agent-runs domain/runs.py, store/runs.py; harness line.py.

**ARCH-16 — Claude's harness tool calls are attributed to a background subagent by equal name and arguments.**
- Evidence: `_call` finds the owner with `(n, a) == (name, args)` over `_calling` (claude_live.py:782–783 @F), filled by the `PreToolUse` relay (`_noted`, :734–753). The main agent making the same call with the same arguments at the same time is taken for the subagent's, so it is governed and recorded under the wrong run. The same content-as-identity appears in `Replay.announced/claimed` (journal.py:505–513 @F) and `Steps.started` (adapters/claude.py:1149–1151 @F).
- Cost: rare misattribution of approval, record and budget between two runs.
- Smallest change: key `_calling` by the call id and pass it through if the SDK exposes the tool-use id to an MCP handler (open question 5); otherwise queue per `(name, args)` in arrival order, as `claimed` already does, rather than "first match".
- Files: bridges/claude_live.py, tools/convert/claude.py.

**ARCH-17 — The journal's new facts are parallel arrays, and middleware writes the journal's internals.**
- Evidence: `answered_by[key][i]` shadows `answers[key][i]`, `reads[key][i]` shadows `calls[key][i]`; alignment is maintained by padding in three places (journal.py:427–432, 676–682, 713–717 @F). `middleware._kept_update` replaces `journal.calls[key][-1]` in place (middleware.py:553–569 @F). Two different context variables are both named `"trellis_calling"` (journal.py:151, middleware.py:635 @F).
- Cost: a fourth writer that forgets the padding corrupts replay silently; the journal's format is owned by two modules.
- Smallest change: give `Replay` a method `keep_update(key, outcome, update)` and call it from middleware; rename one context variable. Folding the parallel lists into one record per entry is cleaner but changes a persisted format — ask first.
- Files: journal.py, middleware.py.

**ARCH-18 — New process-global registries.** The brief forbids them.
- `_BESIDE` (middleware.py:638 @F), `_HOOKED` keyed by `id(outcome)` (tools/bridge.py:140, 175–181 @F), and four `WeakKeyDictionary[Runtime, …]` side tables (tools/convert/openai_local.py:48–54 @F).
- Smallest change: `_ran` already returns a tuple — return "hooks changed it" there and thread it to the two callers instead of `_HOOKED`; put the OpenAI per-run state on one object the adapter creates per invocation; `_BESIDE` goes with ARCH-6, or moves onto `Runtime`.
- Files: tools/bridge.py, middleware.py, tools/convert/openai_local.py, adapters/openai_agents.py.

**ARCH-19 — Small correctness-by-construction gaps.**
- `Turns.accepting` is keyed by `f"{tenant}\0{user}\0{thread}"` with the caller's unresolved `tenant` (agent.py:1475 @C), while everything else uses `conversation_key(start.tenant_id, user, thread)` (:1496). Two callers of one conversation, one passing `tenant=None`, take different locks and lose the arrival order the lock exists for. Fix: resolve the tenant first and use `conversation_key`.
- `Turns` now owns turn-taking *and* two read caches (`seen`, `ended`; conversation.py:355–357 @C); the ref-counted lock dict is written twice (`_held`, `_accepting`, :360–375, :378–412). Fix: one small keyed-lock helper; move `ended` to where `earlier` lives.
- `Holds.release` drops every holder's count (holding.py:116–126 @F) while `let_go` decrements; a later `let_go` from the first holder then decrements a hold someone else took. No caller does this today. Fix: document or make `release` respect the count.
- `async_tasks._watch` asserts the run exists inside a detached loop (async_tasks.py:184, 218 @B); a deleted run kills the watcher with an unretrieved exception and leaves the hold beating. Fix: treat `None` as "ended".
- `_served_elsewhere` does one list plus one `get` per listed run at every first attempt of a session-keeping adapter (pipeline.py:467–489 @F). Fix: stop at the first run newer than `mine` (the list is newest first).

**ARCH-20 — Code kept for tests only, or with no reader.** Details in section A.
- `LiveSession.kept`, `unattended()`, `UNATTENDED_KEPT` (claude_live.py:101, 181–186, 620 @F): production appends, only tests read.
- `Holds.holds()` (holding.py:81–83 @F): 11 uses, all in tests. `Holds(name=)`: only tests pass it.
- Smallest change: delete `kept`/`unattended` and assert in tests on what a user sees (events, the run record); delete `holds()` or use it.

### Nice to have

**ARCH-21 — Library internals read by reflection, with silent fallbacks.** `_specs` scans a Deep Agents function's `__closure__` (async_tasks.py:338–352 @B); `named_subagent_graphs` does the same (adapters/langgraph.py:802–824 @B); `HANDOFF_AGENT = "_agent_ref"` and `_get_failure_error_function` use OpenAI Agents SDK privates (adapters/openai_agents.py:529–533, 668 @F). `handed_to` fails loudly when the field is gone; the others degrade quietly. Since `compat.TESTED` pins the tested range, prefer the loud form everywhere.

**ARCH-22 — `Kind` branches in the shared helper.** `if kind is Kind.SUBAGENT` (cursor), `if kind is not Kind.ASYNC_TASK` (no briefing), and the `begun_again` filter (helpers.py:259, 262, 343 @B). Pass two keyword arguments from the adapter (`restarts_after_pause`, `briefed`) or put them on `Kind`.

**ARCH-23 — Names that mislead.** `Holds.aclose()` is synchronous (holding.py:128 @F) beside six awaited `aclose`. `_strategies(subagents)` pops a key out of the caller's spec dicts (react.py:453–467 @B). `Conversations` in tools/results.py holds graph state. `background.UNFINISHED` is a status set while `helpers.UNFINISHED` is an error code (background.py:101, helpers.py:73 @B). `created_at` is a place (ARCH-15).

**ARCH-24 — Testability.** Behaviour is gated on module constants with real timers: `IDLE_SECONDS`, `WORKING_SECONDS`, `SETTLE_SECONDS`, `ANSWER_SECONDS`, `WAITS_SECONDS` (claude_live.py:92–117 @F), `ATTRIBUTED_SECONDS` (helpers.py:83), `HOLD_SECONDS` (holding.py:45), `WATCH_SECONDS` twice (B), `POLL_SECONDS`/`STILL_SECONDS` (line.py:45–48 @C). Tests must patch constants or wait. ARCH-4 removes three of the loops; for `LiveSession` inject one `clock`/`call_later` pair at construction so idle and settle can be tested deterministically.

**ARCH-25 — Module size, where a seam has a real cost.** claude_live.py (1,294 lines) holds four responsibilities: the session and relay, the background run, cost sharing (`charged`, `spent`, `_tokens`), and the transcript mirror. The cost code is the part with a correctness risk (ARCH-13); split that out first. I would not split the relay from the session: they share one state machine (`owed`, `injected`, `due`, `left`) and separating them would spread it.

---

## A. Unused or unnecessary code (tool evidence)

**Commands.** `ruff check --isolated --target-version py313 --select F,ARG,ERA,PIE,SIM,RET,C901,PLR0912,PLR0915` and `vulture src --min-confidence 60` (run under Python 3.13; under 3.11 vulture skips every file with PEP 695 syntax) on M, F, B, C, RM, RF, RC. Findings were compared with main by file and message, so line shifts do not count.

**Totals.**

| Tree | ruff hits | vulture hits |
|---|---|---|
| harness M / F / B / C | 144 / 148 / 154 / 149 | 115 / 113 / 117 / 113 |
| agent-runs RM / RF / RC | 1 / 1 / 1 | 58 / 59 / 59 |

No F, ERA, PIE, SIM, PLR0912 or PLR0915 hit was added: the project already enforces F, SIM and PL. What this round added:

| Hit | Where | Verified by grep | Class |
|---|---|---|---|
| ARG002 `task` | background.py:324 @B | Passed by the lambda at :322, never read | **Real** — and a missing guard (ARCH-7) |
| ARG001 `args` | claude_live.py:1261 @F | `refused(args)` must match the MCP handler shape | Protocol shape, false positive |
| ARG002 `runtime` ×3 | bridges/background_tasks.py:127, 145, 152 @B | LangChain middleware hook signature | Library calls it, false positive |
| ARG002 `runtime` | middleware.py (one new hook, `abefore_model` of the step limit) @B | same | Library calls it |
| vulture `create_file/update_file/delete_file` | tools/convert/openai_local.py:254–260 @F | The SDK's editor protocol | Library calls it |
| vulture `abefore_agent/abefore_model/aafter_model` | bridges/background_tasks.py, middleware.py @B | LangChain hooks | Library calls it |
| vulture `Invoke` | tools/convert/openai_agents.py:63 @F | Used in a `cast("Invoke", …)` string at :124 | False positive |
| vulture `abandoned` | ticker.py:89 @RF | Dataclass field set at :135 | False positive |

A second pass counted references for every function, class, constant and field added in the three diffs (script `unused.py` in the scratch folder): none has zero source references. The tools miss the following, found by hand:

**Never read in production / used only by tests**
- `LiveSession.kept`, `unattended()`, `UNATTENDED_KEPT` — claude_live.py:101, 181–186, 620 @F. Written at :859; read only in tests/integration/test_claude_background_session.py:220–280 @F. The docstring calls it "the one place a later change delivers them from": a hook for later, which the brief rules out.
- `Holds.holds()` — holding.py:81–83 @F: no source caller at F, B or C; 11 test uses.
- `Holds.__init__(name=)` — only tests/unit/test_holding.py passes it.
- `Tasks.stop()`'s `bool` result — background.py:372 @B: all three callers ignore it (pipeline.py:776, subagents.py:317, background.py:741 @B).
- tests/support/invariants.py:64 @C `ENDED` — defined, no use anywhere under tests/.

**Parameters with one caller or one value**
- `Helpers.start(run_id=, about=)` — only async_tasks.py:148 @B.
- `bridge.call(decision=)` — only openai_local.py:327 @F. `bridge.permitted(framework=)` — only acp.py:1607 @F.
- `Replay.calling(again=)` — only `again=True` at tools/bridge.py:269 @F. `Replay.record_call(scope=)` — only tools/bridge.py:335 @F.
- Defaults never relied on: `LiveSession.charged(own=True, taken=True)` (all three calls pass both, claude_live.py:850–858 @F); `_options(kept=False)` (both calls pass it, adapters/claude.py:492, 620 @F).

**Pass-through wrappers**
- `pipeline._carried` returns `(await _read_carried(...))[0]`, one caller (pipeline.py:562–567 @F).
- `conversation.steering_elsewhere` only calls `_said_elsewhere` (conversation.py:440–444 @C); it copies the existing `steering` wrapper.
- `pipeline._turned` exists only so `attempt` can wrap it in `async with held`; eleven keyword arguments are passed straight through, and again to `_attempt` (pipeline.py:208–222 and :250–262 @F; jscpd reports the two signatures, :223–234 and :333–344, as a clone).
- `claude_live.unattended` is a one-line append (see above).
- `Agent.aclose` is `getattr(self.adapter, "aclose", None)` (agent.py:406–411 @F) — a wrapper around a method the protocol does not declare (section B).

**Defensive code for states that cannot occur, or that hides a state that can**
- `_SILENT`, a fake events object with two no-op methods used by one caller (tools/bridge.py:746–752 @F).
- "A release laid out otherwise" fallbacks in `_specs` and `named_subagent_graphs` (ARCH-21): outside the tested range by definition.
- The opposite case: `assert record is not None` in a detached loop for a state that *can* occur (ARCH-19).

**Compatibility shims for code that never shipped:** none found. New `Journal` fields default sensibly for existing checkpoints, which is needed, not a shim.

---

## B. SOLID

### Single responsibility

Functions this round pushed over complexity 10 (ruff C901; none exceeds 50 statements, PLR0915 is clean):

| Function | Where | Complexity | Note |
|---|---|---|---|
| `_governed_server` | adapters/openai_agents.py:520 @F | 14 → **25** | A class body inside a factory function |
| `governed` | tools/convert/openai_agents.py:89 @F | **16** (new) | |
| `govern_tool_nodes` | middleware.py:1062 @F | **14** (new) | Installs two patches and holds the retry loop (ARCH-6) |
| `through` | tools/convert/openai_local.py:95 @F | 12 (new) | |
| `governed_call` | middleware.py:411 @B | 12 (new at B) | Gains the `adapter.tasks` branch |
| `_called` | tools/bridge.py:341 @F | 11 (new) | The `again` flag adds four branches |
| `invoked` | tools/convert/openai_agents.py:134 @F | 11 (new) | |
| `LiveSession._took`, `LiveSession.close` | claude_live.py:817, 1133 @F | 11 each (new file) | |
| `earlier` | conversation.py:195 @C | 11 (new at C) | |

Modules with more than one reason to change:
- **pipeline.py** — run lifecycle, plus file-store persistence (ARCH-8), plus session staleness across workers (`_served_elsewhere`), plus two kinds of self-started turns (ARCH-11).
- **claude_live.py** — session and relay, background run, cost sharing, transcript mirror (ARCH-25).
- **background.py** (1,035 lines @B) — task records and wording, the in-process driver (`Tasks`), delivery and idleness, question asking, and the five model-facing tools with their text.
- **middleware.py** (3,447 lines @B) — every LangChain middleware plus the ToolNode patches plus the checkpointer.
- **conversation.Turns** — turn-taking plus two read caches (ARCH-19).
- **tools/results.Conversations** — large results, transcripts, and graph state.

### Open/closed: branches on a library, adapter or tool name in shared code

| Where | Branch | New this round? |
|---|---|---|
| pipeline.py:520 @F | `if carries(agent.target)` (LangGraph function) | **Yes** |
| pipeline.py:65 @F | imports `FOREIGN, HITL, holds, is_hitl` from adapters.langgraph; used at :1015, 1144, 1154, 1181 @B | No (M) |
| pipeline.py:907 @B | `agent.adapter.name in NO_MODEL` | No |
| pipeline.py:382, 691, 1255 @B | `background.NOTIFIED / stale / quiet` beside the schedule ones | **Yes** (B) |
| bridges/helpers.py:259, 262, 343 @B | `Kind.SUBAGENT`, `Kind.ASYNC_TASK` | **Yes** (F and B) |
| bridges/async_tasks.py:123, 125 @B | `tool.name == START / UPDATE` | **Yes** (B) — acceptable: the module *is* the Deep Agents translator, but it sits in `bridges/` |
| background.py:840, 898–909 @B | `if name == "start_async_task"` … ladder in `Offered.call` | **Yes** (B) — a dict of five handlers removes it |
| subagents.py:309 @B | `getattr(agent.adapter, "child_cancelled", None)` | **Yes** (B) |
| agent.py:409 @F | `getattr(self.adapter, "aclose", None)` | **Yes** (F) |
| middleware.py:473 @B | `getattr(runtime.agent.adapter, "tasks", None)` | **Yes** (B) |
| agent.py:350, 354 @B | `self.adapter.name == "langgraph" / "openai_agents"` | No |
| agui/threads.py:144 @B | `Fired.of(...) or NOTIFIED in record.metadata` | **Yes** (B) |

tools/bridge.py, journal.py and line.py contain no name branch.

### Liskov

- **Stores.** `LocalRuns` and `FileRuns` do not honour agent-runs' contract at C: cancel of a held run, `cancel_requested` on a hold, and placing (ARCH-2). `FileRuns` is test support (tests/support/shared_runs.py:25 @C), so the multi-process tests exercise the divergent behaviour. tests/contract/test_runs_wire.py checks message *shapes*, with `LocalRuns` standing in for the service (:103–119 @C); I found no test that asserts the same *behaviour* (cancel of a held run, placing) against agent-runs and the in-process stores.
- **Adapters.** The `Adapter` protocol (adapters/base.py:134–176 @B) does not declare `aclose`, `child_cancelled` or `tasks`; three callers probe with `getattr`. An adapter that keeps something alive and forgets one of these names compiles, type-checks and silently leaks. `OpenAIAgentsAdapter.__init__` now takes the target (adapters/__init__.py:34 @F) while the others take nothing.
- **Test stand-ins.** tests/unit/test_claude_live.py:158 and tests/unit/test_holding.py:19 @C define their own `Store` classes; tests/unit/test_bridge_helpers.py:33 builds a harness from `SimpleNamespace`. Several integration tests reach into `store._leases` to lapse a lease (section D). Each is a stand-in that can drift from `RunStore` the same way `LocalRuns` did.

### Interface segregation and dependency inversion

- `cancel_children` took a `RunStore` at F and takes the whole `Agent` at B, to reach `agent.adapter.child_cancelled` and `agent.harness.background` (subagents.py:302–318 @B). ARCH-3/4 give it back a narrow argument.
- `Background.__init__` takes the session and digs `turn.runtime.agent`, `.tenant`, `.framework_options`, `agent.harness.holds`, `agent.harness.runs`, `agent.running`, `agent._continue`, `agent.adapter.name` out of it (claude_live.py:255–299, 319, 333–341 @F).
- `background.idle/notify/delivered/answered` take `Agent` for `agent.turns.live`, `agent.harness.runs`, `agent.id`, `agent._start` (background.py:502–618 @B).
- `claude_live.gone(runtime, …)` needs the run store, a tenant and an events sink (claude_live.py:152–178 @F).
- Good counter-examples to copy: `Holds(runs: Callable[[], Holding])` with a two-method protocol (holding.py:54–78 @F) and `Line(runs, record, seen)` (line.py:89–97 @C).
- Module-level registries added: ARCH-18. Shared code importing an adapter: ARCH-9.

---

## C. Design patterns

**An existing pattern should have been used**

| Pattern | Where a ladder or copy was written instead |
|---|---|
| **State** | No run-side transition table in the harness for helpers, background tasks or live sessions. `Background` is seven independent fields (`over`, `ended`, `lost`, `stopped`, `question`, `turn`, `_answer`; claude_live.py:272–294 @F) with rules spread over `served`, `abandoned`, `finished`, `cancel`, `_over`. `LiveSession` is `owed/injected/due/left/closed` (:624–638). A helper is `background/ended` plus membership in `_open` or `_detached`. Task statuses are three maps: `background.STATUS` (:89 @B), `async_tasks.ENDED` (:67 @B), `results._FURTHER` (:402 @B). agent-runs does it right: `_move` is "the one transition check" (store/runs.py:213 @RF). Smallest step: one enum and one `_become(to)` per class that rejects an invalid move; start with `Background`. |
| **Strategy** | `Offered.call` dispatches five tool names with `if` (background.py:891–910 @B) — a dict of handlers. The update strategy is a string compared in two places (`"interrupt"`/`"enqueue"`, background.py:433, 936; react.py:459 @B) although `conversation.Mode` already is that strategy. |
| **Adapter (capability on the protocol)** | `carries(target)`, `aclose`, `child_cancelled`, `tasks` (section B). |
| **Observer** | Three polls ask the store what the heartbeat and the event log already announce (ARCH-4). C's `Holds(lost=)` and `RunLog(ended=)` are the observer; B does not use them. |
| **Null object** | `_SILENT` is a hand-made null events object for one caller (tools/bridge.py:752 @F) while `events` is consulted through `if again` elsewhere in the same function. Conversely `contextlib.nullcontext()` is used well for "no hold" and "no line" (pipeline.py:203–207 @F, agent.py:1474–1478 @C). |
| **Repository** | The `.carried` store is built from `files.files/read/put/remove` calls inside the pipeline (ARCH-8). |
| **Chain of responsibility** | The "hooks changed this outcome" fact leaves the chain by three side channels: `_HOOKED` (tools/bridge.py:140 @F), an attribute set on a foreign exception (`WORDED`, adapters/openai_agents.py:505 @F), and `_Calling.said` (middleware.py:602 @F). Return it from `bridge.call` once. |

**Ceremony for one implementation**
- `tools.base.Deferring` protocol — one implementer (`_Calling`), one user (ARCH-6).
- `background.Offered` — documented as adapter-neutral with "an OpenAI Agents function tool" (background.py:769–771 @B); only `bridges/background_tasks.py` uses it, and the brief rules out OpenAI background agents. The neutral layer also is not neutral (ARCH-9).
- `CARRIED` as a tuple of channels with generic merge machinery — one channel (ARCH-8).
- `claude_live.Mirror.serve` and `LiveSessions` are single-use but earn their place (they own state); not ceremony.

**Boolean flags that switch behaviour** (the brief asks for two functions or a strategy)
- `_query(last=)`, `_options(kept=)`, `_live(unsent=)`, `_dated(kept=)` — adapters/claude.py:585–592, 1160–1168, 690–699, 734 @F.
- `LiveSession.charged(own=, taken=)` — claude_live.py:1073 @F: four combinations, three used.
- `Helpers.settled(final=)`, `Helpers.attribute(background=)`, `Helpers.finished(cancelled=)` — helpers.py:323, 282 @F; :388 @B.
- `bridge._decided(again=)`, `Replay.calling(again=)` — tools/bridge.py:655–663 @F, journal.py:489 @F.
- `RunStore._placed(started=, answered=)` — store/runs.py:772 @RC.
- `identified(answered=)` — conversation.py:112 @F (acceptable: one line differs).

---

## D. Duplication

**Commands.** `pylint --disable=all --enable=duplicate-code --min-similarity-lines=6` on `src` and on `tests` (pylint 4.1.3 installed with `pip --target` into the scratch folder, Python 3.13), and `npx jscpd --format python --min-lines 6 --min-tokens 60 src tests`, on M, F, B, C; pylint also on the three agent-runs trees.

**Totals.**

| | M | F | B | C |
|---|---|---|---|---|
| pylint groups, src | 1 | 1 | 1 | 1 |
| pylint groups, tests | 6 | 14 | 14 | 14 |
| jscpd clones (src + tests) | 50 | 75 | 76 | 76 |
| jscpd duplicated lines | 498 | 729 | 740 | 736 |

agent-runs: 0 source groups and 1 test group on all three refs; nothing added. The one harness source group is legacy (an `__all__` list re-exported by `trellis.skills` and `harness/skills.py`): leave.

**Clones this round added (tools)**

| # | Locations (@F unless noted) | Size | Kind | Same knowledge? | Home |
|---|---|---|---|---|---|
| T1 | pipeline.py:223–234 ↔ :333–344 | 12 | exact | Yes: the signatures of `_turned` and `_attempt`, one passing every keyword to the other | Remove `_turned`'s pass-through (section A) |
| T2 | adapters/openai_agents.py:213–219 ↔ :233–239 | 7 | exact | Yes: invoke and stream build helpers, agent, options, hooks identically | One `_prepared(target, native_input, run)` in the adapter |
| T3 | tests/integration/test_react_async_subagent.py:608–621 ↔ test_restarts.py:141–154, :307–319, :559–566 | 8–14 | exact | Yes: "a worker dies, its lease lapses, another takes over", including writes to `store._leases` | `tests/support/` helper (and `FileRuns.lapse` already exists at shared_runs.py:55) |
| T4 | tests/integration/test_open_children.py:33–46 ↔ tests/support/killed_server.py:61–75 | 13 | exact | Yes: the test copies its own support module | Import from `tests/support/killed_server.py` |
| T5 | tests/integration/test_create_agent_library.py:50–67 ↔ test_deepagents_library.py:43–60 | 18 | exact | Yes: the `results(events)` helper and reset fixture | `tests/support/` |
| T6 | test_claude_background_session.py:323–335 ↔ :1178–1190; :625–633 ↔ :681–689 ↔ :711–718; :471–477 ↔ :561–567; :1379–1385 ↔ :1511–1517 | 7–13 each | exact | Yes: the same set-up inside one file | Fixtures or parametrize in that file |
| T7 | test_claude_background_session.py:290–297 ↔ tests/unit/test_claude_live.py:127–134 | 7 | exact | Yes: a result-message literal | `tests/support/fake_claude_cli.py` |
| T8 | test_claude_background_session.py:2178–2186 ↔ test_react_async_subagent.py:1262–1270; test_background_by_adapter.py:167–174, 237–243 ↔ test_react_async_subagent.py:427–434, 1198–1204 | 7–9 | near | Yes: the same background scenario asserted per adapter | Parametrize by adapter in test_background_by_adapter.py |
| T9 | test_mcp_servers.py:43–52 ↔ test_openai_agents_mcp.py:49–58; test_function_agent.py:412–419 ↔ test_openai_agents_features.py:437–444, 967–974 | 8–10 | exact | Yes: start an MCP server process; collect events | `tests/support/mcp_server.py` |
| T10 | test_langgraph_library.py:66–76 ↔ test_tool_updates.py:123–133; test_approvals.py:395–402 ↔ tests/unit/test_governance.py:440–447 | 7–10 | exact | Look alike (graph wiring; a literal payload) | Leave |
| T11 | B: test_openai_agents_features.py:151–160 ↔ :213–222. C: test_conversation_line.py:110–116 ↔ :144–150 | 7–10 | exact | Yes | Parametrize |

Worst legacy clones in files this round touched: test_middleware.py:146–180 ↔ test_react_parallel.py:70–101 (34 lines) and test_stored_compare.py:39–64 ↔ test_stored_files.py:44–69 (25 lines). Both are shared fixtures and belong in `tests/support/`.

**What the tools miss (by hand)**

*Near-duplicate functions (same body, different name or constant) — must merge*

| # | Copies | Home |
|---|---|---|
| DUP-1 | Ask a paused child's question through a turn: subagents.py:212–221 @B (the original), claude_live.py:936–951 @F, background.py:715–726 @B. The last two are the same down to the `"background": True` key and the `model_copy` of the answer | One function in subagents.py taking `(runtime, asked, name, background=)` |
| DUP-2 | Stop an attempt: agent.py:606–609 @B, claude_live.py:392–395 @F, background.py:376–381 @B, harness.py:487–490 @C | `Agent.stop_attempt` (ARCH-4) |
| DUP-3 | Child cascades: subagents.py:303–311 @F ↔ pipeline.py:740–765 @F | ARCH-3 |
| DUP-4 | Ref-counted keyed lock: conversation.py:360–375 ↔ :378–412 @C | A small helper in conversation.py |
| DUP-5 | Task record update `{**task, "run_id", "status", "last_checked_at", "last_updated_at"}`: background.py:159–169, 947–948, 1024–1035 @B | `observed()` with an optional status |
| DUP-6 | Claude task messages parsed twice for the same facts: adapters/claude.py:882–903 (`_tasks`) ↔ claude_live.py:865–888 (`_tracked`) @F | One parser in bridges/claude.py returning `(task_id, started/ended, status, summary)` |
| DUP-7 | Token sums over the same four usage keys: claude_live.py:221–225 (`_tokens`), :512–518 (`spent`), adapters/claude.py:1057–1066 (`_spent`) @F | With `SessionCost` (ARCH-13) |
| DUP-8 | `helper("finished", …)` event: helpers.py:372–379 ↔ claude_live.py:337–344 @F | `Helpers.finished` (ARCH-12) |
| DUP-9 | Hold vs lease expiry under a cancel request: store/runs.py:860–880 ↔ :919–932 @RC | ARCH-10 |
| DUP-10 | `holder_lost` sweep, message text included: store/runs.py:908–943 @RF ↔ harness runs.py:694–714 @F | Unavoidable while `LocalRuns` twins agent-runs; share the text through contracts |

*Duplicated knowledge (a constant, key, id format, status set or message defined more than once)*

| What | Places |
|---|---|
| `"interrupted by a newer message of the conversation"` | pipeline.py:268 ↔ background.py:452 @B |
| `"async_subagent"` | background.py:86 ↔ middleware.py:2119 @B |
| The five tool names and their side effects | ARCH-14 |
| `"holder_lost"` | claude_live.py:171, runs.py:708 @B, agent-runs store/runs.py:933 @RF — no constant |
| Resume grace, 60 s | harness runs.py:105 @F ↔ agent-runs config/constants.py:102 @RF |
| "In the line" statuses | line.py:54 @C ↔ store/runs.py:115 @RC |
| The place formula | line.py:75–86 @C ↔ store/runs.py:307–316 @RC (ARCH-15) |
| Poll period 10.0 s | claude_live.py:117 @F, background.py:129, async_tasks.py:64 @B |
| "A turn its caller sent" (`origin is None or kind == "human"`) | adapters/claude.py:766–770 ↔ claude_live.py:829–830 @F |
| "Kept in its caller's process" predicate | store/runs.py:726, 772 @RF; runs.py:407, 425 @F |
| `f"mcp__{SERVER}__"` | 8× adapters/claude.py, 2× claude_live.py, 1× acp.py @F — add `convert.claude.named(tool)` / `bare(name)` |
| `content_key("call", tool, args)` | 15 sites @F, 7 in tools/bridge.py — add one `call_key(tool, args)` in journal.py |
| "Cut off" wording and codes | helpers.py:73, 79 (`HelperUnfinished`), background.py:124–126 (`BackgroundUnfinished`) @B, claude_live.py:120 `LOST` @F |
| Conversation key | `conversation_key(...)` everywhere; one hand-built `f"{tenant}\0{user}\0{thread}"` at agent.py:1475 @C (ARCH-19) |

*Parallel adapter code (the same steps in each adapter)*

| Step | Places | Verdict |
|---|---|---|
| Open the attempt's helpers, run, `helpers.closed(error)` in `finally` | adapters/claude.py:470–490, openai_agents.py:216–228 and 236–261, acp.py:640–650, langgraph.py:228–254 and 263–291 @B | Same knowledge, six copies. One `async with helping(run, name) as helpers:` context manager in bridges/helpers.py. It also gives ARCH-1 a single place to pass "paused" |
| Resume prompts (`RESUMED`, `resumed_call`) | acp.py:98, 984–987 imports them from adapters/claude.py @F | An adapter reaching into another adapter: move both to bridges/ (they are shared) |
| "The framework asks its own user" (`permitted` then `asked`) | Claude calls `permitted` then `asked` separately (adapters/claude.py:1327, 1367 @F); ACP passes `framework=` so `permitted` calls `asked` itself (acp.py:1607, tools/bridge.py:252 @F) | Two ways to do one thing; pick one |
| Keep the conversation's background tasks alive past the turn | claude_live.py, async_tasks.py, background.py | Section 2 |

---

## 4. Well designed — do not change

- **`holding.Holds`** (holding.py @F): a two-method `Holding` protocol, a `holding()` context manager, one heartbeat task per run. It is the right base for everything kept alive; build on it rather than beside it.
- **`line.Line`** (line.py @C): takes only a store, a record and a cache; follows the event log and polls only where no log exists, with backoff and jitter on failure. The docstring states the invariant.
- **agent-runs `_move`** (store/runs.py:213 @RF): one transition check, and the hold is cleared there. Keep new state changes going through it.
- **agent-runs `_placed`** (store/runs.py:772 @RC): the ordering is decided under the key's advisory lock in the transaction's last statement. That part is correct by construction; only its exposure needs work (ARCH-15).
- **Stable ids for child runs** (`stable_id(parent, HELPER, kind, name, task, occurrence)`, helpers.py:208 @F; `run_of`, async_tasks.py:83 @B; the notification run id, background.py:593 @B): retries and other replicas find the same record.
- **`Journal.handed_on`** as the durable statement "this child is someone else's to end" — the right idea; it only needs one consumer (ARCH-3).
- **`background.delivered`** shared by the harness tasks and the Deep Agents tasks (background.py:502 @B), and the "started once per outcome, run by the server whose start made it" rule in `notify` (:578–618).
- **`outbound.validate_url`** (outbound.py @B): pure, small, one job.
- **`handed_to`** (adapters/openai_agents.py:671–684 @F): reads an SDK private, and refuses loudly when it is gone. The model for ARCH-21.
- **`Harness.aclose`** (harness.py:431–455 @F): closes every part whatever another raises, then raises the first error.
- **Refusing unattended calls** in a live Claude session (`UNATTENDED`, claude_live.py:105, 719–730 @F): fails closed and tells the next run how many were refused.
- **The comments.** Nearly every non-obvious line says why. That is why this review could be done by reading.

---

## 5. Open questions for the architect

1. **Is a harness-side retry of the graph's own tools a product requirement?** If not, ARCH-6 is mostly deletion. If yes, it needs an entry in DECISIONS.md, because it contradicts brief §1.
2. **Is a hold a lease?** After RC a hold differs from a lease only in "does not fence writes" and "ends instead of requeueing". Should agent-runs have one liveness mechanism with a lapse policy, or are two columns and two sweeps the intended end state (ARCH-10)?
3. **Should agent-runs end a run's children when the run ends?** Today the harness does it from the client, twice, non-atomically (ARCH-3, ARCH-1). A server-side cascade with a "kept" marker would make it true by construction, but it is a contract change.
4. **Is `start_async_task` a write when remote and a read when local on purpose?** (ARCH-14)
5. **Does the Claude Agent SDK give an in-process MCP handler the tool-use id** (for example in the request's `_meta`)? If it does, ARCH-16 and the `announced/claimed` queue can be replaced by the id.
6. **Why is the conversation's task list kept in the file store rather than derived from the run store?** The runs already exist there with `BACKGROUND`/`HELPER` metadata; a filter on that metadata would remove the `.carried` protocol (ARCH-8). Is the blocker only that `iterate` cannot filter on metadata?
7. **May `created_at` be rewritten?** `_placed` moves it (store/runs.py:813 @RC). Any consumer that treats it as immutable (webhooks, retention, reports) now sees a different value than the one first returned.
8. **What does `bridges/` mean?** As "what several adapters share" it is false for four of six modules. Rename the rule or move the modules (ARCH-9).
9. **Which should win when B and C fold: B's watchers or C's `Holds(lost=)`?** I recommend C's (ARCH-4); B then loses `Tasks._watch` and half of `AsyncTasks._watch`.
10. **Should a never-held `RUNNING` run with a `concurrency_key` be ended after a grace?** It protects the line (ARCH-5) but ends runs of any client that starts a keyed run and never heartbeats.
