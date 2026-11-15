# Targeted refactor plan: one owner for stopping, one cascade, one `helping()`, and budgets in CI

Measured on `origin/main` at `0047ecfa` (2026-11-15). Nothing in `src/` or `tests/` changes in
this commit. The refactor runs **after** PRs #273, #275 and #276 and the branches `-scenfix`,
`-decide2`, `-onerule`, `-livedirect` and `-sdkfull` have merged. Section 6 lists how each of
them touches the same functions.

The approved scope, and nothing more:

1. One owner of "stop an attempt", and of "its run ended in the store while it worked here".
2. One cascade that ends a run's children when the run ends.
3. One `helping()` for the six adapter copies that open and close an attempt's helpers.
4. Duplicate-code and complexity budgets in CI, so the duplication cannot grow back.

All paths below are relative to `src/trellis/harness/`. A citation `file:function L<a>-<b>` is
the function's span on `0047ecfa`; "at L<n>" is the line that does the thing.

## 0. Re-check this plan on the newer main (minutes)

The plan names every copy by function as well as by line, so it survives line drift. On the
newer main, before step 1, run these from the repository root. They use the standard library
only, so any Python 3.12+ works.

```bash
python docs/research/refactor-plan-2026-11-15/recheck.py numbers   # lines, complexity, duplicates
python docs/research/refactor-plan-2026-11-15/recheck.py copies    # where each copy is now
# for each branch still open: which functions it shares with this refactor
python docs/research/refactor-plan-2026-11-15/recheck.py overlap origin/main origin/<branch>
python docs/research/refactor-plan-2026-11-15/recheck.py overlap <main sha> <unpushed checkout dir>
```

[`recheck.py`](refactor-plan-2026-11-15/recheck.py) finds copies by what they do, not by name:
an attempt stopped by hand (`.cancelled =` next to `.cancel()`), a refused ending write read as
"ended meanwhile" (`ConflictError`), a run's children listed (`iterate(parent_run_id=...)`), a
keeper stopped for a child, and an attempt's helpers opened by hand (`Helpers(...)`,
`helpers.closed`). The output on `0047ecfa` is what sections 2-5 cite. When a count differs on
the newer main, a branch added a copy: fold that copy in the same step as its siblings.

## 1. Numbers on main today, and after

| Measure | Today (`0047ecfa`) | After steps 1-12 | How measured |
|---|---|---|---|
| Lines of src (`src/trellis`, 125 files) | 54,456 (47,537 non-blank) | ≈ 54,490 (about +30) | `recheck.py numbers` |
| Functions / mean cyclomatic complexity | 2,946 / 4.1185 | ≈ 2,950 / ≤ 4.12 | as `agent-memory-service/tests/unit/test_complexity_budget.py` counts |
| Functions above complexity 15 | **62** | **62** | same |
| Worst function | cc 34, `adapters/acp.py:926 ACPAdapter._attempt` | same | same |
| Longest function | 198 lines, `react.py:265 ReAct` | same | same |
| Duplicate blocks (> 4 kept lines, within a file too) | **39** | **33** | `recheck.py numbers` (pylint's Symilar rules, stdlib) |
| pylint 3.3.9 `duplicate-code`, `min-similarity-lines` 4 / 5 / 6 | 11 / 3 / 1 | 8 / 2 / 1 | cross-check, between files only |
| radon 6.0.1 `cc`, above 15 (harness only) | 64 | 64 | cross-check |
| ruff `C901`, `max-complexity = 15` | 2 | 2 | cross-check (counts nested functions apart) |

The lines go **up** by about 30. Each fold deletes 4-30 lines in each place it removes a copy.
Each new owner (`stopping.py`, `children.py`, `helping()`) adds a module or function with its
docstring. What the refactor buys is single ownership, not size.

Complexity above 15 stays at 62. None of the functions this refactor touches is above 15,
apart from `bridges/live.py:395 Kept.served` (cc 16), which this plan leaves as it is (B7 in 2.1).
Two of them are close to 15: `pipeline.py:825 _cancelled` is cc 14 and `pipeline.py:946
_concluded` is cc 13. Step 11 must not add a branch to either. Their new work goes in a
helper (4.4).

Duplicate blocks fall from 39 to 33. The six blocks removed are the six pairs among the
`helpers.closed(error, paused=helpers.suspended(error))` tails of item 3. These are
`acp.py:910 ↔ claude.py:590`, `langgraph.py:273 ↔ langgraph.py:314`, and `acp.py:910` and
`claude.py:590` each against `langgraph.py:273` and `:314`.

None of the copies in items 1 and 2 shows up as a line-for-line duplicate. Each has different
names, guards and statuses, which is why no tool caught them and why sections 2 and 4 list them by
hand.

### 1.1 The 39 duplicate blocks on main (length in kept lines, then the two places)

```text
 11  agent.py:502  agent.py:541
 11  middleware.py:2058  middleware.py:3342
 10  agent.py:489  agent.py:503
 10  agent.py:489  agent.py:542
 10  pipeline.py:230  pipeline.py:263
  9  agent.py:489  agent.py:597
  9  agent.py:503  agent.py:597
  9  agent.py:542  agent.py:597
  7  skills.py:1028  ../skills.py:27
  6  adapters/acp.py:910  adapters/claude.py:590  *
  6  adapters/langgraph.py:273  adapters/langgraph.py:314  *
  6  agent.py:1643  subagents.py:194
  6  agui/__init__.py:372  agui/__init__.py:406
  6  agui/threads.py:215  agui/translate.py:34
  6  middleware.py:2074  middleware.py:2107
  5  adapters/acp.py:1693  adapters/acp.py:2127
  5  adapters/acp.py:719  adapters/claude.py:399
  5  adapters/acp.py:749  adapters/claude.py:423
  5  adapters/acp.py:910  adapters/langgraph.py:273  *
  5  adapters/acp.py:910  adapters/langgraph.py:314  *
  5  adapters/claude.py:590  adapters/langgraph.py:273  *
  5  adapters/claude.py:590  adapters/langgraph.py:314  *
  5  agent.py:494  agent.py:1650
  5  agent.py:508  agent.py:1650
  5  agent.py:547  agent.py:1650
  5  agent.py:596  agent.py:1642
  5  agui/translate.py:167  subagents.py:95
  5  bridges/acp_live.py:261  bridges/claude_live.py:324
  5  bridges/acp_live.py:699  bridges/claude_live.py:570
  5  bridges/acp_live.py:700  bridges/claude_live.py:548
  5  clients/bifrost.py:89  clients/bifrost.py:109
  5  evals.py:962  harness.py:432
  5  pipeline.py:1290  pipeline.py:1340
  5  react.py:169  react.py:201
  5  react.py:200  react.py:332
  5  react.py:357  react.py:385
  5  react.py:357  react.py:632
  5  react.py:385  react.py:632
  5  tools/base.py:439  writes.py:232
```

Paths are relative to `src/trellis/harness/`, and `../skills.py` is `src/trellis/skills.py`.
`*` marks the six this refactor removes. The others are out of scope. Most are keyword
pass-throughs (`agent.py` `run`/`stream`/`start` calls, `pipeline.py` `_turned` → `_attempt`);
`skills.py` is the legacy `__all__` re-export. The budget (5.1) keeps them from growing.

### 1.2 The 62 functions above complexity 15 on main

```text
34 adapters/acp.py:926 ACPAdapter._attempt             19 middleware.py:3017 FitWindow._fitted
33 agent.py:1235 Agent.remembered                      19 outbound.py:26 validate_url
31 bridges/claude_live.py:598 LiveSession._took        19 governance/shell.py:561 _approved
30 tools/bridge.py:725 _decided                        19 governance/shell.py:1242 _placed
29 react.py:265 ReAct                                  19 governance/shell.py:728 _writes
27 pipeline.py:391 _attempt                            18 middleware.py:528 governed_call
25 tools/bridge.py:399 _called                         18 tools/query.py:1032 DuckDB._read
25 middleware.py:3116 _narrowed                        18 tools/bridge.py:476 _ran
24 bridges/claude.py:177 after_theirs                  18 middleware.py:3617 RunCheckpointer.put
24 pipeline.py:1564 _with_facts                        18 adapters/claude.py:1692 _recorded
24 tools/query.py:502 query                            18 tools/sources.py:230 documented
23 skills.py:534 _make_backend_class                   18 adapters/claude.py:1701 _recorded.post_tool_use
23 harness.py:133 Harness.__init__                     18 governance/shell.py:1199 _reads
23 adapters/claude.py:686 ClaudeAdapter._query         17 adapters/acp.py:803 ACPAdapter.extract
23 middleware.py:3388 _cut                             17 agui/threads.py:64 Threads.stream
23 tools/sources.py:464 _operation_tool                17 adapters/claude.py:928 _subagents
21 agent.py:853 Agent._resolution                      17 tools/base.py:216 data_of
21 adapters/claude.py:1588 _asked                      17 bridges/acp_live.py:558 ACPSession._logged
21 adapters/claude.py:1600 _asked.can_use_tool         16 adapters/openai_agents.py:453 governed_agents
21 bridges/claude_live.py:770 LiveSession.charged      16 tools/query.py:779 _answer
20 conversation.py:214 earlier                         16 bridges/claude_live.py:412 LiveSession._hook
20 adapters/langgraph.py:323 LangGraphAdapter.extract  16 middleware.py:400 declare_files
20 agent.py:1116 Agent.tools_for                       16 journal.py:849 Replay.passed
19 tools/base.py:343 execute                           16 adapters/langgraph.py:788 _continued
19 agent.py:1686 Agent._start                          16 bridges/claude_live.py:422 LiveSession._hook.relay
19 evals.py:1039 _item                                 16 bridges/live.py:395 Kept.served
19 tools/ask.py:158 ask_question                       16 bridges/session.py:57 MemorySession.get_items
19 middleware.py:3332 _remembered_less                 16 tools/links.py:160 links
19 schedules.py:407 _schedule_task                     16 governance/shell.py:508 _commands
19 adapters/claude.py:1522 _decide                     16 adapters/acp.py:532 _kimi_rules
19 tools/query.py:1303 _json_rows                      16 repository.py:160 _value
```

Counting: one per `if`, loop, `except`, `with`, `assert`, conditional expression,
comprehension and `match` case, and one per extra operand of `and`/`or`. A nested function's
branches count toward its parent too, as in the memory service's test, so its numbers compare
across the two repositories. radon finds 64 above 15, because it counts nested functions apart
and boolean operators differently. ruff `C901` (mccabe) finds 2:
`agui/__init__.py:514 _router` (16) and `skills.py:534 _make_backend_class` (20).

## 2. Item 1: one owner of "stop an attempt" and "it ended elsewhere"

### 2.1 Every copy today

**A. Stopping an attempt:** set why on the runtime, then cancel the task it runs in. There
are three copies, with three different rules for which reason is kept:

| # | Copy | What it does | How it differs |
|---|---|---|---|
| A1 | `agent.py:Agent.stop` L643-655 (at L653-654) | `runtime.cancelled = reason; running_in.cancel()` only `if runtime.cancelled is None`; with no attempt here, `self.turns.cancel(run_id, reason)` (a run waiting its turn) | **first reason wins**; returns the future to wait on; also covers a run waiting in its line |
| A2 | `bridges/live.py:Kept.cancel` L385-392 (at L391-392) | `self.stopped = reason`; when an attempt runs, `runtime.cancelled = reason; running_in.cancel()` | **last reason wins** (overwrites); never touches the line; returns nothing |
| A3 | `background.py:Tasks.stop` L417-426 (at L425-426) | finds the agent in its own map, then `runtime.cancelled = runtime.cancelled or reason; running_in.cancel()` | **first reason wins** but cancels the task again on a second stop; never touches the line |

These call A1, which is correct and stays: `Harness._lost` (`harness.py` L494-509, at
L508-509), `Harness._renewed` (L511-523, at L523) and `Agent.cancel` (L614-641, at L631).
`pipeline.py:_attempt` L472-473, `except RunCancelled: runtime.cancelled = str(exc)`, is not a
stop. It records why `Kept.served` refused to start (below). The pipeline's own way of ending a
stopped attempt (`_cancelled` L825-858, `_concluded` L946-1009) reads `runtime.cancelled` and is
the consumer, not a copy.

**B. Deciding "the run store ended it while it worked here".** There are four detectors and
four places that decide what to do once it is known:

| # | Copy | Decision | Notes |
|---|---|---|---|
| B1 | `harness.py:Harness._lost` L494-509 | reads the record; `record.status.final` → `CANCELLED_ELSEWHERE` if `CANCELLED` else `ENDED_ELSEWHERE` (constants `harness.py` L85-87) → `agent.stop` on every agent | the only place the two reasons are chosen. Fired by `holding.py:Holds._beating` L155-170 (at L167, refused heartbeat) and by `runlog.py:RunLog._send` L103-122 (at L114-118, refused event write) through `pipeline.py:_emitter` L1106-1111 |
| B2 | `pipeline.py:_left_the_line` L326-354 (at L340-354) | a run that took no turn: `turns.cancelled(run_id)` → settle `CANCELLED` here; else read the record and report the store's ending | PR #276 (`-decide15`) changes this exact decision: it reads the store first, and settles `CANCELLED` only `if reason is not None and not found.status.final` |
| B3 | `pipeline.py:_settle_cancelled` L1620-1660 (at L1648) | `finish(CANCELLED)` refused with `LeaseLostError`/`ConflictError` → "ended or taken over elsewhere", returns `False` (no cascade) | the same knowledge, refused write = ended elsewhere, also appears in 11 `suppress(ConflictError)` sites (B6) |
| B4 | `line.py:Line.ahead` L160-167 / `Line.turn` L200-223 | `ahead()` is `None` once the run is out of `IN_LINE` (the store ended it while it waited), and `turn()` returns `False` | reads the line's listing, not one record: **stays** (it is the line's own question). The consumer is B2 |
| B5 | `conversation.py:Turns.cancel` L486-499, `Turns.cancelled` L510-514 | marks a waiting run cancelled (`Here.cancelled`); B2 reads it | the waiting half of A1; **stays**, reached only through `Agent.stop` |
| B6 | the "it ended meanwhile" suppressions: `pipeline.py:_arrived` L313, `_children_ended` L890; `subagents.py:cancel_children` L313; `background.py:Tasks.aclose` L436, `follow_up` L528, `stop` L593, `abandoned` L879; `bridges/live.py:Kept.abandoned` L377; `bridges/async_tasks.py:AsyncTasks._closed` L269, `AsyncTasks.stopped` L325 | `with contextlib.suppress(ConflictError[, LeaseLostError])` around an ending or cancel write | 11 copies. Some suppress `LeaseLostError` too, some do not; all mean "the store ended it first" |
| B7 | `bridges/live.py:Kept.served` L395-429 (at L403-405) | `self.stopped or record is None or record.final` → `RunCancelled(self.stopped or CANCELLED_FIRST)` | a kept run's attempt refuses to start; **stays** (cc 16; its reason text is its own, `bridges/live.py` L99). It is the producer of the `RunCancelled` above |
| B8 | `bridges/live.py:Kept._answered` L343-359 (at L354), `Keeper.asks` L605-634 (at L617); `bridges/async_tasks.py:AsyncTasks._watch` L210-234 (at L227-231) | `status is not PAUSED` → stop waiting (live); `CANCELLED` → stop the task, other final → close it (async) | polls of a run that **this process keeps outside any attempt**. These are "no longer waits for a person" and "ended on another server" for a task. **Stay**, because they are not attempts. `_watch` keeps its own poll (the task's server must be read anyway) |

Not copies, and left alone: `pipeline.py:_concluded` at L959 (`runtime.lease_lost`, set by
`runtime.py:Runtime.progress` at L652), which concerns a worker's lease, and
`pipeline.py:_recorded` L1506-1524, which reads back a refused write and must never swallow
`LeaseLostError`. The surfaces' cancels (`a2a/executor.py` L132, `agui/__init__.py` L380) and
`schedules.py` L456/L585 also suppress `ConflictError`, for a different reason: a person asked
twice, or a schedule refused. `recheck.py copies` lists them, and they stay.

The stop reason "interrupted by a newer message of the conversation" is defined twice:
`conversation.py` L74 and `background.py` L542.

### 2.2 Where it lives: `src/trellis/harness/stopping.py` (new)

```python
"""What stopping a run's work in this process means: one owner for every place that stops an
attempt, and for deciding that the run store ended a run while this process worked on it."""

CANCELLED_ELSEWHERE: Final = "cancelled from another process"  # moved from harness.py
ENDED_ELSEWHERE: Final = "ended in the run store while it worked here"  # moved from harness.py


def stop_attempt(runtime: Runtime, reason: str) -> asyncio.Task[Any]:
    """Stop the attempt ``runtime`` runs, for ``reason``: the first reason given stands and its
    task is cancelled once; later stops change nothing. The task, to wait on."""


def ended_elsewhere(record: RunRecord | None) -> str | None:
    """Why what this process does for a run stops because the run store ended it
    (CANCELLED_ELSEWHERE, ENDED_ELSEWHERE); None while the store has not ended it, or has no
    such run."""


async def unless_ended(write: Awaitable[object]) -> bool:
    """Make an ending or cancel write; False when the store had ended the run first, or another
    holder has it (ConflictError, LeaseLostError) - the write is then nobody's to make."""
```

`Agent.stop(run_id, reason)` keeps its signature and stays the one public way to stop a run's
work here. Its body becomes `runtime is None → self.turns.cancel(...)`, else
`stopping.stop_attempt(runtime, reason)`.

### 2.3 Each call site's change

| Site | Change |
|---|---|
| `agent.py:Agent.stop` | body as above (L652-655 → one call) |
| `bridges/live.py:Kept.cancel` | `self.stopped = reason; self.agent.stop(self.run_id, reason)` (L389-392 → one call). **Behaviour:** a second stop no longer overwrites the first reason |
| `background.py:Tasks.stop` | `agent = self._agents.get(run_id); if agent is not None: agent.stop(run_id, reason)`. **Behaviour:** a task's run that waits its turn in its line here now leaves the line at once (`Turns.cancel`), instead of after the run ahead leaves. Its ending is unchanged: the store already has it cancelled, and it reports that |
| `harness.py:Harness._lost` | `why = stopping.ended_elsewhere(record)`; `if why is not None: for agent ...: agent.stop(run_id, why)`. The constants move; `harness.py` imports them (no other importer in src or tests) |
| `pipeline.py:_left_the_line` | the decision becomes `reason is not None and stopping.ended_elsewhere(found) is None`. This is PR #276's rule written with the owner: rebase onto it, do not re-derive it |
| `pipeline.py:_settle_cancelled.write` | `return await stopping.unless_ended(agent.harness.runs.finish(...))`, keeping the `log.info` on `False` |
| The 11 B6 sites | `await stopping.unless_ended(<the write>)`. Each site that suppressed only `ConflictError` makes a `cancel` (no worker, so no `LeaseLostError`). No behaviour change |
| `background.py` L542 | `from trellis.harness.conversation import INTERRUPTED`; delete the copy |

These stay as they are: `Kept.served`, `Kept._answered`, `Keeper.asks`, `AsyncTasks._watch`,
`Line.*`, `Turns.*`, `_recorded`, `_concluded`'s lease check, and the surfaces' and schedules'
suppressions.

### 2.4 Tests that pin it today, and tests to add

Pinned on main:

- A1, B1: `integration/test_conversation_line.py::test_a_refused_hold_stops_only_a_run_cancelled_in_the_store` (calls `h._lost` directly: not in the store, still running, cancelled), `::test_a_run_the_store_cancelled_unheld_stops_at_its_next_event` (the `RunLog` path), `::test_conv04_a_cancel_from_another_server_stops_the_call_at_the_next_heartbeat`, `::test_a_run_cancelled_while_it_waits_its_turn_never_runs` (B2, B5); `integration/test_scenario_findings.py::test_scn13_a_run_the_store_ended_stops_where_it_works` (`ENDED_ELSEWHERE`); `integration/test_run_limits.py::test_a_run_cancelled_elsewhere_is_stopped_by_its_worker`; `unit/test_worker.py::test_a_stream_closed_after_its_run_ended_elsewhere_writes_nothing`.
- A2, B7: `integration/test_claude_background_session.py::test_cancelling_a_background_subagents_run_stops_that_subagent_only`, `::test_a_background_run_cancelled_before_its_first_attempt_began_stops_its_task`, `::test_a_turn_cancelled_before_its_background_subagents_run_began_stops_it_too`.
- A3: `integration/test_react_background.py::test_cancelling_the_turn_cancels_its_tasks`, `::test_a_tasks_run_ended_elsewhere_stops_its_attempt_here`.
- B3, B6: `integration/test_durable.py::test_a_lost_lease_stops_the_run_without_writing`, `integration/test_pipeline_edges.py::test_a_lost_lease_on_the_write_is_never_read_as_a_conflict` (`_recorded` keeps refusing `LeaseLostError`), `integration/test_react_async_subagent.py::test_h_a_server_that_refuses_the_cancel_is_said_and_the_run_ends_all_the_same`.
- B8: `integration/test_react_async_subagent.py::test_i_a_tasks_run_cancelled_elsewhere_cancels_the_task_on_its_server`.
- Arriving with PR #273: `integration/test_run_limits.py::test_a_run_stopped_twice_ends_once_with_the_first_reason` (A1's first-reason rule) and `integration/test_conversation_line.py::test_a_run_the_store_ended_while_it_waited_its_turn_never_runs`, `::test_a_waiting_run_whose_hold_the_store_refuses_stops_and_overwrites_nothing` (B2). PR #276 adds B2's new rule with its tests in the same file.

To add (each in the commit named in section 7):

- T1 `tests/unit/test_stopping.py`: `ended_elsewhere` is `None` for no record and for `QUEUED`/`RUNNING`/`PAUSED`; `CANCELLED_ELSEWHERE` for `CANCELLED`; `ENDED_ELSEWHERE` for `SUCCESS`/`ERROR`/`TIMEOUT`/`PARTIAL`. `unless_ended` returns `True`, returns `False` on `ConflictError` and on `LeaseLostError`, and lets any other error through. `stop_attempt` called twice keeps the first reason and cancels once.
- T2 `tests/unit/test_claude_live.py`: a kept run stopped twice ends with the **first** reason (the A2 behaviour change; it fails on main by design, so it lands with the change).
- T3 `tests/unit/test_background.py`: a task's run waiting its turn here, stopped by `Tasks.stop`, leaves the line without waiting for the run ahead (the A3 behaviour change).

## 3. Item 3: one `helping()` for the six adapter copies

Item 3 goes before items 1 and 2 in the step order, because it is the most mechanical.

### 3.1 The six copies

Each one creates `Helpers(run.runtime, self.name)` (which sets `runtime.helpers`), runs the
framework, and in a `finally` or `except BaseException` calls
`helpers.closed(error, paused=helpers.suspended(error))`.

| # | Copy | How it differs |
|---|---|---|
| H1 | `adapters/claude.py:ClaudeAdapter._helped` L569-594 | before running, `helpers.attribute(found.task_id, None)` for a background subagent's run (`live.attended`); per message `_subagents(helpers, message)`; async generator |
| H2 | `adapters/acp.py:ACPAdapter._turns` L892-914 | the bare shape; async generator |
| H3 | `adapters/openai_agents.py:OpenAIAgentsAdapter.invoke` L249-267 | `Helpers` made in `_prepared` L391-399 (the governed agent needs it); `MaxTurnsExceeded` is an answer, not an error; closes in `except BaseException` and, on success, through `_closed` L444-450, **which leaves the helpers open when the SDK paused on its own approval** (`result.interruptions`) |
| H4 | `adapters/openai_agents.py:OpenAIAgentsAdapter.stream` L269-300 | as H3, and in `finally` waits for the SDK's `run_loop_task` before closing on error; yields `Output(result)` after closing |
| H5 | `adapters/langgraph.py:LangGraphAdapter.invoke` L244-278 | the bare shape, and after `closed` it calls `self.tasks.cancelled(run.runtime)` (item 2 removes that call) |
| H6 | `adapters/langgraph.py:LangGraphAdapter.stream` L280-321 | as H5; yields `Output(...)` after the `finally` |

Deep Agents and `ReAct` run through H5 and H6: a Deep Agent or a `ReAct` graph is a LangGraph
graph. Their per-call helper code is a different thing. `middleware.py:_helped` L1208-1236
handles Deep Agents' `task` tool, which is also `ReAct`'s sub-agents. The others are
`tools/convert/openai_agents.py:_helped` L353-402 (an OpenAI agent tool),
`adapters/claude.py:_briefed`/`_subagents`, and `adapters/acp.py:Deciding._helpers_started`/`_helpers_ended`.
That is the call-level start → work → end of one helper. It is not in the approved scope, and
two of those copies are message-driven (they start and end in different callbacks), so they
cannot share a context manager anyway.

### 3.2 Where it lives: `bridges/helpers.py`, next to `Helpers`

```python
@contextlib.asynccontextmanager
async def helping(runtime: Runtime, framework: str) -> AsyncIterator[Helpers]:
    """The attempt's helpers, open while the block runs (``runtime.helpers``) and closed as it
    ends: with the error it ended with - a pause or a stop leaves them to the next attempt
    (:meth:`Helpers.suspended`) - or with none, unless the block left them open
    (:attr:`Helpers.left_open`: the run pauses on an approval of the framework's own that one
    of them waits on)."""
    helpers = Helpers(runtime, framework)
    error: BaseException | None = None
    try:
        yield helpers
    except BaseException as exc:
        error = exc
        raise
    finally:
        if error is not None or not helpers.left_open:
            await helpers.closed(error, paused=helpers.suspended(error))
```

`Helpers` gains one attribute, `left_open: bool = False`, which replaces `_closed`. An async
generator body that is closed early raises `GeneratorExit` into the `async with`, exactly as it
reaches today's `except BaseException`, so a stopped stream closes its helpers as paused, as
today.

### 3.3 Each call site's change

- H1: `async with helping(run.runtime, self.name) as helpers:` around the `attribute` call, the
  source, `aclosing(source)` and the loop. Delete `error` and the try/except/finally.
- H2: the same, without `attribute`.
- H3: `options = _options(...)` and `gated = await _gated(run)` stay **before** the block, so a
  bad `framework_options` still fails before any helper work. That is today's order: no
  `_left_open` read on an attempt that never ran. Inside the block:
  `agent = self._agent(target, run, helpers, gated)`, the run, and
  `helpers.left_open = bool(result.interruptions)`. `_prepared` loses `Helpers(...)` and takes
  `helpers`. Delete `_closed`.
- H4: as H3. The `finally: await asyncio.wait({streamed.run_loop_task})` stays inside the
  block, so the SDK's tool calls end before the helpers close. `yield Output(result)` moves
  after the block.
- H5, H6: `async with helping(run.runtime, self.name) as helpers:`. Item 2's step removes the
  `self.tasks.cancelled(...)` line. If item 3 lands first, keep it as the first statement after
  the block.

### 3.4 Tests that pin it today, and tests to add

Pinned on main:

- `closed`/`suspended`: `unit/test_bridge_helpers.py::test_a_pause_or_a_crash_leaves_the_helper_to_the_next_attempt`, `::test_a_failed_run_ends_its_open_helpers_as_errors`, `::test_a_helper_an_earlier_attempt_left_open_and_the_next_did_not_take_up_ends`, `::test_a_background_helper_of_a_run_that_failed_ends_with_why_the_run_did`.
- H1: `integration/test_open_children.py` (all seven: out of time, a stalled model call, cancelled, the worker stopping it, a background subagent outliving its turn); `integration/test_native_helpers.py::test_a_claude_subagent_is_a_child_run_its_turns_in_memory`, `::test_a_claude_subagent_cut_by_a_crash_is_ended_by_the_next_worker`.
- H2: `integration/test_acp.py::test_a_failed_sub_agent_ends_its_child_run_failed`, `::test_each_swarm_member_is_a_child_run`; `integration/test_acp_background.py::test_with_no_log_of_the_agent_a_background_sub_agent_is_cut_off_with_its_attempt`.
- H3, H4: `integration/test_native_helpers.py::test_an_agent_tool_paused_inside_is_the_same_child_after_the_resume`, `::test_an_agent_tool_that_fails_is_a_failed_child`; `integration/test_openai_agents_features.py::test_an_sdk_approval_inside_an_agent_tool_continues_the_agent_inside`, `::test_an_sdk_approval_inside_an_agent_tool_resumes_in_another_process` (these pin `left_open`), `::test_a_cancel_stops_a_call_under_way_in_the_agent_handed_off_to`.
- H5, H6 (Deep Agents, `ReAct`): `integration/test_native_helpers.py::test_a_deep_agents_sub_agent_is_a_child_run_its_turns_in_memory`, `::test_a_deep_agents_sub_agent_that_fails_is_a_failed_child`, `::test_a_react_helpers_approved_call_is_feedback_on_its_run`.
- The 100% line-and-branch gate already makes some test reach every line and branch of H1-H6.

To add:

- T4 `tests/unit/test_bridge_helpers.py`: `helping` closes with `None` on success; with the exception on failure (helpers end `ERROR`); as paused on `CancelledError` and `GeneratorExit` (holds released, helpers open); and with `left_open` it does not close on success but does close on an error.

## 4. Item 2: one cascade that ends a run's children

### 4.1 Every copy today

| # | Copy | Which children | How each ends | Keepers stopped | Called from |
|---|---|---|---|---|---|
| C1 | `subagents.py:cancel_children` L302-318 | every child **not final** (`QUEUED`, `PAUSED`, `RUNNING`); none spared | `runs.cancel(reason)`, `ConflictError` suppressed | adapter `child_cancelled` (only LangGraph has one: `adapters/langgraph.py` L198-201 → `AsyncTasks.stopped`) **and** `background.stop` | `Agent.cancel` (`agent.py` L640), `Agent._continue` (L950-955, a person's cancel of a paused run) |
| C2 | `pipeline.py:_children_ended` L868-897, with `_open_children_ended` L1012-1033 | only `RUNNING` children (`iterate(status=RUNNING)`); spares `kept` at the top level only | `finish(status, error)` when the parent failed or timed out, else `cancel(reason)`; `ConflictError` and `LeaseLostError` suppressed | `background.stop` only, **never** `child_cancelled` | `_cancelled` L852-857 (kept = `journal.handed_on`), `_concluded` L1006-1007 → `_open_children_ended` (kept = `handed_on`, plus `background.of_turn` unless cancelled; reason `ENDED_FIRST` or the cancel's; errors logged, the result stands) |
| C3 | `background.py:abandoned` L869-881 | one named child: the task whose question the person cancelled | `runs.cancel(reason)` | `background.stop` | `Agent._continue` (L957) |
| C4 | `bridges/live.py:Keeper._detach` L580-592 (at L585-589) | the session's background subagents whose parent is the turn's run, when the turn's runtime was cancelled | its attempt stopped (`Kept.cancel`); the run then ends `CANCELLED` from that attempt | itself | every Claude/ACP turn's detach |
| C5 | `bridges/async_tasks.py:AsyncTasks.cancelled` L300-307, from `adapters/langgraph.py` L278 and L319 (`finally` of `invoke`/`stream`) | the Deep Agents async tasks this run started, when its runtime was cancelled | `AsyncTasks.stopped` L309-328: cancelled on its server, then `runs.cancel` | itself | every LangGraph attempt's end |

The rule today, read from all five together: **a run that is cancelled ends all its children,
the ones handed on past the attempt included.** C2 spares the handed-on ones, and then C1 (via
`Agent.cancel`), C4 or C5 ends them. A run that ends any other way spares the helpers it handed
on and its conversation's background tasks. The copies disagree on three points:

- PAUSED and QUEUED children: C1 ends them, C2 does not.
- The adapter hook: only C1 calls it.
- Who stops the live and async keepers: C4 and C5, from inside the adapters.

`Agent.cancel` on a running turn runs C2 and then C1 (ARCH-3 in the earlier review).

### 4.2 Where it lives: `src/trellis/harness/children.py` (new)

```python
#: why a child's keeper stops when its parent's ending names no reason (today two texts:
#: "its parent was cancelled" in C1, "its parent ended" in C2, for ``background.stop`` only)
PARENT_ENDED: Final = "its parent ended"


async def end_children(
    agent: Agent,
    run_id: str,
    tenant: str,
    *,
    status: RunStatus,
    reason: str | None = None,
    error: AgentError | None = None,
    kept: Collection[str] = (),
) -> None:
    """The run ``run_id`` ended ``status``: each of its children but those in ``kept`` ends with
    it (:func:`end_child`) - failed with ``error`` when the run failed or ran out of time, else
    cancelled with ``reason``. A child that ended already is left as it is (one listing of the
    children, as ``cancel_children`` reads them; final ones skipped)."""


async def end_child(
    agent: Agent,
    run_id: str,
    tenant: str,
    *,
    status: RunStatus,
    reason: str | None = None,
    error: AgentError | None = None,
) -> None:
    """One child run ends - unless the store ended it first (``stopping.unless_ended``) - what
    this process keeps at work for it stops (:func:`stop_kept`, with ``reason`` or
    :data:`PARENT_ENDED`), and its own children end with it (:func:`end_children`, none
    kept)."""


async def stop_kept(agent: Agent, run_id: str, reason: str) -> None:
    """What this process keeps at work for run ``run_id`` outside any attempt of its caller
    stops: a harness background task's attempt (``background.Tasks.stop``) and the adapter's
    keepers (its ``child_cancelled``: a live session's background subagent, a Deep Agents async
    task). A run nothing keeps here is no change."""
```

**One rule:** every child that has not ended ends with its parent, apart from those listed in
`kept`. `kept` is computed in one place, the pipeline's
`_children_kept(runtime, journal, *, cut_off: bool) -> list[str]`, and has three cases:

- `[]` when someone cancelled the run (`runtime.cancelled` is set);
- `journal.handed_on` when the attempt was cut off because its caller went away (a closed
  stream: `_cancelled` with no `runtime.cancelled`). This is today's rule: its handed-on
  helpers go on, and its turn's background tasks end;
- `[*journal.handed_on, *background.of_turn(runtime)]` for every other ending. That gives today's outcome from one function. It is still one listing per run: C2 asks the store for `RUNNING` children, while `end_children` lists them all and skips the final ones, as C1 does.

The adapters' keepers get one protocol method, which LangGraph already has:
`async def child_cancelled(self, run_id: str, reason: str | None) -> None`. Claude and ACP each
gain one that delegates to a new `LiveSessions.cancel(run_id, reason)` in `bridges/live.py`:
`found = self.attended.get(run_id); if found is not None: found.cancel(reason)`.

### 4.3 What it changes

| | Today | After |
|---|---|---|
| PAUSED/QUEUED child of a run that times out, fails or answers | left as it is (C2 lists `RUNNING` only), so it waits on nobody for ever | ended with the parent. This is a behaviour change; T6 pins it (see 4.5). A sub-agent's question pauses its parent too, and the conversation's background tasks are in `kept`, so outside `kept` such a child is one an earlier attempt left behind |
| The reason a child's keeper is stopped with, when the parent's ending names none | "its parent was cancelled" (C1) or "its parent ended" (C2); only `background.stop` sees it | one text, `PARENT_ENDED` |
| `Agent.cancel` | C2 (here) then C1 | `end_children(..., status=CANCELLED, reason=reason)` once from the pipeline (kept `()`), then once from `Agent.cancel`. The second pass is idempotent and is kept: with `LocalRuns` the store cancel makes the attempt's own `finish` a conflict, and the pipeline then does not cascade |
| Live and async keepers of a cancelled run's children | stopped by C4 and C5 from the adapters' exits | stopped by `stop_kept` from the cascade. When the run was cancelled but **not ended here** (the store ended it first: `_lost` → `Agent.stop`), the pipeline still stops the keepers of `journal.handed_on`: `for child in handed_on: await children.stop_kept(...)`. That is the one case C4 and C5 covered that the store cascade does not |
| `background.abandoned` | cancels one run | `end_child(..., status=CANCELLED, reason=reason)`. Its children, if any, are now cancelled too |

### 4.4 Each call site's change

| Site | Change |
|---|---|
| `agent.py:Agent.cancel` L640 | `await children.end_children(self, run_id, tenant, status=RunStatus.CANCELLED, reason=reason)` |
| `agent.py:Agent._continue` L950-955 | the same, with `reason_of(resolution)` |
| `background.py:abandoned` L878-881 | `await children.end_child(parent, run_id, asked.tenant_id, status=RunStatus.CANCELLED, reason=reason)` |
| `pipeline.py:_cancelled` L852-857 and `_concluded` L1005-1007 | both call one helper, `_ended_with_children(agent, runtime, journal, status, *, ended, cut_off=False, error=None)`; `_cancelled` passes `cut_off=True`. When `ended`, it calls `end_children(... kept=_children_kept(runtime, journal, cut_off=cut_off))`, with errors logged as `_open_children_ended` logs them today. When not `ended` and `runtime.cancelled` is set, it calls `stop_kept` for each handed-on child. Neither caller gains a branch (`_cancelled` is cc 14, `_concluded` cc 13) |
| `pipeline.py:_children_ended`, `_open_children_ended`; `subagents.py:cancel_children` | deleted |
| `bridges/live.py:Keeper._detach` L585-589 | the loop is deleted; `_detach` keeps its other work |
| `bridges/async_tasks.py:AsyncTasks.cancelled` L300-307; `adapters/langgraph.py` L278, L319 | deleted (`child_cancelled` → `AsyncTasks.stopped` stays) |
| `adapters/claude.py`, `adapters/acp.py` | `async def child_cancelled(self, run_id: str, reason: str \| None) -> None: self.live.cancel(run_id, reason or children.PARENT_ENDED)` |
| docstrings naming the old functions | `bridges/helpers.py` L481, `bridges/async_tasks.py` L28, `subagents.py` L28 and L249, `background.py` L16-17, `pipeline.py` L18, L844, L1016, L1633, `adapters/langgraph.py` L200 |

### 4.5 Tests that pin it today, and tests to add

Pinned on main:

- C1: `integration/test_subagents.py::test_cancelling_a_parent_cancels_its_children`, `::test_cancelling_a_parent_cancels_its_nested_run`; `integration/test_react_background.py::test_cancelling_a_paused_parent_with_no_framework_work_cancels_its_child`, `::test_cancelling_the_turn_cancels_its_tasks`; `integration/test_react_async_subagent.py::test_g_a_turn_cancelled_at_its_approval_cancels_its_task_too` (the `child_cancelled` hook).
- C2: `integration/test_open_children.py::test_a_sub_agents_run_ends_out_of_time_with_the_run_whose_call_it_works_in`, `::test_a_foreground_helper_ends_out_of_time_with_its_run`, `::test_a_sub_agents_run_ends_with_the_run_whose_model_call_stalled`, `::test_a_foreground_helper_and_a_sub_agents_run_are_cancelled_with_their_run`, `::test_a_worker_stopping_its_attempt_ends_its_open_sub_agent_run_as_cancelled`, `::test_a_background_subagent_outlives_a_later_turn_that_runs_out_of_time`, `::test_a_background_subagent_outlives_the_turn_that_started_it_running_out_of_time` (the `kept` rule). Arriving with PR #273: `integration/test_run_limits.py::test_a_run_whose_open_children_cannot_be_listed_keeps_its_result`.
- C3: `integration/test_react_background.py::test_cancel_as_the_answer_to_the_tasks_question_cancels_it`.
- A cancel ends only the cancelled run's children, not the background work an earlier turn of the conversation started: `integration/test_claude_background_session.py::test_cancelling_a_turn_ends_it_at_its_next_call_and_leaves_the_background_work`.
- C4: `integration/test_claude_background_session.py::test_cancelling_the_turn_that_started_a_background_subagent_stops_it_too`, `::test_a_turn_cancelled_before_its_background_subagents_run_began_stops_it_too`; `integration/test_acp_background.py::test_a_cancelled_turn_cancels_its_background_sub_agents_run`.
- C5: `integration/test_react_async_subagent.py::test_e_cancelling_the_turn_cancels_the_task_it_started`, `::test_i_a_tasks_run_cancelled_elsewhere_cancels_the_task_on_its_server`.

To add **before** the change (characterisation tests; they must pass on main):

- T5 `tests/integration/test_kept_stops.py` (a new file, so no in-flight branch conflicts with it):
  - (a) A turn interrupted from another process (`Harness._renewed` → `Agent.stop(INTERRUPTED)`, which is not `Agent.cancel`) stops its live background subagent and its Deep Agents async task here. Today C4 and C5 do this.
  - (b) A turn cancelled on another server, whose hold here is then refused (`_lost`), stops them too. The store ended the turn first, so the pipeline does not cascade.
  - (c) A turn whose stream its caller closed (`_cancelled` with no `runtime.cancelled`) leaves its live background subagent working and ends the harness background task it started.
  - (a) proves the `[]` case of `_children_kept`, (b) the "not ended here" branch of `_ended_with_children`, and (c) the `cut_off` case.

To add with the change:

- T6 `tests/integration/test_open_children.py`: a run that ends `TIMEOUT` with a child `PAUSED` for a person ends that child with it (the 4.3 change).
- T7 `tests/unit/test_children.py`: `end_children` skips `kept` and children that are final already; writes `finish(status, error)` with an error and `cancel(reason)` without one; recurses into grandchildren with nothing kept; works for an adapter with no `child_cancelled` (the function and OpenAI adapters have none). `LiveSessions.cancel` for a run it attends and for one it does not.

## 5. Item 4: duplicate-code and complexity checks in CI

### 5.1 The checks

These are two unit tests in the style of `agent-memory-service/tests/unit/test_complexity_budget.py`.
They use the standard library only: no new dependency, nothing to install, and they run
offline. They live under `tests/unit/`, so CI's existing `make test` step runs them on every
push with no workflow change. Together they take about 2 s.

| Test | Budget at step 1 (re-measure on the newer main) | After the refactor |
|---|---|---|
| `tests/unit/test_complexity_budget.py`, `SRC = src/trellis`, counted as the memory service counts | `MAX_COMPLEXITY = 34`, `MAX_FUNCTION_LINES = 198`, `MAX_OVER_15 = 62`, `MAX_MEAN_COMPLEXITY = 4.12` | unchanged; the mean may only fall |
| `tests/unit/test_duplication_budget.py`, `SRC = src/trellis`, `MIN_LINES = 4`: a block is more than 4 lines in a row, stripped, with comments, docstrings, imports and def signatures left out, seen twice (pylint's `duplicate-code` rules, within a file too) | `MAX_DUPLICATE_BLOCKS = 39` | `33` once item 3 has landed (step 5) |

On failure each test prints the worst or newest offenders with `file:line` and the budget, as
the memory service's does. Lowering a number is the point of the ratchet. Raising one needs a
reason in the commit message. Both module docstrings say so.

### 5.2 Why these tools and this threshold

- **Duplication:** pylint's `duplicate-code` is what the brief names. On `0047ecfa`, with its
  default ignores, it finds 11 blocks at `min-similarity-lines=4`, 3 at 5 and 1 at 6. Only at
  4 does it catch the item 3 copies; at 6 it sees nothing but the legacy `skills.py` export.
  It also compares only *between* files, so it misses `langgraph.py:273 ↔ :314`, the same
  module's two copies. And it would add a dev dependency and about 20 s per run. The stdlib
  test applies the same line rules within files as well, finds 39 at 4, includes all six item
  3 blocks, and runs in about 1 s. `recheck.py` holds the reference implementation.
- **Exact-AST clones** (names and literals normalised, statement runs inside functions) were
  tried and rejected. They find only 15 groups at 6 statements and none of the targeted copies:
  those copies differ in their middle and match only in their tails.
- **Complexity:** the memory service's AST count makes the two repositories comparable and
  needs nothing installed. ruff `C90` is already installed, but mccabe counts nested functions
  apart and finds only 2 functions above 15. radon is not installed. Neither is added.

Cross-checks, on demand only (nothing to install in CI):

```bash
uv pip install --target /tmp/pl pylint==3.3.9 radon==6.0.1
PYTHONPATH=/tmp/pl python -m pylint --rcfile=/dev/null --disable=all --enable=duplicate-code \
    --min-similarity-lines=4 --recursive=y src/trellis      # 11 on 0047ecfa
PYTHONPATH=/tmp/pl python -m radon cc -s -n C src/trellis/harness   # 64 above 15
rm -rf /tmp/pl
```

## 6. Risk notes: the in-flight branches against these functions

Measured with `recheck.py overlap` (`git diff -U0 origin/main...<ref>`, mapped onto the base's
functions). The last three branches were not on `origin` when this was measured. They were
measured from their checkouts under `/var/tmp/cons/<name>/agent-harness`, uncommitted changes
included.

| Branch | Head measured | Functions it touches / shared with this refactor | What is shared | If it is still open at step 1 |
|---|---|---|---|---|
| PR #273 `-open-bugs` | `2bf808c0` | 39 / 1 shared + 3 adjacent | `background.py:follow_up` (L519, 7 lines above the B6 site at L528). Adjacent: `bridges/live.py:Keeper._attach` (directly above `_detach`), `LiveSessions.place`/`aclose` and a new `LiveSessions.lapse` (beside the new `LiveSessions.cancel`); `bridges/helpers.py:named`, `held`, `_naming`, `Helpers._started_as` (the module `helping` joins); `bridges/async_tasks.py:AsyncTasks._started`/`starting`. It adds the pinning tests named in 2.4 and 4.5, and edits `test_conversation_line.py`, `test_claude_background_session.py`, `test_react_background.py`, `test_react_async_subagent.py`, `test_claude_live.py` and `test_acp_background.py` | steps 8 (`follow_up`) and 9-11 (`bridges/live.py`, its tests) wait for it |
| PR #275 `-livestack` | `dc51ff44` (base `873e630e`) | 5 / 0 | `bridges/session.py`, `clients/memory.py` only | no wait |
| PR #276 `-decide15` | `68a8a648` | 12 / 2 | `pipeline.py:_left_the_line`: **the B2 decision itself**. Rebase step 7 onto it and keep its rule. `adapters/claude.py:ClaudeAdapter._helped`: only `aclosing` → `contextlib.aclosing` (H1) | steps 3 and 7 wait for it |
| `-scenfix` | `c36c0d6c` (contains #273) | 53 / 1 shared + adjacent, as #273 | also `pipeline.py:_attempt` (only the `_continued_session` call, L409; far from L472), `_continued_session`, `_served_elsewhere` | as #273 |
| `-decide2` | `a27f4c63` | 15 / 0 | `adapters/langgraph.py:_continued` (a module function, not H5/H6), `middleware.py`, `automatic.py`, `react.py` | no wait |
| `-onerule` | checkout at `0047ecfa` + uncommitted | 13 / 0 | `adapters/claude.py:_decide`, `_asked.can_use_tool`, `tools/bridge.py`, `tools/mcp.py`, `governance/catalog.py` | no wait |
| `-livedirect` | checkout at `151d2a3a` + uncommitted | 6 / 0 | `bridges/session.py`, `clients/memory.py`, `middleware.py:_answer_now`; the rest is `tests/live` | no wait |
| `-sdkfull` | checkout at `ae24de22` (moved during measuring) | 7 / 0 | `adapters/claude.py:ClaudeAdapter.stream`, `_declared`; `pipeline.py:_pause`; `governance/catalog.py` | no wait |

The risk, then, is low. The refactor shares one decision (B2) with PR #276 and one site (B6 in
`follow_up`) with PR #273 and `-scenfix`. Everything else is adjacent at most. Ordering the
refactor after all eight leaves only those two, and the rebase keeps their versions. The new
files (`stopping.py`, `children.py`, `tests/unit/test_stopping.py`,
`tests/unit/test_children.py`, `tests/integration/test_kept_stops.py`, the two budget tests)
conflict with nothing.

Re-run `recheck.py overlap` for any branch that is still open on the day. A non-zero "shared"
count other than these two means that branch must merge first, or the step touching it waits.

## 7. Step order

Each step is one commit, and each commit keeps `make test` green at 100% line and branch
coverage of `trellis.harness`. That means every line a commit adds is reached by a test in the
same commit, and every line it deletes takes its now-dead branches with it. Before each
commit, run the gates from `GATES_NOTE`:

- `ruff check .` and `ruff format --check .`;
- `pyright` over the whole repository with the `typings/trellis` links;
- `python scripts/check_docs.py`;
- one pytest at a time on every test file that calls a changed function. Grep callers in src
  **and** tests, because several tests call these functions directly: `h._lost`,
  `background.stop`, `Helpers.closed`.

| Step | Commit | Item | Tests in it |
|---|---|---|---|
| 0 | (no commit) Re-check on the newer main: `recheck.py numbers / copies / overlap`; update the line numbers in this plan if they moved | - | - |
| 1 | Tests: complexity and duplication budgets at today's numbers | 4 | the two budget tests |
| 2 | Tests: pin how a cancelled turn's kept work stops, here and from another server | 2 | T5 (passes on main) |
| 3 | Helpers: one `helping()` opens and closes an attempt's helpers; Claude and ACP through it | 3 | T4 |
| 4 | OpenAI Agents through `helping()`: `Helpers.left_open` replaces `_closed` | 3 | existing H3/H4 tests |
| 5 | LangGraph through `helping()`; duplication budget 39 → 33 | 3, 4 | existing H5/H6 tests |
| 6 | Stopping: one owner (`stopping.py`); `Agent.stop` and `Harness._lost` use it; constants move | 1 | T1 |
| 7 | `Kept.cancel` and `Tasks.stop` stop through `Agent.stop`; `_left_the_line` reads `ended_elsewhere` (on top of #276) | 1 | T2, T3 |
| 8 | `unless_ended` at the 11 "ended meanwhile" sites and in `_settle_cancelled`; one `INTERRUPTED` | 1 | T1 already covers both branches |
| 9 | Children: one cascade (`children.py`); `cancel_children` and `background.abandoned` through it, under `cancel_children`'s own rules | 2 | T7 (part) |
| 10 | The pipeline's endings through `end_children` (`_children_kept`, every non-final child); `_children_ended` deleted | 2 | T6, T7 |
| 11 | Keepers through `stop_kept`: `child_cancelled` on Claude and ACP (`LiveSessions.cancel`); handed-on keepers stopped when not ended here; `Keeper._detach`'s loop, `AsyncTasks.cancelled` and the two LangGraph calls deleted | 2 | T5 stays green; T7 (rest) |
| 12 | Budgets lowered to what is measured then (mean, blocks); docstrings naming deleted functions fixed | 4 | the two budget tests |

Why this order:

- The budgets come first, so the refactor is measured against them.
- Item 3 comes next: it is mechanical and adapter-local.
- Item 1 comes before item 2, because the cascade uses `unless_ended`.
- The characterisation test T5 lands before the step that could break it.

**If T5 fails at step 11** (a keeper is only stopped from the adapter's exit, at a timing the
cascade does not reach), keep C4 and C5, but have each call `children.stop_kept` instead of
its own body, and report it. The cascade's ownership of the store writes stands either way.

Steps 6-8 can be dropped together without affecting items 2-4, apart from `end_child` using
`unless_ended`: if they are dropped, inline the suppress there. Step 8 alone can be dropped:
it is the most mechanical step and the one most exposed to in-flight edits (`follow_up`).
