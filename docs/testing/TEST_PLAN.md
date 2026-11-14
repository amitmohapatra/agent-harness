# Trellis stack: master test plan

Scope: agent-harness (all adapters, custom ReAct), agent-memory-service, agent-runs,
agent-contracts. Branch under test: `claude/cool-maxwell-w5avo8-final` plus the open feature
branches (bg-kimi, react-background, bg-claude-da, openai-upgrade, chat-order).

Goal: every feature of every library works as the library intends **through** the harness,
every harness / memory / runs feature serves its real purpose on every adapter, and every
known limitation is written down instead of discovered by a user.

---

## 1. Quality goals and invariants

These are checked automatically **after every test** (a shared fixture), not only in the tests
that target them. A test that passes its own assertion but breaks an invariant fails.

| ID | Invariant |
|---|---|
| INV-1 | No run is left `RUNNING`/`PAUSED` without a live holder, lease or person to answer (sweep the run store at teardown). |
| INV-2 | Every user turn is recorded in memory exactly once; every tool call that ran has exactly one tool record. |
| INV-3 | One question per call: count of pauses = count of distinct calls that needed a person. |
| INV-4 | A call approved runs once; rejected runs zero times; edited runs once with the edited args. |
| INV-5 | No data crosses a user or tenant boundary (memory, history, files, approvals, runs, traces). |
| INV-6 | Cost and token totals of a run = sum of its model calls (child runs included in the parent's budget view, never double counted). |
| INV-7 | Every request the harness sends to memory/runs validates against agent-contracts / agent-runs OpenAPI. |
| INV-8 | No secret (keys, headers, tokens) appears in logs, traces, events or memory. |
| INV-9 | No orphan OS processes (CLI, sandbox, kimi) and no temp files outside the test folder at teardown. |
| INV-10 | Events on the stream are well-formed and ordered (start → … → finished/error), one start per logical tool call. |

## 2. Risk-based priority

| Priority | Meaning | Examples |
|---|---|---|
| P0 | Data loss, data leak, double side effect, security | INV-4/5/8, a write run twice, user A sees user B's memory, a rejected call runs |
| P1 | Wrong answer or stuck system | run stuck RUNNING, lost message, lost background result, wrong cost |
| P2 | Degraded behaviour | extra question, slow recovery, unclear error text |
| P3 | Cosmetic / docs | wording, docs drift |

## 3. Test design techniques used

* **Equivalence partitioning + boundary values** for every documented limit (sizes, counts,
  timeouts, budgets): at, one below, one above (done for harness by mutation round: 1,235 mutants).
* **State-transition testing** for runs (QUEUED → RUNNING → PAUSED → RUNNING → SUCCESS/ERROR/
  CANCELLED/PARTIAL/TIMEOUT), conversations (idle / busy / paused) and background children.
  Every valid transition and every invalid one (must be refused).
* **Pairwise (all-pairs) combination** for adapter × feature × surface × store
  (LocalRuns / agent-runs) × memory (on / off / down) — full cartesian is the matrix
  (12,688 cells); pairwise is used where the cartesian explodes (failure injection × phase).
* **Concurrency testing**: deterministic interleavings with gates (never sleeps), plus
  randomized stress runs with seeds recorded so any failure replays.
* **Fault injection / chaos**: SIGKILL at every phase, service down/slow/flaky, clock skew,
  disk full, partial writes.
* **Mutation testing**: tests must fail when the code is broken (done for harness; to extend to
  runs and memory).
* **Contract testing**: consumer (harness) vs provider (memory, runs) schemas, both directions.
* **Differential testing**: same scenario on every adapter must produce the same harness-level
  outcome (statuses, records, questions, costs).
* **Soak / scale**: long runs and many users to find leaks and drift.
* **Real-model acceptance** (needs Haiku credit): behaviour that a scripted model cannot prove
  (model actually chooses background, follows steering, recalls memory).

## 4. Environments and oracles

| Env | What it is | Used for |
|---|---|---|
| E1 unit | in-process, fakes | logic, boundaries |
| E2 integration | in-process harness, LocalRuns, fake memory, scripted models, fake Claude CLI, real Kimi binary with scripted model | adapter behaviour, matrix |
| E3 private stack | real Postgres, real memory API, real agent-runs API + ticker, 2–3 harness server processes, worker processes, real AG-UI/A2A HTTP | scenarios, multi-process, chaos |
| E4 real CLI probe | bundled Claude CLI / kimi against a scripted Messages API on localhost | library truth (what the CLI really emits) |
| E5 real model | Haiku only, private stack | acceptance, accuracy (blocked on credit) |
| E6 infra | Docker, Qdrant, OpenFGA, GCS emulator | memory/runs tests skipped today |

Oracles: run store rows, memory DB rows, event streams, journal, process table, file system,
contract validators, scripted-model call logs (what the model was actually sent).

---

## 5. Test catalogue

Each area lists the cases; every case runs on **every adapter where it applies** (A = all ten:
function, ReAct, ReAct+DeepAgents, create_agent, create_agent+middleware, hand-built graph,
Deep Agents, OpenAI Agents, Claude, Kimi). "Owner" = the job/branch executing it.

### 5.1 Conversation concurrency (CONV) — owner: chat-order + scenarios

| ID | Case | Pri |
|---|---|---|
| CONV-01 | Two messages same chat, same server, default queue: answered in order, second sees first in history | P1 |
| CONV-02 | Burst of 5 / 20 messages same chat: each answered once, in arrival order, no lost, no duplicate | P1 |
| CONV-03 | Same chat, messages hit **different servers** (load balancer): still one at a time, in order | P0 |
| CONV-04 | `interrupt` same server / across servers: working turn cancelled, its question kept in history, side effects of the cancelled turn not repeated | P1 |
| CONV-05 | `steer` on steering adapters: message reaches the turn, one answer covers both; non-steering adapters fall back to queue (said once) | P1 |
| CONV-06 | Steer arrives after the turn's last step: runs as its own turn, not lost | P1 |
| CONV-07 | One user, 3 chats in parallel: run concurrently, histories separate, user-level memory shared | P1 |
| CONV-08 | Two users, same thread id: never wait for each other, fully isolated | P0 |
| CONV-09 | A turn paused for approval + new message in same chat: behaviour defined and consistent across servers | P1 |
| CONV-10 | Holder of the turn SIGKILLed: chat freed after hold lapse, next message runs, killed run ends `holder_lost` | P1 |
| CONV-11 | Same message id submitted twice (client retry): one run, one answer (idempotency) | P0 |
| CONV-12 | `agent.start` (queued) and direct `run` mixed in one chat: one at a time | P1 |
| CONV-13 | Cancel a queued message before it starts; cancel while waiting for the turn | P2 |
| CONV-14 | 100 chats × 5 messages concurrently on 3 servers (stress, seeded) | P1 |

### 5.2 Background work (BG) — owner: bg-kimi, react-background, bg-claude-da, openai wiring

| ID | Case | Pri |
|---|---|---|
| BG-01 | Start background task → turn replies immediately → user chats → result delivered (next turn and idle push) | P1 |
| BG-02 | Background child run: RUNNING while working, SUCCESS with result, ERROR when cut off, never SUCCESS at launch | P1 |
| BG-03 | Approval inside a background task reaches the person in the parent chat; one question | P0 |
| BG-04 | Cancel the turn → its background tasks cancelled; cancel the task alone → turn unaffected | P1 |
| BG-05 | Server killed while background task runs → task ends `holder_lost` or resumes; conversation told | P1 |
| BG-06 | Two background tasks at once; ten at once (admission limit respected) | P1 |
| BG-07 | Background task outlives many turns and a restart; result still delivered once | P1 |
| BG-08 | Budget / step limits apply to background work; cost lands on the right run | P1 |
| BG-09 | check / list / update / cancel tools (Deep Agents shape) on harness-native and remote tasks | P1 |
| BG-10 | Claude background subagent + background Bash; Kimi background Agent + Bash; Claude Workflow (one run per workflow or refused) | P1 |
| BG-11 | Resume on another worker while a background task runs on the first | P1 |
| BG-12 | Background result arrives while the chat is busy with another turn: queued, not interleaved mid-answer | P1 |

### 5.3 Workflows (WF) — owner: scenarios

| ID | Case | Pri |
|---|---|---|
| WF-01 | LangGraph multi-step with branches, Send fan-out, subgraphs: every node's calls governed and recorded | P1 |
| WF-02 | Approval mid-graph, resume on another worker; checkpointed and non-checkpointed graphs | P0 |
| WF-03 | Crash mid-workflow: resume from journal, no completed step re-run (writes especially) | P0 |
| WF-04 | Cancel mid-workflow at every node | P1 |
| WF-05 | Deep Agents planning/todos + `task` sub-agents (parallel), each a child run | P1 |
| WF-06 | OpenAI handoff chain and agents-as-tools pipeline; handoff cycles | P1 |
| WF-07 | Library retry policies (RetryPolicy, ToolRetryMiddleware) inside workflows: exact call counts | P1 |
| WF-08 | Claude Workflow tool: per decision on bg-claude-da | P1 |

### 5.4 Long-running tasks (LR) — owner: scenarios

| ID | Case | Pri |
|---|---|---|
| LR-01 | Turn working many minutes: hold renewed, never lapses while alive | P1 |
| LR-02 | 50+ / 200 step task: step limit, stall guard, context fitting, compaction | P1 |
| LR-03 | Run timeout / deadline / budget hit mid-task: ends with the right status, children ended | P1 |
| LR-04 | Worker lease extended by progress; worker stopped gracefully hands runs back | P1 |
| LR-05 | Restart mid long task: resumes or ends clearly (never stuck) | P1 |
| LR-06 | Long chat 50+ turns: early facts still recalled; costs exact across turns | P1 |

### 5.5 Approvals and governance (APR) — owner: done (test_one_question, 583 cases) + scenarios

| ID | Case | Pri |
|---|---|---|
| APR-01 | Sources: undeclared tool, catalog rule, write, hook ask, library gate (interrupt_on, needs_approval, can_use_tool, ACP permission) × approve/reject/edit/stream/resume | P0 |
| APR-02 | Same call twice; reject then approve (asked twice); per-call rejection | P0 |
| APR-03 | Always allow / allow for conversation / learned rules start from asking | P1 |
| APR-04 | Hook's own question is not an approval | P0 |
| APR-05 | Approval across workers, after restart, after the catalog changes, catalog down > 300 s | P1 |
| APR-06 | Approval of a background or sub-agent call routed to the right run | P0 |
| APR-07 | Nested args kept structured in feedback | P2 |

### 5.6 Memory (MEM) — owner: features (+ Haiku accuracy later)

| ID | Case | Pri |
|---|---|---|
| MEM-01 | Turn recorded once; tool records once; large outputs bounded | P1 |
| MEM-02 | Fact told in chat 1 recalled in chat 2 of the same user | P1 |
| MEM-03 | Never recalled for another user / tenant (DB-level check) | P0 |
| MEM-04 | Extraction → facts and beliefs; consolidation; no duplicates (same-speaker and both-ways) | P1 |
| MEM-05 | Forget/delete: gone from recall **and** from storage (incl. vectors, archive) | P0 |
| MEM-06 | Retention expiry | P2 |
| MEM-07 | Profile / about_user up to 16 lines, budget rules (8,000 tokens; quarter of small windows, min 2,000) | P2 |
| MEM-08 | Feedback: decision and outcome, once per decision, on the right run | P1 |
| MEM-09 | Approval-rule learning end to end | P1 |
| MEM-10 | Memory down mid-chat: turn answers with documented fallback; writes not lost when it returns | P1 |
| MEM-11 | Admin role checks on every admin route; every route refuses missing key | P0 |
| MEM-12 | Accuracy: LoCoMo / LongMemEval with Haiku (blocked on credit) | P1 |
| MEM-13 | Concurrency: two chats writing memory for one user at once: no lost update, no duplicate | P1 |

### 5.7 Runs (RUN) — owner: features

| ID | Case | Pri |
|---|---|---|
| RUN-01 | Every status transition valid; every invalid one refused | P1 |
| RUN-02 | Priority order; concurrency_key one at a time | P1 |
| RUN-03 | Claims, leases, heartbeats, holds, `end_abandoned`, resume grace | P1 |
| RUN-04 | Retries and breaker | P2 |
| RUN-05 | Webhooks run.paused / run.finished, signed, retried, no loopback/SSRF | P0 |
| RUN-06 | Events log and SSE across replicas; reconnect with last event id | P1 |
| RUN-07 | Inbox paging | P2 |
| RUN-08 | Idempotency keys | P0 |
| RUN-09 | Rate limits | P2 |
| RUN-10 | Cancel at each status; child runs and parent link; cancel cascades | P1 |
| RUN-11 | Retention | P2 |
| RUN-12 | Every OpenAPI endpoint exercised by a purpose-level test | P1 |
| RUN-13 | Migrations up/down/up with data; upgrade with runs in flight | P0 |

### 5.8 Schedules (SCH) — owner: features + scenarios

Cron, once, condition; missed fires; overlap; jitter; expiry; DST forward/back; schedule made
from chat delivered to the right thread (open/closed chat); stale hour in live sessions;
schedule fires while the user is mid-turn; schedule of a deleted user.

### 5.9 Files, sandbox, skills, query (FS) — owner: features

Upload/download (signed links 5 min), names ≤120, per-user sandbox, reaper (double delete
fixed), 512 processes, read limits 256 KiB, query_result limits (200 rows, 20,000 chars, SQL
5,000), wide tables (1,000 columns), skills run, all under concurrency.

### 5.10 Surfaces (SURF) — owner: features

AG-UI (interrupts, replay latest 20 runs, buffers 2,048 events / 256 runs, problem codes),
A2A (card limits, push URL limits, identity header 4,096, task states), worker CLI, doctor.

### 5.11 Reliability and chaos (REL) — owner: scenarios

SIGKILL at each phase (queued, running, tool call in flight, paused, resuming, finishing);
memory down/slow/500s; runs down/slow; model 429/500/timeout/overflow; disk full; clock skew;
network partition between harness and runs; duplicate delivery of webhooks.

### 5.12 Security (SEC) — owner: **phase 2 job (not yet started)**

| ID | Case | Pri |
|---|---|---|
| SEC-01 | Tenant isolation fuzz: every endpoint with another tenant's ids | P0 |
| SEC-02 | Prompt injection via tool output / memory / file content tries to approve its own call, exfiltrate secrets, or call disallowed tools | P0 |
| SEC-03 | Secret redaction in logs, traces, events, memory (INV-8) | P0 |
| SEC-04 | SSRF: webhooks, A2A push, MCP URLs, Deep Agents `webhook`/`url` | P0 |
| SEC-05 | Sandbox escape attempts (path traversal, symlinks, reads outside run folder) | P0 |
| SEC-06 | AuthN/Z on every route; key rotation; expired keys | P0 |
| SEC-07 | Shell classifier bypass attempts (read-only claims) | P0 |

### 5.13 Performance, soak, scale (PERF) — owner: **phase 2 job (not yet started)**

SSE p95 latency; 10k users memory load; 1,000 concurrent chats; 24 h soak (memory/FD/process
leaks, journal growth); backpressure when memory/runs are slow; DuckDB memory under load.

### 5.14 Compatibility and upgrade (COMPAT) — owner: **phase 2 job**

Python 3.12 and 3.13; library version ranges (min and max tested: langgraph, langchain,
deepagents, openai-agents 0.24, claude-agent-sdk, kimi-code, ACP); runs paused on old library
versions resume after upgrade (OpenAI run-state schema 1.17 → 1.21 checked); contracts 0.7.0
vs 0.7.1 mixes.

### 5.15 Contracts (CON) — owner: done + features

Consumer/provider schema checks both ways; OpenAPI regenerated equals committed; every field
limit (contracts at 100% coverage).

### 5.16 Real-model acceptance (ACC) — owner: Haiku phase (blocked on credit)

ReAct 357-turn accuracy; memory LoCoMo/LongMemEval; model actually uses background tools for
long work; steering understood; Claude background subagents with the real CLI; Deep Agents
real; cost/usage accuracy vs provider bills.

---

## 6. Known framework limitations (current, honest list)

| # | Limitation | Where | Status |
|---|---|---|---|
| L1 | Same chat on two servers can run two turns at once | harness pipeline | being fixed (chat-order) |
| L2 | `steer` not available on function, Kimi, plain graphs; Claude only after a tool call | library hooks | by design; falls back to queue |
| L3 | OpenAI Agents SDK 0.24 has no background agents; Responses background not polled | library | harness mechanism being built |
| L4 | Deep Agents async tasks need a user-hosted Agent Protocol server | library | harness-native tasks being built (react-background) |
| L5 | Kimi background work lost at process exit | harness acp | being fixed (bg-kimi) |
| L6 | Claude Workflows: agents not attributable per agent | Claude CLI | one run per workflow being tried, else disabled |
| L7 | Claude SendMessage / teams disabled (resumes agents unattributably, reaches other sessions) | Claude CLI | disabled by design |
| L8 | Two identical concurrent calls from two Claude background subagents can be credited to the wrong run | SDK hides tool-use id from sdk tools | open; needs own MCP server |
| L9 | After a crash the real Claude CLI finishes a background subagent's model calls (~30 s), runs no tools | Claude CLI | documented |
| L10 | OpenAI 0.23+ hides tool error details from the model by default | library | kept as library default; user decision pending |
| L11 | `after_tool` hooks see every harness retry try | harness design | documented |
| L12 | `max_tool_calls` counts harness tool calls only, not library built-ins | harness design | documented |
| L13 | Kimi: a member's calls that Kimi does not ask about are invisible over ACP | ACP protocol | documented |
| L14 | Live Claude session keeps tool wording from its start (current hour now sent per turn) | Claude session | mitigated |
| L15 | A paused background child whose question no turn has asked stays PAUSED until answered | harness | documented |
| L16 | When the store alone ends a child, the chat hears only via `run.finished` webhook | harness | documented |
| L17 | All Claude background / parallel coverage is against the fake CLI; real CLI only probed | test env | needs Haiku |
| L18 | No real-model validation of this round | test env | needs Haiku credit (~$40) |
| L19 | Memory accuracy ceiling ~0.87 LoCoMo, limited by the reader model | memory | parked |
| L20 | Tests needing Docker, Qdrant, OpenFGA, GCS, model weights not run in this environment | test env | needs E6 |
| L21 | No scale / soak / SSE latency evidence yet | test env | phase 2 |
| L22 | GitHub CI does not run on branch pushes; results are local | CI config | runs on PR |

## 7. Execution plan

| Phase | Jobs | Status |
|---|---|---|
| 1a | bg-kimi, react-background (+shared mechanism, DA gaps), bg-claude-da (Workflow), chat-order | running |
| 1b | scenarios (CONV, WF, LR, REL, multi-user), features (MEM, RUN, SCH, FS, SURF, harness inventory) | running |
| 1c | OpenAI wiring to shared background mechanism | after react-background |
| 2 | SEC, PERF, COMPAT jobs | after phase 1 lands (machine capacity) |
| 3 | Fold all into final; full suite + matrix + scenarios + features + live stack; INV-1..10 on every test | after 1–2 |
| 4 | ACC with Haiku | when credit is added |

## 8. Exit criteria

* Every P0 and P1 case passes on every applicable adapter; skips explained.
* INV-1..10 hold on every test of every suite.
* No test weakened; every failure fixed with a failing test first or listed as a limitation
  with the user's sign-off.
* Matrix, suite, scenarios, features, live: 0 unexplained differences vs base.
* Limitations table reviewed and accepted by the user.

---

## 9. Scenario suite: execution status and findings (job: scenarios)

Branch `claude/cool-maxwell-w5avo8-scenarios` (from `claude/cool-maxwell-w5avo8-final` d2bf754).
Suite: `tests/scenarios/` (marker `scenario`, `make test-scenarios`); invariants:
`tests/support/invariants.py` (INV-1..INV-10, an autouse fixture: every scenario test is checked
at teardown). Environment E3: a private stack (`scripts/scenarios/stack.sh`): real PostgreSQL,
the real memory API with its job worker, the real agent-runs API and ticker, harness server and
worker processes of the test's own (`tests/scenarios/serve.py`), real AG-UI/A2A HTTP, every model
scripted (`tests/scenarios/brains.py`: a "brain" that answers from the request it is sent), the
real `kimi` binary, the scripted Claude CLI. No paid model call.

### How to run

```bash
SCEN_PG_PORT=<cluster port> SCEN_HOME=/var/tmp/scen scripts/scenarios/stack.sh reset   # databases, keys, services
scripts/scenarios/stack.sh test                       # every scenario (or: make test-scenarios)
scripts/scenarios/stack.sh test -k "s04 and react"    # a part
python scripts/scenarios/plan_report.py <junit.xml>   # results per plan id
scripts/scenarios/stack.sh drop                       # processes, databases, folders gone
```

Without `MEMORY_URL`/`RUNS_URL`/`TRELLIS_API_KEY` every scenario is skipped, never failed.

### Findings

| ID | Pri | Plan | What | Root cause | Status |
|---|---|---|---|---|---|
| SCN-1 | P1 | CONV-02 | A burst to one chat is not answered in the order sent (every adapter, bursts of 2, 5, 20) | the turn is taken (`pipeline.py:248` `Turns.taken`) only after `Agent._opened` (`agent.py:1490`, the run store's start) and the hold (`pipeline.py:203`): awaits that end in any order | **fixed** by chat-order (in to-main); regression test `tests/integration/test_scenario_findings.py::test_scn1…` and the burst scenarios pass |
| SCN-2 | P2 | WF-02/03, INV-10 | A checkpointed graph resumed on another worker gives its new calls the ids of calls before the pause | `tools/bridge.py:359` `ref = call_id or f"{run_id}:{step}"`, `step` from `Runtime.next_step()` (`runtime.py:480`) restarting at 0 each attempt (`runtime.py:296`) | **fixed** by scn-fixes (in to-main); `tests/scenarios/test_s15_findings.py::test_scn2…` passes |
| SCN-3 | P2 | WF-02, INV-10 | A graph run again from its input after a resume announces its replayed calls again (TOOL_CALL_START… a second time) | `tools/bridge.py:360-363`: a replayed outcome emits the call's events | **fixed** by scn-fixes (in to-main); `test_s15_findings.py::test_scn3…` passes |
| SCN-4 | P1 | CONV-01/02/07, MEM-01 | The next turn misses the previous turn when memory takes > 5 s to keep it (seen 21× under load: a chat's third turn without its second) | `agent.py:1296-1306` `recorded_run`: the transcript write waits `TURN_WRITE_SECONDS` (5 s, `agent.py:158`) then is queued and the run returns | **fixed** by scn-fixes (in to-main): `tests/integration/test_scenario_fixes.py` |
| SCN-5 | P3 | APR | docs/api.md says the run's user is "its approvals' default assignee"; approvals have no assignee (anyone's, `runtime.py:737`), so `h.inbox("user:<user>")` lists none | docs drift | reported |
| SCN-6 | P1 | MEM-10, INV-2 | Writes (transcripts, tool records, outcomes) are "failed and is lost" once the memory client's circuit opens (seen 27× in one file under 504s) | `writes.py:66/217`: 3 attempts, 0.5 s backoff (~3.5 s), no spool in a container (`settings.default_spool`) | **fixed** by scn-fixes (in to-main): `tests/integration/test_scenario_fixes.py` |
| SCN-7 | P2 | MEM-01, INV-7 | A message over 4,000 characters: memory context refused (422) and tool records lost | the whole message sent as `ContextRequest.query` / `HintsRequest.task` / `ToolInvocationIn.task` (`maxLength` 4000) | **fixed** (`clients.memory.fitted`), test `tests/integration/test_long_questions_to_memory.py` |
| SCN-8 | P1 | CONV-13 | Cancelling a message that waits its turn in process: the store says CANCELLED, but the run still runs when its turn comes and its finish fails (409) — `agent.run` raises `ConflictError` | `agent.py:601` cancels only runs in `self.running`; a run waiting in `Turns.taken` (`pipeline.py:248`) is not there, and nothing re-checks its record after it takes the turn | **fixed** in to-main: `test_scenario_findings.py::test_scn8…` and `test_s11_corners.py::test_cancel_a_message_waiting_its_turn…` pass |
| SCN-9 | P1 | CONV-11 | AG-UI: the same `runId` posted twice at once started the run twice | `agui/__init__.py` checked the id before its awaits and marked it after | **fixed**, test `tests/integration/test_agui_retries.py` |
| SCN-10 | P2 | SURF | AG-UI: a message with empty text and no `state` started a run that failed (`EmptyInputError`) instead of a 422 | `agui/__init__.py`: `latest_user_text() or state` fell back to `None` | **fixed**, same test file |
| SCN-11 | P2 | WF-07 | A LangGraph node's `RetryPolicy` never retries a failed harness read: the tool's failure comes back as its result text (`flaky failed: …`), the node does not raise, and the harness did not try the read again either (`ConnectionError`) — the read ran once | the bridge answers a failed call as an outcome (`tools/bridge.py`, the framework tool built by `tools/convert`), not an exception | failing scenario `test_s16_reliability_more.py::test_a_nodes_retry_policy…` (xfail strict); owner: langchain-fixes |
| C-1 | P3 | INV-7 | memory OpenAPI lists no 504 on any of its 73 operations; its deadline middleware answers 504 (seen on /v1/context and /v1/verify under load) | `memory_service/api/deadline.py:107` vs `api/openapi.py` `_document_responses` | reported (agent-memory-service); a known drift in the invariants |
| C-2 | P3 | INV-7 | the harness files a run's outcome with `source: "system"`; agent-contracts' `FeedbackSource` has no `system` (the memory service accepts it) | `agent.py:1332` vs `agent-contracts feedback.py:47` | reported; a known drift in the invariants |

Memory-service resident memory was sampled every 30 s for the whole run: 3.44-3.52 GB from
start to end (its models), no growth. It was OOM-killed several times by the shared cgroup
(other jobs' processes), never by its own growth.

### Status per plan id (scenarios job; written = a test exists in tests/scenarios, tagged with the id)

Run state: each file was run at least once against the private stack; after the merge of
`claude/cool-maxwell-w5avo8-to-main` only the tests the merge touched were run again (the whole
pass is the user's, once everything is merged).

| Plan id | State | Where (tests/scenarios/) | Note |
|---|---|---|---|
| CONV-01 | written, passing | `test_s01::test_messages_one_after_another`, `test_a_message_to_a_busy_chat[queue]` | 10 adapters |
| CONV-02 | written, passing after chat-order | `test_s01::test_a_burst_of_messages` (2, 5, 20) | re-run after the merge on function and ReAct only |
| CONV-03 | written, passing after chat-order | `test_s14::test_one_chat_on_two_servers_is_one_turn_at_a_time` | |
| CONV-04 | written, passing | `test_s01::test_a_message_to_a_busy_chat[interrupt]`; across servers `test_s16::test_an_interrupt_sent_to_another_server` | the cross-server case failed once in a combined run and passed 3 of 3 alone: watch it at the fold |
| CONV-05 | written, passing | `test_s01::test_a_message_to_a_busy_chat[steer]` | function, Kimi, plain graphs fall back to queue |
| CONV-06 | written | `test_s14::test_a_steer_after_the_last_step_is_its_own_turn` | passed on the pre-merge run; uses a timed stall (no gate on a model's last step) |
| CONV-07 | written | `test_s02_s03::test_three_chats_at_once_stay_apart`, `test_recall_across_chats_is_user_level` | failed before the merge on SCN-4 (fixed since); not re-run |
| CONV-08 | written, passing | `test_s02_s03::test_two_users_on_one_thread_id`, `…approvals_files_schedules_and_runs_stay_apart` | |
| CONV-09 | written, passing | `test_s14::test_a_new_message_while_the_chat_waits_for_an_approval`, `test_s04::test_a_pause_while_another_chat_works` | |
| CONV-10 | written, passing | `test_s06::test_a_server_killed_mid_turn`, `test_s16` (killed while paused; killed while resuming) | |
| CONV-11 | written, passing (SCN-9 fixed here) | `test_s11::test_the_same_message_submitted_twice_runs_once`, `tests/integration/test_agui_retries.py` | |
| CONV-12 | written, passing after chat-order | `test_s14::test_a_queued_turn_and_a_direct_one_in_one_chat` | |
| CONV-13 | written, passing after the merge | `test_s11::test_cancel_a_queued_run`, `test_cancel_a_message_waiting_its_turn_and_a_running_one`, `test_cancel_a_paused_run_and_a_resuming_one` | |
| CONV-14 | written | `test_s14::test_stress_many_chats_on_three_servers` (seeded, `SCEN_STRESS_CHATS`) | failed pre-merge under memory-service load (SCN-4/6, fixed since); not re-run: needs a quiet machine |
| BG-01, BG-02, BG-10 | written, passing (Claude subagent; Deep Agents async tasks on ReAct and create_deep_agent) | `test_s05` | Kimi background shell: xfail(strict) "bg-kimi"; ReAct's own sub-agents in the background: xfail(strict) "react-background"; Claude Workflow: xfail(strict) "bg-claude-da" |
| BG-03 | written, passing | `test_s17::test_an_approval_inside_a_background_task_reaches_the_parent_chat` | Claude |
| BG-04, BG-06 | written, passing | `test_s17::test_two_background_tasks_one_cancelled_alone` | two at once; "ten at once" not written (admission limit is the features job's RUN/X07) |
| BG-05 | written, passing | `test_s13::test_a_background_subagent_when_its_server_is_killed` | |
| BG-07 | written, xfail | `test_s05::test_kimi_background_shell_result_reaches_a_later_turn` | "bg-kimi"; restart half not written |
| BG-08 | not written | | covered in process by `tests/integration/test_claude_background_session.py` (budget / time stop a background run); no stack-level case |
| BG-09 | written, passing | `test_s05::test_deep_agents_async_task_while_the_user_chats` | start and check; update / cancel / list not written (in process: `test_react_async_subagent.py`) |
| BG-11 | not written | | needs a live Claude session moved across workers: pending bg-claude-da |
| BG-12 | written, passing | `test_s17::test_a_result_that_arrives_while_the_chat_is_busy` | |
| WF-01, WF-02 | written | `test_s12::test_an_approval_mid_workflow_resumed_on_another_worker`, `test_langgraphs_own_interrupt_mid_graph` | graph with Send, subgraph, branch; checkpointed and not; pre-merge failures were SCN-2/3 (fixed since); not re-run whole |
| WF-03 | written | `test_s12::test_a_worker_killed_mid_workflow` | as above |
| WF-04 | written | `test_s12::test_cancel_mid_workflow` | one node (the slow step), not every node |
| WF-05 | written | `test_s12` (`wf_deepagents`: todos, a `task` sub-agent) | failed pre-merge on a scripted-model parsing error of the test's own (fixed); not re-run |
| WF-06 | written | `test_s12` (`wf_handoffs` with a cycle, `wf_tools`) | handoff agents need their own tools (documented) |
| WF-07 | written, xfail(strict) | `test_s16::test_a_nodes_retry_policy_runs_a_failed_read_exactly_twice` | SCN-11 (langchain-fixes) |
| WF-08 | written, xfail(strict) | `test_s05::test_claude_workflow_is_one_child_run` | "bg-claude-da" |
| LR-01 | written, passing | `test_s13::test_a_turn_working_for_minutes_at_real_length` (200 s, the product's own hold), `test_a_turn_working_well_past_its_hold` | |
| LR-02 | written, passing | `test_s13::test_a_two_hundred_step_task`, `test_s06::test_a_step_limit…`, `test_s07` | |
| LR-03 | written, passing | `test_s06` (time limit, budget), `test_s13` (budget, deadline queued and working) | |
| LR-04, LR-05 | written | `test_s13::test_a_queued_long_task_across_workers` (graceful, killed) | not run after the pass was stopped by the machine's memory limit |
| LR-06 | written, passing | `test_s07::test_fifty_five_turns` | 6 adapters; spending exact on ReAct and Kimi |
| REL | written | `test_s06`, `test_s16`, `test_s11` | SIGKILL while running / paused / resuming / mid-workflow; memory down; agent-runs slow or down; model error; not written: disk full, clock skew, network partition, duplicate webhook delivery (need fault injection the stack script does not have), model 429/overflow (the provider route's, features job) |
| APR-01/02/03/05/07, MEM-08 | written, passing | `test_s04` | Kimi re-run after a fix of the test's scripted model: one open observation, SCN-12 |
| SCH | written, passing (react-fire re-run) | `test_s08` | one case waits for the real ticker |
| FS | written, passing | `test_s09` | sandbox provider is the in-process stand-in (no Docker daemon) |
| CON, INV-7 | written, passing | `test_s10`, the invariants on every test | |

Open observation, not diagnosed: **SCN-12** — `test_s04::test_always_allow_in_one_chat_applies_in_another[kimi]`
passes its assertions and breaks INV-4 ("note ran though it was rejected") on the probe run it
rejects: either Kimi made the rejected call again and it ran under the standing approval learned
meanwhile, or the invariant misreads Kimi's events. Owner to decide: bg-kimi / scn-fixes.

### Round 3 findings (scenarios job)

| ID | Pri | Plan | What | Root cause | Status |
|---|---|---|---|---|---|
| SCN-13 | P0 | REL | A server cut off from agent-runs while it works a turn: agent-runs ends the run (`holder_lost`), and when the network is back the server carries the turn on — a write ran 30 s after its run was ended | `harness.py:490` `Harness._lost` stops the attempt only when the store has the run `CANCELLED`; a run the store ended (`ERROR`) is read back and left to go on | failing tests (xfail strict): `tests/integration/test_scenario_findings.py::test_scn13…` (in process), `tests/scenarios/test_s18_faults.py::test_a_network_cut…`; proposed: stop the attempt for any ended status, writing nothing (as a lost lease does, `pipeline.py:876`); owner: scn-fixes |
| SCN-14 | P2 | REL, FS | A file that did not fit the file store left its first bytes there, so the next, small file failed too | `files.py` `FileStore.put`: `pipe_file` with no clean-up on failure | **fixed** (the half-written file is removed), tests `tests/integration/test_file_store_full.py`, scenario `test_s18_faults.py::test_a_full_disk_under_the_file_store` |
| BG-11 | P1 | BG-11 | A background task's outcome, on the server that keeps its live Claude session, is not told to the conversation's next turn when that lands on another server (the task's run does end `SUCCESS` with its result; nothing is left `RUNNING`) | the notification is the live session's (in one process); nothing carries it to the conversation elsewhere | failing scenario (xfail strict) `test_s17_background_more.py::test_the_next_turn_lands_on_another_server…`; owner: bg-claude-da |
