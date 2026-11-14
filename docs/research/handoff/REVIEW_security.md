> **Status note (2026-11-14).** This review was written against the branches as they stood
> before the fixes of this round. Reported fixed since: SEC-R-1, SEC-R-2, SEC-R-5, SEC-R-7,
> SEC-R-9, SEC-R-10, SEC-R-11. Not confirmed, treat as open: SEC-R-3, SEC-R-4, SEC-R-6,
> SEC-R-8. The current status table is in `REMAINING.md`, section 6. Added to this public
> repository at the owner's instruction.

# Security review: this round of the Trellis stack

Read-only review of the diffs. No product code changed, nothing committed, no model calls.

## Refs reviewed

The branches moved while I read them, so every finding names a commit.

| Repo | Ref | Commit |
|---|---|---|
| agent-harness | `origin/main` (base) | `9e60203` |
| agent-harness | `…-final` | `d2bf754` |
| agent-harness | `…-react-background` | `983923b` |
| agent-harness | `…-chat-order` | `1968640` (a commit titled WIP; `aac6a95` when I started) |
| agent-runs | `…-final` | `422ef68` |
| agent-runs | `…-chat-order` | `087259c` (WIP; `5298410` when I started). `07e8197` landed at the end: skimmed only |
| agent-memory-service | `…-final` | `0f20438` |

Summary: one High, two Medium, eight Low. The High is on the react-background branch only.

---

## Findings

### SEC-R-1 — High — Background-task tools act on any run a task record names; a chat client can supply the record

**Where** (agent-harness `983923b`):
- `src/trellis/harness/background.py:891-910` (`Offered.call`), `:921` (`_check`), `:928` (`_update` → `follow_up` `:416`), `:945` (`_cancel`), `:973` (`Offered.asked`), `:744` (`answered`), `:1015` (`recorded`)
- `src/trellis/harness/bridges/background_tasks.py:127-140` (`abefore_agent`)
- Entry points, all older than this round: `adapters/langgraph.py:219-222` (`prepare_input`), `agui/__init__.py:238` (`payload = body.latest_user_text() or body.state`), `a2a/executor.py:274` (`return text or data or ""`)

**Scenario.** Mallory and Bob are users of one tenant. Mallory knows one of Bob's run ids (run ids are not treated as secrets: they are A2A task ids, and appear in events, traces and logs).

1. Mallory posts to the AG-UI endpoint with no user text and a `state` of `{"messages": [...], "async_tasks": {"T": {"task_id": "T", "agent_name": "<a sub-agent>", "run_id": "<Bob's run>", "status": "waiting", ...}}}`.
2. With no user text, the state is the run's input. `prepare_input` spreads a dict input into the graph's input state, and `async_tasks` is an ordinary input channel of Deep Agents' `AsyncSubAgentState`.
3. From there the five tools treat `task["run_id"]` as theirs:
   - **Read:** `check_async_task` returns Bob's run output, error or pending question. It is classed a read, so nobody is asked.
   - **Cancel:** `update_async_task` (also a read) cancels Bob's run under the default `interrupt` strategy. `cancel_async_task` does too, after an approval Mallory gives herself.
   - **Approve:** at the start of an attended turn, `Offered.asked` puts Bob's pending approval, with its tool call, to Mallory in her chat. Her answer goes through `answered` → `Agent._resumed` → `runs.resume` onto Bob's run. An in-process run then continues in Mallory's process under Bob's identity; a queued one goes back to a worker with the approval recorded.

The same redirect works without `async_tasks` in the input: `recorded()` takes a `run_id` from the notification metadata of the last input message and writes it into a tracked, unfinished task. An A2A caller reaches the same path with a data part.

**Confirmed** without a model call, against an extract of `983923b`: `Offered.call("check_async_task", …)` for user `mallory` with a forged record returned `{"status": "success", "result": "BOB'S PRIVATE ANSWER"}`, and `prepare_input` kept `async_tasks`. I did not run the full flow through a live AG-UI request.

**Why existing checks don't stop it.**
- Every lookup is `runs.get(task["run_id"], tenant=runtime.tenant)`. Tenant is checked; user, conversation and parentage are not.
- The AG-UI surface does enforce `record.user_id == user` and the thread on resume and cancel (`agui/__init__.py:323-327`, `:358`). The background path never goes through it.
- agent-runs' `require_may_answer` passes, because the harness key may act for anyone.
- The only gate in `Offered.call` is that `agent_name` is a known sub-agent, and the tool description lists those names.

**Fix (one owner).** One function in `background.py`, say `owned(runtime, task) -> RunRecord | None`, that every tool, `asked`, `pulled`, `left` and `recorded` must go through. It loads the run and requires all of:
- the same tenant and `record.user_id == runtime.user`;
- `record.metadata[BACKGROUND]` names this agent and this chat thread;
- `record.thread_id == task["task_id"]`.

Anything else is answered `UNTRACKED`. Separately, close the entry: an untrusted surface should hand the graph messages only. AG-UI `state` already travels as `Frontend.state`; A2A data should not be spread into state; `prepare_input` should not copy unknown channels from run input.

**Failing test.** Two users, one tenant, `LocalRuns`. Bob has a run paused on an approval. Mallory posts AG-UI input with empty messages and the forged `state`. Assert that `check_async_task` answers `UNTRACKED`, Bob's run is still `PAUSED` with no resolution, and no interrupt of Mallory's run carries Bob's tool call. Add a unit test that `recorded()` ignores an event whose `run_id` is not a run of this conversation.

---

### SEC-R-2 — Medium — Background tasks have no cap and can keep themselves going

**Where** (agent-harness `983923b`): `background.py:400` (`start`), `:304` (`Tasks.run`), `:578` (`notify`), `SIDE_EFFECTS` (start and update are reads); `bridges/background_tasks.py:116` (`awrap_model_call`).

**Scenario.** A user, or text inside a tool result or a task's own result, tells the model to start many tasks. The limits that do not apply:
- `start_async_task` is a read, so no person is asked.
- Each task is a child run with the sub-agent's own budget and time limit, not the turn's.
- Tasks outlive the turn: `_concluded` keeps them past the turn's timeout.
- Nothing counts live tasks per conversation, user or tenant, and `Tasks._at_work` is unbounded.

When a task ends and the chat is idle, `notify` starts a new turn of the agent with the task's result as input. That turn has a front end, so it is offered the task tools again. A result that says "start another" therefore produces task → notification turn → task with no user message, each run on a fresh budget.

**Why existing checks don't stop it.** Budgets, the step limit and the stall rule are per run. `stale()` only suppresses outcomes already told. The result is HTML-escaped in the notification, which stops it posing as the system, not the model following it.

**Fix (one owner).** `background.start` owns admission:
- a cap on unfinished tasks per conversation (and per user or tenant);
- a generation counter in the `BACKGROUND` / `NOTIFIED` metadata, so a notification turn inherits its chain's depth and a chain past N starts nothing;
- task spend charged to a conversation-level allowance.

**Failing test.** A scripted model that answers every notification with one `start_async_task`; assert the chain stops at the cap. A second test: one turn that emits 50 start calls gets at most the cap.

---

### SEC-R-3 — Medium — A cancel from another process is no longer prompt, and the holder ignores it (WIP pair)

**Where:**
- agent-runs `087259c` `src/agent_runs/store/runs.py:606` (`cancel`), `:816-839` (`_hold`), `:993` (`end_abandoned`)
- agent-harness `1968640` `src/trellis/harness/holding.py:143-157` (`_beating`), `harness.py` `_lost`

**Scenario.** A turn runs on replica A. The user presses stop, or sends `mode="interrupt"`, and the request lands on replica B (or goes straight to agent-runs).
- At `5298410` the run became `CANCELLED` at once; A's next heartbeat was refused and `_lost` stopped the attempt, within about 20 s.
- At `087259c` a held run is only asked to stop: the hold answers `cancel_requested: true` and is not extended.
- `_beating` discards the heartbeat's result, and nothing in the harness reads `cancel_requested` for a hold.
- So A keeps executing until the hold lapses (up to 60 s), the ticker cancels the run, and a later heartbeat is refused (up to 20 s more).

For roughly 80–100 s after the stop, the agent keeps running reads, announced writes and calls already approved for the run. The interrupting message waits in line behind it.

**Why existing checks don't stop it.** The store half is done; the harness half is not. `_lost` fires only on a refused heartbeat, and the heartbeat now succeeds until the lapse. On `…-final` (`d2bf754`) `_beating` ignores a refusal altogether, as `main` did: there a cancel from another process never stops the attempt.

**Fix (one owner).** `Holds._beating` reads the lease. On `cancel_requested` it calls the `lost` path with a "cancel asked" reason, which cancels the runtime and finishes the run `CANCELLED` (or releases the hold). The two WIP commits should land together.

**Failing test.** Two `Harness` instances on one store. A runs a turn blocked in a slow tool; B calls `agent.cancel`. Assert A's `runtime.cancelled` is set within one heartbeat period and the tool task is cancelled, with a fake clock so the test does not sleep.

---

### SEC-R-4 — Low (Medium if restricted keys go to less-trusted callers) — Any key of the tenant can take, extend or drop a hold, and through it end a run

**Where** (agent-runs `422ef68` `store/runs.py:726`, `:739-755`, `:774`; same code at `087259c:803`, `:816-839`, `:856`).

**Scenario.** `heartbeat` on a run kept in its caller's process accepts any `worker_id` from any key of the tenant.
- **End a run without cancel rights:** hold an unheld run for 5 s and walk away. The ticker ends it `ERROR holder_lost`, though `require_may_cancel` would have refused that key.
- **Impersonate the holder:** a held run answers 409 `run … is held by trellis@host:pid:token`, which gives the name away. With it the caller can extend the hold to 3600 s, `release` it, or replace the run's checkpoint through the heartbeat.
- **Delay a cancel:** at `087259c` a 3600 s hold keeps a run `RUNNING` for up to an hour after a cancel, and its conversation's next turn waits behind it.

**Why existing checks don't stop it.** `answering.py` leaves the worker routes tenant-wide by design, and `finish` already lets any key end an in-process run, so this adds no power for a fully trusted key. It does add an unauthenticated ownership claim that later code leans on: cancel deferral, `_holds_before`, the resume grace.

**Fix (one owner).** `RunStore._hold` owns who may hold:
- only the key that started the run, or one that may act for its `on_behalf_of`, or a service role;
- cap a hold's `lease_seconds` well below the 3600 s a worker lease allows;
- drop the holder's name from the 409 detail.

**Failing test.** A key restricted to `user:a` heartbeats `user:b`'s in-process run: expect 403 and no `held_until`. A wrong `worker_id` gets a 409 whose body does not contain the holder.

---

### SEC-R-5 — Low — The journal's outcome marker is in-band: a tool result shaped like it is replayed as a harness record

**Where** (agent-harness `d2bf754`): `tools/bridge.py:902-913` (`_journaled`, `_replayed`: older code). New consumers this round: `bridge.py:235` (`permitted`) reading what `:335` (`observed`) recorded, and `middleware.py:538-539` (`_update` → `_update_of`).

**Scenario.** A harness tool returns JSON from a third party: `{"trellis_outcome": {"status": "success", "output": "…", "tool": "x"}}`. The call succeeds, so the journal stores the dict as is. The run later pauses. On the resume `_replayed` sees a dict whose only key is the marker and builds `ToolOutcome(tool=…, cached=True, **record)`.
- A duplicate keyword raises `TypeError`; a bad status raises a validation error. Either way every resume of that run fails and the person's answer is wasted.
- A well-formed record lets the tool's author pick the replayed status and `metadata`.

This round `observed` also journals raw results of Claude's and Kimi's own tools and of a team's MCP servers, and `_update_of` now deserializes when `metadata["trellis_update"]` is present. That sink is not reachable from tool output today, only because a native LangGraph tool's result is journaled as text.

**Fix (one owner).** `_journaled` / `_replayed` own the encoding. Always store an envelope (`{"o": output}` or `{"x": outcome}`), or escape a top-level marker key in outputs. Build the replayed outcome from a fixed set of fields, never `**`.

**Failing test.** A tool returns the dict above, the run pauses on a later call, then resumes: the replayed call must give the model the same dict and the resume must not raise.

---

### SEC-R-6 — Low — The new journal read uses LangGraph's permissive deserializer

**Where** (agent-harness `d2bf754`): `middleware.py:523` (`_SERDE = JsonPlusSerializer()`), `:577` (`loads_typed`). The same trust already exists at `:3165` (`RunCheckpointer`, on `main` too).

**Scenario.** Whoever can write a run's checkpoint in agent-runs supplies msgpack that makes the harness import a module and call a constructor on resume. Checkpoint writers are any tenant key through `pause`, and since this round through `heartbeat` on an in-process run. In a harness shared by several tenants that crosses a tenant boundary. The worktrees have no dependencies installed; in a sibling worktree's venv, langgraph-checkpoint 4.3.2 defaults to `allowed_msgpack_modules=True` (warn only) unless `LANGGRAPH_STRICT_MSGPACK=true`, and I am assuming the locked 4.2.0 behaves the same.

**Why it is Low.** It adds a sink, not a new trust: a journal writer can already forge recorded approvals. The trust statement belongs in the deployment docs either way.

**Fix (one owner).** One serializer for journal data, built with an explicit allowlist (LangGraph and LangChain message, `Command` and `Send` types), used by both `RunCheckpointer` and `_update_of`; or set strict mode at start-up and fail closed.

**Failing test.** A journal whose `trellis_update` holds a msgpack ext naming `os.system`: `_update_of` must raise or return the message, and must not call it.

---

### SEC-R-7 — Low — Claude's carried approval falls back to the tool's name when a call has no id

**Where** (agent-harness `d2bf754` `adapters/claude.py:1318`, `:1337`, `:1380`, `:1397`).

**Scenario.** `steps.answered` holds the ids of calls a person approved in `PreToolUse`, so `can_use_tool` does not ask twice. Both sides fall back to the tool name when no `tool_use_id` arrives, and the set is never cleared. With a CLI or relay that omits the id on both sides, one approved `Bash` call puts `"Bash"` in the set and every later `Bash` call of the query is allowed unasked, whatever its arguments. The bundled CLI sends ids, so this is latent.

**Fix (one owner).** `Steps` owns the carry: add only a real id, treat "no id" as not answered, and discard the id in `Steps.ended`.

**Failing test.** Drive `decide` then `can_use_tool` with no `tool_use_id` for an approved call, then a second call with different arguments: the second must reach `bridge.asked`.

---

### SEC-R-8 — Low — An edited computer action may name any attribute of the computer object

**Where** (agent-harness `d2bf754` `tools/convert/openai_local.py:320`).

**Scenario.** When an approver's EDIT or a `before_tool` hook changes a computer action, the action runs as `getattr(self._inner, edited.pop("type"))(**edited)`. The edited `type` is not checked against the computer interface, so an approver can invoke any method of the team's computer object, with keyword arguments of their choosing.

**Fix (one owner).** `_act` checks the edited `type` against `AsyncComputer.__abstractmethods__` and binds the arguments to that method's signature; otherwise the call is refused as arguments that do not fit.

**Failing test.** Edit an approved `click` to `{"type": "close"}`, or to a private helper: expect a refusal and no call.

---

### SEC-R-9 — Low — `outbound.validate_url` is lexical, and its new caller has nothing behind it

**Where** (agent-harness `983923b` `outbound.py:23-41`, `react.py:656-668`, `a2a/push.py:45-54`, `:88-89`).

- `https://localhost./`, `https://x.internal./` (trailing dot) and `https://2130706433/` pass `validate_url`, as does any name that resolves privately.
- A2A push is still covered, because `check_addresses` resolves and refuses. `react._remote`, the only new caller, does the lexical check alone.
- Impact is small: the webhook is the developer's own setting and is called by their sub-agent server, not by the harness. But the module's docstring now presents this as the rule for every address.
- A2A push still resolves once to check and again to connect. https with certificate verification makes a rebind to an internal address fail in practice.
- No new caller skips the rule: `async_tasks.py` reaches the sub-agent server only through Deep Agents' own tools and the developer's `url`.

**Fix (one owner).** `outbound.py` owns both halves. `validate_url` strips a trailing dot and refuses all-numeric hosts. A `checked(url)` does the resolve-and-refuse, and returns the address so the connection can be pinned to it.

**Failing test.** Each of the three URLs above is `TargetRefused`.

---

### SEC-R-10 — Low — Feedback `metadata.args` is stored raw, around the tool's `redact` paths, and now up to 64 KiB

**Where** (agent-memory-service `0f20438` `api/validation.py:34`, `:63-76`; `modules/feedback/service.py:883`, `:1113`, `:1145`).

**Scenario.** A tool declares `redact: ["auth.token"]`. Its recorded invocation keeps `args_redacted` and a hash of the raw arguments. When a person approves the call, the harness sends the raw arguments as `metadata.args` (they must hash the same), and the feedback row keeps them and returns them in `FeedbackResponse.metadata`. The value the tool said never reaches storage is stored. Small arguments already fitted before; this round raises the bound from 8 KiB at depth 2 to 64 KiB at any depth.

**Fix (one owner).** `FeedbackService._project_tool_call` owns what is kept: once `_backed` has matched, store the arguments' hash and the descriptor's `redacted_args`. `_spent` then compares stored hashes.

**Failing test.** Register a tool with `redact`, record a call, post approving feedback with the raw arguments, read the feedback back: the token must not be in it.

---

### SEC-R-11 — Low — A conversation's line is listed by thread name, so other users' runs are read into it

**Where** (agent-harness `1968640` `line.py:103-125`).

**Scenario.** `Line.ahead` lists every `RUNNING` and `QUEUED` run of the agent with this `thread_id`, over all pages. A thread id is a per-user name, so "main" is shared by everyone who uses it. Other users' runs are dropped only after a `runs.get` each, and are read again whenever their `updated_at` moves. One user with many runs on a common thread name makes every other user's waiting turn do that work on each wake. A listing that pages while rows move may also miss the caller's own run, which `ahead` reads as "left the line", so the turn is dropped; I did not verify the cursor's stability.

**Fix (one owner).** agent-runs lists by `concurrency_key`, which it already indexes (`ix_runs_concurrency`), and `Line` asks for exactly that key.

**Failing test.** 200 running runs of another user on the same thread name: `ahead()` makes no `get` for them and returns `[]`.

---

## Reviewed and found sound

- **Conversation, session and turn registries** (`final`, `chat-order`). Keyed by `conversation_key`, a digest of the JSON of tenant, user and thread, or by run id, which agent-runs keeps unique across tenants. This covers `LiveSessions`, `Agent.sessions` / `turned` / `results`, `Turns` and the carried-state folder.
- **Claude live sessions, between turns** (`claude_live.py`, `d2bf754`). With no turn and no background run, every `PreToolUse` relay denies, `can_use_tool` denies, and a harness tool call returns an error. A background subagent's calls are decided in its own run's attempt and refused once that run is over. Its question is asked in the same conversation. `gone()` reads the run and interrupt from a payload the harness wrote, with the `SUBAGENT` key set last. One fairness note: the 16 sessions per agent are shared by all users, and a full table falls back to a process per query.
- **Approval bookkeeping** (`journal.py`, `bridge.py`). Approvals are keyed by tool and arguments; a rejection belongs to the call id that got it; `approvals_read` is context-local; the approval payload is harness-built. `runtime.decided` reuses an answer only for the same decision and the same arguments. A call replayed as "ran before" is refused, not re-run.
- **OpenAI Agents.** Local tools (shell, patch, custom, computer) all go through `bridge.call`, and a pause the SDK swallows is raised again at the next hook. A `Handoff` now hands off to the governed copy and fails closed when the SDK field is missing. An MCP call raises out of the SDK while the run is pausing.
- **LangGraph tool nodes.** Retries count as one call. A paused call waits for its siblings. A kept `Command` is written only for the call it belongs to.
- **agent-runs SQL and locks.** All SQLAlchemy expressions, no string-built SQL; the one `text()` is a constant index predicate. Advisory lock ids are bound parameters and include the tenant. Heartbeat, release, cancel and resume load the row by tenant and run under `FOR UPDATE`. A queued or leased run is still fenced by its worker id. `end_abandoned` touches only runs that were held. `_placed` runs after the answer's authorization check. `07e8197` moves the key lock to the end of the transaction, which removes the lock-across-payload-write I had noted at `087259c`.
- **Memory-service validation** (apart from SEC-R-10). The 32-key limit and the bounds on the other keys are unchanged; `args` is bounded like recorded tool arguments; the body cap is unchanged. Very deep nesting gives a 400 or 500, as it already did for other fields.
- **Subprocess, temp files, deserialization.** No new shell or argv construction (the ACP diff changes only who is named in the question). `LocalRuns` uses `mkdtemp` (0700) and removes its own folder. No pickle, YAML or eval was added.
- **Secrets.** No keys or headers in new logs, events or errors. A sub-agent server's `url` goes into run metadata without its headers (a URL with embedded credentials would be stored). The holder name is host, pid and a random token.

## Not reviewed, or not verified

- **What the Claude CLI does when the governance hook raises.** `decide` (`claude.py:1313`) catches only `Paused`; any other exception becomes an error response from the SDK. Strings in the bundled CLI suggest some hook failures refuse the call, but I could not confirm it for callback hooks. If the CLI treats it as "no decision", a team running with `allowed_tools` or a permissive mode has no second check. Suggested: wrap `pre_tool_use` to deny on any unexpected exception, and add a scripted-CLI test.
- **Live-session gateway headers.** `_shape` leaves `CUSTOM_HEADERS` out of the comparison, so a kept process may keep sending its first turn's headers. Same tenant and user; whether budgets or traces at the gateway are attributed to the wrong run needs a look at `GatewayHeaders`.
- **No test suites were run and no live services were used.** I ran one offline check, for SEC-R-1. The agent-runs holding and line behaviour was read, not run against Postgres.
- **Older code the diffs lean on,** read only as far as needed: `Agent.cancel` / `resume` authorization at the surfaces, `governance.check`, `files.part`, the sandbox providers, Kimi's home and environment, `a2a/client.py`, and the memory service's feedback authorization.
- **`agent-contracts` and `bifrost-sdk`** were out of scope. The Alembic migration for `holder` / `held_until` is outside `src` and was not reviewed.
- **Method.** I followed the method in the request directly and did not run the gstack `/cso` or `/review` skills.
