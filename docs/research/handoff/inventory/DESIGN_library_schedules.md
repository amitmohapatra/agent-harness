# Design: the libraries' own schedule tools, run through agent-runs

Design job with probes, 2026-11-14. No product code was changed, nothing was committed, no paid
model was called. Every probe ran a real CLI against a scripted local endpoint.

The owner's requirement (verbatim): "We are enterprise framework. Also if any libraries have
schedule, if can leverage those in harness that is better but they should work like harness and
runs involved like today. Where not possible should use harness middleware."

## 0. Result in one page

| Library | Its scheduling feature | Verdict | Mechanism | Contract change |
|---|---|---|---|---|
| Claude Agent SDK / Claude Code | `CronCreate` | **A** (library tool, backed by agent-runs) | `PreToolUse` hook creates the schedule through `schedules.py` and rewrites the CLI's own job into an inert one; `PostToolUse` hook replaces the tool output with the agent-runs record | none |
| | `CronList` | **A** | `PostToolUse` hook replaces the output with the user's schedules from agent-runs | none |
| | `CronDelete` | **B** (`cancel_schedule`) | no seam: the CLI checks the id against its own store before any hook runs | none |
| | (no library tool for change, pause, resume) | **B** (`update_schedule`) | the library has no such tool | none |
| | `ScheduleWakeup` and the dynamic `/loop` | **B** (stays disallowed) | its only use is a fired turn re-arming itself every 1 to 60 minutes; a fired harness run sets no schedules, and agent-runs fires at most hourly | none (an agent-runs change would be needed to allow it: section 3.4) |
| | `/loop <interval> <prompt>` (bundled skill) | follows `CronCreate` | the skill calls `CronCreate`; an interval under the floor is refused in a sentence, as today | none |
| | `RemoteTrigger` (cloud routines) | **B** (add to the disallowed list) | runs in Anthropic's cloud, outside governance, memory and the thread; only enabled with a claude.ai login | none |
| Kimi Code | `CronCreate`, `CronList`, `CronDelete` | **B** (the four harness tools, as today) | no seam: ACP can only allow or reject a call, Kimi's hooks can only block, MCP tools are always named `mcp__<server>__<tool>`, and Kimi keeps its copy in a session file on one machine | none |
| OpenAI Agents SDK 0.24.0 | none | **B** (as today) | nothing to leverage | none |
| LangChain 1.5 / LangGraph 1.2.22 | none in the library; cron only on LangGraph Server through `langgraph_sdk` | **B** (as today) | nothing in process to leverage | none |
| Deep Agents 0.7.38 | none | **B** (as today) | nothing to leverage | none |

Verdict letters: (A) the library's own tool, backed by agent-runs; (C) an adapter presenting the
library's tool shape over the harness schedule; (B) the harness's schedule tools replace it.
No tool gets (C): where the built-in cannot be backed, an MCP copy would be named
`mcp__trellis__CronCreate`, which is not the library's tool and adds a second set of tool
definitions for the same work.

**One decision for the architect before any code (D1, section 7).** For Claude, the seam is
complete for create and list and absent for delete. Taking (A) for the two tools gives the
model Claude's own `CronCreate` and `CronList`, and costs four harness behaviours at set time
(section 3.3). Keeping (B) for all of Claude costs nothing and changes nothing. The owner's
rule ("prefer the library") points to (A) for the two tools; the plan in section 6 implements
that, in steps that each leave the tree working.

All verdicts end in the one existing path: `src/trellis/harness/schedules.py` on agent-runs'
schedules. No second scheduler, no second store, no copy of the limits.

---

## 1. What each library has (verified)

Evidence notation: `H:` is `/var/tmp/cons/final-agent-harness/src/trellis/harness/`;
`CLI283` is the Claude Code 2.1.283 binary bundled in claude-agent-sdk 0.2.160
(`/home/user/agent-harness/.venv/lib/python3.13/site-packages/claude_agent_sdk/_bundled/claude`),
quoted by byte offset, with the extracts kept in `/var/tmp/cons/inventory-tmp/schedules/`;
`CLI325` is 2.1.325 bundled in 0.2.182 (`/var/tmp/cons/bgclaude-tmp/sdk182`);
`K262:` is a line of Kimi Code 2.6.2
(`/root/.kimi-code-npm-262/lib/node_modules/@moonshot-ai/kimi-code/dist/main.mjs`), `K211:` of
2.1.1 (`/root/.kimi-code-npm/...`). "Probe" names a log under
`/var/tmp/cons/inventory-tmp/schedules/probe/`.

### 1.1 Claude Code (Agent SDK 0.2.160 / CLI 2.1.283, checked again on 0.2.182 / 2.1.325)

**Tools the model is given** (probe `m1.log.schemas`, the schemas as sent to the Messages API):

- `CronCreate` — input `{cron: string, prompt: string, recurring?: boolean, durable?: boolean}`,
  required `cron`, `prompt`, no other properties. Output object `{id, humanSchedule, recurring,
  durable?}` (CLI283 offset 227042695, `cron_create_tool_283.txt`); 2.1.325 adds `nextFireAt`
  (epoch ms) (probe `run3.jsonl`). The model reads a sentence the CLI builds from that object:
  `Scheduled recurring job <id> (<humanSchedule>). <Session-only (not written to disk, dies when
  Claude exits) | Persisted to .claude/scheduled_tasks.json>. Auto-expires after 7 days. Use
  CronDelete to cancel sooner.` or, one-shot, `Scheduled one-shot task <id> (<humanSchedule>).
  <...>. It will fire once then auto-delete.`
- `CronList` — input `{}`. Output `{jobs: [{id, cron, humanSchedule, prompt, recurring?,
  durable?}]}`; the model reads one line per job, `<id> — <humanSchedule> (recurring|one-shot)
  [session-only]: <prompt, 80 chars>`, or `No scheduled jobs.` (CLI283 offset 227055647).
- `CronDelete` — input `{id: string}`. Output `{id}`; the model reads `Cancelled job <id>.`
  (CLI283 offset 227049666).
- `ScheduleWakeup` — input `{delaySeconds?: number, reason?: string, prompt?: string, stop?:
  boolean, noop?: boolean}`; output `{scheduledFor, clampedDelaySeconds, wasClamped, stopped?,
  cancelledWakeups?}`; delay clamped to 60..3600 s (`wakeup_283.txt`, `wakeup_impl_283.txt`).
  It creates a one-shot session cron of kind `loop` (probe `run1.jsonl`: `33 18 * * *`).
- `RemoteTrigger` — actions `list, get, create, update, run, create_webhook_trigger, list_runs,
  get_run_log` against `claude.ai` `/v1/code/triggers` (CLI283 offset 202916248). Enabled only
  with a first-party claude.ai login and policy (`isEnabled(){return Gn()&&pt()&&
  !a.CLAUDE_CODE_REMOTE&&Xt("allow_remote_sessions")&&Xt(TY)}`, offset 227092508). It was not
  in the tool list of any probe (API key, scripted endpoint).
- `/loop` is a bundled skill (a prompt), not a tool: with an interval it makes the model call
  `CronCreate`; without one, `ScheduleWakeup`
  (https://code.claude.com/docs/en/scheduled-tasks, "Run a prompt repeatedly with /loop").

In 2.1.283 the three `Cron*` tools are in the request's tool list. In 2.1.325 they are
**deferred**: named in a system reminder and loaded through `ToolSearch`; `ScheduleWakeup` is
still listed (probe `m3.log`, first request).

**Storage.** Session-only jobs live in the CLI process's memory. `durable: true` writes
`<project root>/.claude/scheduled_tasks.json`, guarded by `.claude/scheduled_tasks.lock`
(CLI283 offsets 202617987, 231284667; `cron_core_283.txt`, `cron_sched_283.txt`). Limit: 50 jobs
(`Too many scheduled jobs (max 50)`).

**Recurrence.** 5-field cron in the process's local time; steps, ranges, lists; no `L`, `W`,
`?`, names; day-of-month and day-of-week both set match either (docs, "Cron expression
reference"). `recurring: false` fires once and deletes itself. Recurring jobs expire 7 days
after creation: one last fire, then deleted (`recurringMaxAgeMs: 604800000`, offset 202617987).
Jitter: recurring late by up to half the period, at most 30 minutes (`recurringFrac: 0.5,
recurringCapMs: 1800000`); one-shots on :00 or :30 up to 90 s early (docs, "Jitter"). Minimum
interval one minute.

**Firing.** The scheduler ticks every second and fires only while the session is idle and its
input is open (`isLoading:()=>Fe.running||Fe.inputClosed`, CLI283 offset 221789770,
`cron_fire_print_283.txt`). A fire enqueues the job's prompt as a new turn:
`{mode:"prompt", value:<prompt>, priority:"later", isMeta:true, skipSlashCommands:true,
modelScheduledOrigin:true, wakeupSource:"schedule_wakeup"}`. Probe `run1.jsonl` / `m1.log`
(2.1.283) and `run3.jsonl` / `m3.log` (2.1.325): the model receives one user message whose text
is the prompt **verbatim** (`{"role":"user","content":[{"type":"text","text":"CRONFIRE ping"}]}`);
a `UserPromptSubmit` hook fires with `prompt` equal to it and **no `source` field**; the turn
ends in a `ResultMessage` whose `origin` is `None`. So `claude._own` (`H:adapters/claude.py:766`) cannot tell a cron-fired turn from one a caller sent.

**Durability.** Docs, "Limitations": jobs fire only while Claude Code runs and is idle; no
catch-up for missed fires; on `--resume` the session's `CronCreate` jobs are restored, except
expired recurring ones and one-shots whose time has passed. Probes confirm restore over the
SDK's `resume=` on both versions (`run5_283_b.jsonl`, `run5_325_b.jsonl`). The `Stop` hook input
lists the session's jobs (`session_crons: [{id, schedule, recurring, prompt}]`, both versions).
`CLAUDE_CODE_DISABLE_CRON=1` turns the tools and the scheduler off together (same gate:
`oH()`, offset 202631589).

**Why the CLI's own firing cannot meet the requirement** (candidate "let the CLI fire inside a
kept live session and record the turn as a run"): a job fires only in the one process that
holds the session, and only while that process is alive, idle and its input open. The harness
keeps a live session 300 s after its last activity (`H:bridges/claude_live.py:92`). A restart
loses session-only jobs unless the session is resumed, and nothing resumes it at fire time.
Missed recurring fires are not caught up. The durable file is one machine's disk and one
project folder (the harness's `cwd` is shared by every user of the agent:
`H:adapters/claude.py:434-436`). It fails "any replica", "survives restarts" and "missed
fires".

### 1.2 Kimi Code (2.6.2; the same in 2.1.1 unless said)

**Tools** (`K262:239608-239917`; probe `test_kimi_cron.py` on 2.1.1 for the shapes):

- `CronCreate` — `{cron: string, prompt: string (1..8192 bytes), recurring?: boolean = true}`.
  Result text: `id: <ULID>\ncron: <cron>\nhumanSchedule: <words>\nrecurring: <bool>\nnextFireAt:
  <local ISO time with offset>`. Refusals as error text: invalid cron, no fire within 5 years,
  cap of 50 per session, a one-shot more than about a year out (`K262:239649-239733`).
- `CronList` — `{}`. Result: `cron_jobs: <n>` then per job `id, cron, humanSchedule, prompt
  (200 bytes), nextFireAt, recurring, ageDays, stale`, or `No cron jobs scheduled.`
  (`K262:239832-239887`).
- `CronDelete` — `{id}` matching `^(?:[0-9a-f]{8}|[0-9A-HJKMNP-TV-Z]{26})$`. Result `Deleted
  cron job <id>.` or the error `No cron job with id <id>.` (`K262:239761-239794`).
- Main agent only (`K262:230040`). `CronList` is auto-approved; `CronCreate` and `CronDelete`
  ask (`K262:286587`); both are refused in plan mode (`K262:226768`).

**Storage.** Durable session events `cron.add`, `cron.delete`, `cron.cursor` in the session's
own log (`K262:135156-135171`, `K262:135480-135516`): the jobs belong to one Kimi session, are
reloaded when that session is resumed, and are cleared in a fork. Sessions are files under
`~/.kimi-code/sessions` on one machine.

**Recurrence.** 5-field cron in the process's local time; recurring jobs go stale after 7 days
(`STALE_THRESHOLD_MS = 6048e5`, `K262:135379`): the last fire carries `stale="true"` and the job
is deleted. Jitter: recurring late by at most min(10% of the period, 15 minutes); one-shots on
:00 or :30 at most 90 s early (`K262:135094-135119`).

**Firing.** A 1 s timer (`K262:135418-135422`); a fire is delivered only while the agent loop is
not running (`K262:135331-135336`) as a user message with this text (`K262:135064-135072`,
`K262:135239-135257`):

```
<cron-fire jobId="<id>" cron="<cron>" recurring="true|false" coalescedCount="<n>" stale="true|false">
<prompt>
<the prompt, verbatim>
</prompt>
</cron-fire>
```

Missed fires collapse into one delivery with `coalescedCount > 1`.

**Config and environment seams** (`K262:134740-134765`, the same names at `K211:218270-218291`):
`KIMI_DISABLE_CRON=1` (the tools stay listed and answer "Cron scheduling is disabled");
`KIMI_CRON_MANUAL_TICK=1` (no timer: a tick only on `SIGUSR1`); `KIMI_CRON_POLL_INTERVAL_MS`;
`KIMI_CRON_NO_JITTER`; `KIMI_CRON_NO_STALE`; `KIMI_CRON_CLOCK`. `config.toml` has
`[tools] enabled / disabled` lists that hide built-in tools (`K262:238768-238792`,
`K211:218066`); there is no environment variable or `kimi acp` flag for it (`kimi acp --help`
on 2.6.2: only `--login`, `--region`).

Probe (2.1.1, through the harness, with the forced `KIMI_DISABLE_CRON` and the tools' risk
changed in the test process's memory only): with `KIMI_CRON_MANUAL_TICK=1` the three tools work
and return the shapes above; `CronDelete` of an id Kimi does not hold answers `No cron job with
id ...`.

### 1.3 OpenAI Agents SDK 0.24.0

No scheduling: no tool, no API. A search of the installed package
(`/var/tmp/cons/oaiup/.venv/.../agents`) for `cron` and `schedul` finds only asyncio task
scheduling and retry delays (`agents/run.py:2595`, `agents/mcp/server.py:2567`).

### 1.4 LangChain 1.5.0 / LangGraph 1.2.22

No scheduling in `langgraph`, `langchain` or `langchain_core` (zero files mention `cron`;
"scheduled" in `langgraph` means nodes scheduled within a step). `langgraph_sdk` 0.4.14 has
`CronClient` (`langgraph_sdk/_async/cron.py:34`: `create_for_thread`, `create`, `delete`,
`update`, `search`, `count`; arguments `schedule`, `input`, `end_time`, `enabled`, `timezone`,
`webhook`, `multitask_strategy`). It is an HTTP client for a LangGraph Server ("not supported on
all licenses", `cron.py:52-56`): the schedule lives in that server and its fire is a run of that
server, not of this process. It is not a model tool.

### 1.5 Deep Agents 0.7.38

No scheduling. `schedule` appears only in one model profile's wording rules
(`deepagents/profiles/harness/_nvidia_nemotron_3_ultra.py:346-1409`) and a prompt sentence
(`deepagents/graph.py:119`).

---

## 2. The seams, tested

### 2.1 Claude: what a hook can and cannot do

Verified with the bundled CLIs 2.1.283 and 2.1.325 driven by the SDK against a scripted Messages
endpoint (`probe/model.py`, `probe/drive.py`).

| Candidate | Finding | Evidence |
|---|---|---|
| `PreToolUse` supplies the tool result | **No.** Its output is `permissionDecision` (allow, deny, ask, defer), `permissionDecisionReason`, `updatedInput`, `additionalContext`. A `deny` reaches the model as an error: `PreToolUse:CronCreate hook error: <reason>`, `is_error: true` | SDK `types.py:437-444`; probe `run2_c.jsonl` |
| `PreToolUse` rewrites the input | **Yes.** `updatedInput` changed the job the CLI made (cron, recurring, durable) | probe `run2_a.jsonl` |
| `PostToolUse` replaces the result | **Yes, for an object of the tool's own output shape.** `updatedToolOutput` `{id, humanSchedule, recurring, durable}` made the model read `Scheduled recurring job sch_0123... (every minute (server)). Persisted to .claude/scheduled_tasks.json. Auto-expires after 7 days. Use CronDelete to cancel sooner.`; a replaced `{jobs: [...]}` made `CronList` read our rows. A value of another shape is dropped and the original kept. 2.1.325 accepted the object without its new `nextFireAt` | SDK `types.py:447-462`; probes `run2_a`, `run2_b`, `run3` |
| A hook adds context | **Yes.** `additionalContext` from either hook arrives after the result as `<system-reminder>PostToolUse:CronCreate hook additional context: ...</system-reminder>` | probe `m2.log` |
| A hook sees `CronDelete` of an id the CLI does not hold | **No.** The CLI's `validateInput` runs first and answers `<tool_use_error>No scheduled job with id '<id>'</tool_use_error>`; neither `PreToolUse` nor `PostToolUseFailure` fires. Same on both versions | CLI283 offset 227049666; probes `run1`, `run2_a`, `run3` |
| Same-named tools from the harness's MCP server | **No.** An MCP tool is always `mcp__<server>__<tool>` to the model; the harness's own tools already reach Claude as `mcp__trellis__...` | `H:adapters/claude.py:1318`, `:1476` |
| Blocking the CLI's own fire | **Yes.** A `UserPromptSubmit` hook answering `{"decision": "block"}` stops the fired turn before any model request (a `ResultMessage` with `num_turns: 0`). Not needed by the design below; kept as a fallback | probe `run5_block.jsonl`, `m5.log` |
| Resume restores the CLI's copy | **Depends on the recorded result.** A job whose recorded output says `durable: false` is restored on `resume=` (with the model's original input, not the hook's rewrite: 2.1.283, `run4_283_b`). A job whose output the hook replaced (`durable: true`, the server's id) is **not** restored: 4 of 4 runs, both versions | probes `run4_*`, `run6_*` |

What follows from these facts:

- `CronCreate` and `CronList` can be the library's own tools with agent-runs behind them.
- `CronDelete` cannot: the model must hold an id the CLI process itself knows, and the harness
  cannot put one there. The CLI's ids exist only in the process that made them (or a session
  resumed with live copies, which would then also fire in a kept session).
- The CLI's own copy of a job can be made harmless: rewritten at `PreToolUse` into a one-shot a
  year away, and recorded with the replaced output so a resumed session does not bring it back.

### 2.2 Kimi: what ACP, hooks, MCP and config allow

| Candidate | Finding | Evidence |
|---|---|---|
| ACP `session/request_permission` | allow once, allow always, reject. No result, no rewritten input, no reason text (the harness already follows a rejection with a prompt of its own) | `H:adapters/acp.py:142`, `:1249-1300`; inventory `kimi.md` rows 16-19 |
| ACP `tool_call` / `tool_call_update` | the harness can read the call and its result text (so it could mirror a create and learn Kimi's id); it cannot change either | `H:adapters/acp.py:1208-1228`, `:1154-1162` |
| Kimi's own hooks (`[[hooks]]`) | `PreToolUse` can only block (`permissionDecision: "deny"`); `PostToolUse` is fire-and-forget | `K262:227252`, `K262:227993` |
| Same-named tools from the harness's MCP server | no: Kimi names MCP tools `mcp__<server>__<tool>` | `K262:137743`, `K262:240169` (`"mcp__*"`) |
| Turning Kimi's firing off while keeping its tools | yes: `KIMI_CRON_MANUAL_TICK=1` | `K262:135418-135427`; probe `test_kimi_cron.py` |
| Hiding the built-ins | only through `[tools] disabled` in `config.toml`; no flag or variable | `K262:238768-238792` |
| Kimi's copy of a job | a durable event in the session's file on one machine; reloaded on resume; removed only by `CronDelete` or by Kimi's own tick | `K262:135480-135516`, `K262:135311-135324` |
| Another replica | a session this machine does not hold is not resumed: the harness starts a new Kimi session | `H:adapters/acp.py:884-911` |

What follows: a create could be mirrored into agent-runs, but nothing can make `CronList` show
agent-runs' truth or make `CronDelete` accept an id Kimi's session does not hold. On another
replica, or after the session is gone, Kimi's list is empty and its delete answers `No cron job
with id`. With Kimi's timer off (it must be off, or it fires a second time), Kimi never removes
a fired one-shot or an expired job from its own list, and its cap of 50 per session fills. A
create-only split would leave the model holding Kimi's ULID and Kimi's wording ("Tasks fire
only while this session is alive and idle, and never carry over to a new session",
`K262:239623`) with no channel to correct either. Verdict (B) for all three.

---

## 3. Mapping (Claude `CronCreate` / `CronList`)

### 3.1 The library's arguments to `ScheduleSpec`

| `CronCreate` | `ScheduleSpec` / metadata | Notes |
|---|---|---|
| `cron` | `cadence` | 5 fields, checked by `schedules._refused` (croniter), then agent-runs |
| "local time" | `timezone` = `snapshots.zone(runtime)` or `"UTC"` | the CLI means the process's zone; the harness means the user's. The turn's context already says the hour in the user's zone when a schedule tool is offered (`claude._dated`, `H:adapters/claude.py:734`) |
| `prompt` | `input` (verbatim), `name` = first 200 characters, `metadata[TASK] = {"task": prompt, "condition": None, "once": not recurring}` | `TASK` marks a schedule set in chat, so `_mine`, `update_schedule`, `cancel_schedule`, `Fired.of` all work unchanged |
| `recurring: false` | `metadata[TASK]["once"] = True` | deleted after its run by `schedules.ended` (`H:schedules.py:197-220`), as today |
| `durable` | ignored | every schedule is kept by agent-runs; the hook sends the CLI `durable: false` |
| (none) | `on_behalf_of`, `tenant_id`, `agent_id`, `timeout_seconds`, `agent_version`, `metadata[THREAD]`, `metadata[TOUCHED]`, the run options of `agent._metadata` | exactly as `schedule_task` sets them (`H:schedules.py:431-455`) |

### 3.2 Back, for `CronList` and the result of `CronCreate`

| Output field | From |
|---|---|
| `id` | `schedule.schedule_id` (`sch_` + 32 hex). `CronDelete` is not offered, so the id only has to work with `cancel_schedule` and `update_schedule`, which take it today |
| `cron` | `schedule.cadence` |
| `humanSchedule` | the harness's own wording: `'<cadence>' (<timezone>)`, plus `, paused` when not enabled and `, only when <condition>` when the schedule has one (`schedules._label` gives the first part) |
| `prompt` | `metadata[TASK]["task"]` |
| `recurring` | `not metadata[TASK]["once"]` (the CLI omits the key when false) |
| `durable` | `True` on create; omitted in the list (the CLI's list marks only `durable: false`) |
| `nextFireAt` (2.1.325) | `schedule.next_fire_at` in epoch ms when enabled; omitted otherwise |

Any other key in the CLI's own response is copied through unchanged, so a field a later CLI
adds does not make the replaced object mismatch.

### 3.3 What cannot be represented

Library to harness:

- **A cron under the floor** (`*/5 * * * *`, which the tool's own description offers as an
  example): the harness's `TRELLIS_SCHEDULE_MIN_INTERVAL` (300 s) and then agent-runs' fixed
  floor of one run an hour (`MIN_INTERVAL = timedelta(hours=1)`,
  `/var/tmp/cons/final-agent-runs/src/agent_runs/domain/cadence.py:34`) refuse it. The model is
  told so in a sentence, as with `schedule_task` today. No change needed for parity with today.
- **The 7-day expiry of recurring jobs.** `ScheduleSpec` has no end time. Harness schedules do
  not expire (they ask "Still need this?" after 90 days untouched, `H:schedules.py:81`). The
  design keeps the harness behaviour and tells the model so (section 4.2). See D2.
- **The CLI's jitter** (late by up to half a period, 30 minutes at most; one-shots up to 90 s
  early): agent-runs' own applies (0 to 59 s after the tick, `cadence.py:36-38`, `:87-92`).
- **50 jobs per session**: the harness's own limit applies (20 per user and agent by default).

Harness to library (lost when the model sets a schedule through `CronCreate` instead of
`schedule_task`):

1. `condition` and "reported once until it stops holding". A fired run that answers exactly
   `NO_REPORT` is still not delivered (`schedules.quiet`, `H:schedules.py:187-194`), so a prompt
   that says so works; the once-only memory does not.
2. The question when a new schedule looks like one the user has (`_twin_asked`). The very same
   schedule again is still one schedule: agent-runs upserts on tenant, agent, user, cadence and
   input (`H:../docs/runs.md:381-383`).
3. `choices` for a vague ask.
4. `tools`: allowing a run's writes when the schedule is set. Without it, a fired run's write
   asks in the chat each time, which is the safe default.

`update_schedule` (time, task, condition, pause, resume) and `cancel_schedule` stay offered and
work on these schedules, so a condition can still be added afterwards.

### 3.4 Contract changes

**None is needed for the design.** Flagged for the record, each with the alternative that
avoids it:

| Would need | Only if | Alternative that avoids it |
|---|---|---|
| agent-runs: a configurable floor below one hour (`domain/cadence.py:34`); no agent-contracts change | the owner wants `/loop 5m`-style polling or `ScheduleWakeup` to work | keep the floor; the model is told in a sentence (today's behaviour) |
| agent-contracts: an end time on `ScheduleSpec` | the owner wants the library's 7-day expiry enforced by the store | none needed if harness schedules keep not expiring; or enforce it in `schedules.ended` from `created_at`, with no contract change |
| agent-contracts: a first-class "once" | never for this design | today's `metadata[TASK]["once"]` and the delete in `schedules.ended` |

---

## 4. The mechanism for Claude (verdict A for `CronCreate` and `CronList`)

### 4.1 The call

1. The model calls `CronCreate {cron, prompt, recurring?}` (its own tool, its own schema).
2. The existing `PreToolUse` hook (`claude._governance`, `H:adapters/claude.py:1297-1346`)
   governs the call as it governs any built-in: `before_tool` hooks, governance rules, the
   journal. `CronCreate` is declared a write that is announced, not asked, as `schedule_task` is
   (`Tool(confined=True)`, `H:schedules.py:404`).
3. Still in the hook, `schedules.py` makes the schedule: the same limits (`_limits`,
   `_refused`, `_full`), the same `ScheduleSpec`, the same store call. A refusal is the hook's
   `deny` with the sentence `schedule_task` would have returned; the model reads it as the
   tool's error, the way it reads any governance denial today.
4. On success the hook allows the call with `updatedInput`: the same prompt, `recurring: false`,
   `durable: false`, and a cron pinned to yesterday's date (day capped at 28), which is about a
   year away. The CLI validates and stores that inert job.
5. The existing `PostToolUse` hook (`claude._recorded`, `:1462-1500`) answers with
   `updatedToolOutput` built from the agent-runs record (section 3.2) and an
   `additionalContext` line (section 4.2). The model reads the CLI's own sentence with the
   server's id. The run records the call with the replaced output (what the model read).
6. If the CLI fails the call after the hook allowed it (`PostToolUseFailure`), the hook deletes
   the schedule it made.

`CronList`: the `PostToolUse` hook replaces `{jobs}` with the user's schedules
(`schedules._mine`: this user, this agent, this tenant, set in chat), mapped as in section 3.2.

A run paused and resumed makes the call again: the upsert returns the same schedule, so the
replay creates nothing twice.

### 4.2 What the model is told beside the CLI's sentence

The CLI's text is fixed ("Auto-expires after 7 days. Use CronDelete to cancel sooner." and
"Persisted to .claude/scheduled_tasks.json"). The `additionalContext` of the same result says
what is true here, in one line built once in `schedules.py`:

> Kept by the server as `<id>`, in `<zone>`: it runs even when this chat is closed and its reply
> comes back to this chat. It does not expire after 7 days. To change, pause or cancel it use
> update_schedule or cancel_schedule with this id.

### 4.3 What the run offers

| Tool | Offered when schedules are on and the run is not a fired one |
|---|---|
| `CronCreate`, `CronList` | yes (Claude's own) |
| `update_schedule`, `cancel_schedule` | yes (harness) |
| `schedule_task`, `list_schedules` | no (the library's tools do this work) |
| `CronDelete`, `ScheduleWakeup`, `RemoteTrigger` | never (disallowed) |

A fired run, or a run `without=["schedules"]`, or a run store that cannot schedule: `CronCreate`
and `CronList` are disallowed too. A live session keeps the tools its process started with, so
the `PreToolUse` hook also refuses `CronCreate` in a fired run it serves ("A scheduled run does
its task: it sets no schedules.").

### 4.4 Why the CLI never fires its own copy

- In the process that made it, the copy is a one-shot about a year away; a live session closes
  300 s after its last activity.
- A query's own process (not a live session) closes its input when the prompt is sent, and the
  CLI never fires with its input closed.
- A resumed session does not restore a job whose recorded output was replaced (probe, both
  versions).

So agent-runs is the only thing that fires, and `schedules.py` the only code that decides.

---

## 5. Firing

Unchanged path: agent-runs' ticker queues the run with the schedule's metadata; any worker
claims it; `pipeline` reads `Fired.of(record.metadata)` (`H:pipeline.py:379`); the run's thread
is the schedule's (`metadata["thread_id"]`, `agent_runs/firing.py:145-147`); the AG-UI thread
stream delivers it to an open chat and the conversation holds it for a closed one
(`H:agui/threads.py:142-226`). Missed fires (once after an outage), overlap (skip while the last
run has not ended), jitter, DST (computed in the schedule's zone), three failures pausing the
schedule, and the 90-day question are agent-runs' and `schedules.py`'s, untouched.

**Claude.** The fired run's prompt is the schedule's `input`, which is the cron's prompt
verbatim: the text the CLI itself would have enqueued (section 1.1). It goes to the
conversation's live session when one is alive on the worker that claimed the run, else to the
session resumed from the session store, as any next turn of that conversation does. Today's
`"Scheduled task: "` prefix (`schedules._input`, `H:schedules.py:633-641`) is not added to a
schedule set through `CronCreate`; a condition added later with `update_schedule` adds the
harness's condition wording as it does today.

One difference from the CLI's own fire, to confirm with a real run: the CLI enqueues its fire
with `skipSlashCommands: true`. A prompt that begins with `/` sent by the harness as a turn may
be read as a slash command.

**Kimi, OpenAI Agents, LangGraph, Deep Agents.** Verdict (B): nothing changes; the fired run is
asked `Scheduled task: ...` as today. Had Kimi been (A), the prompt would have been the
`<cron-fire ...>` envelope of section 1.2.

---

## 6. Implementation plan

Smallest steps; each leaves the suite green. Tests first in every step. Paths are in
`/var/tmp/cons/final-agent-harness`.

### Step 1. One function that makes a chat schedule (`src/trellis/harness/schedules.py`)

- Move the body of `_schedule_task` from the limits to the store call (`:423-458`) into
  `async def make(runtime, about, when, zone, *, input, allowed=()) -> Schedule | str` (a
  `str` is the sentence of a refusal). `_schedule_task` (`:407`) keeps its questions (twin,
  choices, writes) and calls `make` with `input=_input(task, condition)`.
- No behaviour change. Proof: `tests/integration/test_schedules.py` (20 tests) unchanged and
  green.

### Step 2. The library-facing entry (`src/trellis/harness/schedules.py`)

- `async def set_by_library(runtime, cron, prompt, *, recurring) -> Schedule | str`: refuses in a
  fired run or when schedules are off, then `make(...)` with `input=prompt`,
  `about={"task": prompt, "condition": None, "once": not recurring}`, zone as `_schedule_task`
  resolves it (`:413`).
- `def kept_note(schedule) -> str`: the line of section 4.2.
- `_mine` (`:674`) is made public as `mine`; `_label` (`:779`) gains the paused and condition
  suffixes used by section 3.2.
- Tests first (unit, `LocalRuns`): the spec made (cadence, zone, user, tenant, thread, input
  equal to the prompt, `once`); each refusal sentence (bad cron, under the floor, the user's
  limit, a fired run); the same call twice is one schedule; another user's and another tenant's
  schedules are not in `mine`.

### Step 3. What an adapter says its library has (`adapters/base.py`, `agent.py`)

- `Adapter` (`adapters/base.py:119`) gains `schedule_tools: frozenset[str]`: the harness
  schedule tools the framework's own tools stand in for. Claude:
  `{"schedule_task", "list_schedules"}`; every other adapter: empty.
- `Agent.tools_for` (`agent.py:998-1000`) leaves those names out of `schedule_tools(runtime)`.
- Tests first: a Claude run is offered `update_schedule` and `cancel_schedule` and not the other
  two; every other adapter is offered all four (the existing golden requests stay as they are).

### Step 4. Claude's hooks (`src/trellis/harness/adapters/claude.py`)

- `OUTLIVES_THE_QUERY` (`:196`) becomes `("CronDelete", "ScheduleWakeup", "RemoteTrigger")`, and
  its comment (`:188-195`) says why each stays. `_disallowed` (`:1258-1267`) adds `CronCreate`
  and `CronList` when the run offers no schedules (a fired run, `without=["schedules"]`, a
  store that cannot schedule).
- `BUILTIN_SIDE_EFFECTS` (`:146`): `CronCreate: "write"`, `CronList: "read"`; `CronCreate` is
  decided as a confined write (announced, not asked), the rule `schedule_task` has.
- `_governance.decide` (`:1312-1344`): after `bridge.permitted` lets a `CronCreate` through, call
  `schedules.set_by_library`; a `str` is `_decided("deny", sentence)`; a `Schedule` is kept on
  `steps` by call id and the answer is `_decided("allow", updated=<the inert input>)`.
- `_recorded.post_tool_use` (`:1471-1500`): for `CronCreate`, return `updatedToolOutput` (the
  CLI's response with `id`, `humanSchedule`, `recurring`, `durable`, `nextFireAt` overwritten)
  and `additionalContext = schedules.kept_note(...)`; for `CronList`, `updatedToolOutput` from
  `schedules.mine`; on `PostToolUseFailure` of a `CronCreate`, delete the schedule made. The
  recorded outcome is the replaced output.
- `_dated` (`:734-744`): the hour in the user's zone is added when `CronCreate` is offered, as it
  is for `schedule_task`.
- Tests first, in two kinds:
  - with the fake CLI (`tests/support/fake_claude_cli.py`): create, list, the
    refusals, governance (a `before_tool` hook that denies `CronCreate` makes no schedule; the
    call is in the run's record with the replaced output), a fired run refused, a replayed call
    making nothing twice, `RemoteTrigger`, `CronDelete` and `ScheduleWakeup` not offered;
  - against the **bundled CLI** and a scripted Messages endpoint (the probe's method;
    `probe/model.py` and `probe/drive.py` are the starting point), because the seam is the CLI's
    behaviour and a fake cannot prove it: the model reads the server's id in the CLI's sentence;
    `CronList` reads the server's rows; the CLI's own copy is inert (`Stop` hook's
    `session_crons`); a resumed session lists no job of its own; nothing fires in a session kept
    idle past a minute boundary. This test is the canary for every CLI version in the
    supported range (`compat.py:32`).
- Replace `tests/integration/test_claude.py:693` and
  `tests/integration/test_claude_background_session.py:2286` (which assert the four tools are
  disallowed) with the assertions above.

### Step 5. Firing and delivery (tests only; the code path exists)

Write these against `LocalRuns` and, where it says so, agent-runs' contract suite:

- a schedule set through `CronCreate` fires as a run of the same agent, user and thread, its
  prompt the cron's prompt verbatim;
- **another replica**: two harnesses on one run store; the one that did not set the schedule
  claims the fire, with no live session, and continues the conversation's session;
- **delivery**: the reply reaches an open AG-UI thread stream, and a closed chat finds it in the
  conversation when opened; a run that answers `NO_REPORT` reaches neither;
- **a missed fire**: the workers are down past two ticks; one run on recovery;
- **a restart**: the process that set the schedule is gone; the schedule fires;
- **once**: `recurring: false` is deleted after its run;
- **governance in the fired run**: a write asks in the chat; three failures pause the schedule
  and say so;
- **tenants**: a `CronList` in tenant A never shows tenant B's schedule; `cancel_schedule` with
  B's id answers "No schedule ... of yours".

### Step 6. The record

- `docs/runs.md` "Schedules users set in chat" (`:408-478`): the table of tools per framework,
  and what `CronCreate` does not carry (section 3.3).
- `/var/tmp/cons/inventory/claude.md` rows 196-199, 211 and 383; `kimi.md` row 57 (the reason
  recorded there, "the process does not outlive the attempt", becomes "no seam", section 2.2).
- Kimi, OpenAI Agents, LangGraph and Deep Agents need no code. Their existing tests
  (`tests/integration/test_acp.py:466`, `tests/integration/test_kimi.py:113`) stay.

### What a real-model run must confirm later

1. Asked "every Monday at 9 check stock", Claude calls `CronCreate` (through `ToolSearch` on
   2.1.325, where the tool is deferred) with a cron in the user's zone, and does not tell the
   user about a 7-day limit or a session-only job.
2. Asked to cancel or move it, Claude calls `cancel_schedule` / `update_schedule` with the id it
   was given, and does not look for `CronDelete`.
3. "What are my schedules" calls `CronList` and reads the server's rows.
4. A fired prompt beginning with `/` behaves as the user would expect.
5. The `/loop 2h <prompt>` skill ends in one `CronCreate`; `/loop 5m` is refused in a sentence
   the model passes on.
6. Kimi, told its cron is disabled, uses `mcp__trellis__schedule_task` (today's behaviour,
   already covered by `tests/integration/test_kimi.py:113` with a planned model only).

---

## 7. Risks and open questions for the architect

**D1. Take (A) for Claude's `CronCreate` and `CronList`, or keep (B) for all of Claude.**
(A) gives the model its own tools and loses four set-time behaviours (section 3.3). It also
makes Claude's schedule vocabulary mixed: the library's words for create and list, the
harness's for change and cancel, with one corrective line after each create. (B) is today,
unchanged, with every harness behaviour. The seam is verified; the choice is about product
behaviour, not feasibility.

**D2. The 7-day expiry.** The library's tool text tells the model recurring jobs expire after 7
days. The design keeps harness behaviour (no expiry) and corrects the model in the result's
context. The alternative is to honour the library's rule in `schedules.ended` for schedules set
through `CronCreate`. That is a product choice; neither needs a contract change.

**D3. Conditions through `CronCreate`.** If "tell me only when X" must keep its once-only
reporting on Claude, a convention inside the free-text `prompt` (a last line `Report only if:
<condition>`, read by `set_by_library` into the schedule's condition) restores it with the
existing `_input`. Not in the plan; a small addition to step 2 if wanted.

**D4. `ScheduleWakeup` and the dynamic `/loop`.** It stays disallowed. Allowing it means a fired
run re-arming itself every 1 to 60 minutes: against the rule that a fired run sets no
schedules, and under agent-runs' hourly floor. It also passes sentinel prompts
(`<<autonomous-loop-dynamic>>`) that only the CLI expands.

**Risks.**

1. **The seam rests on CLI behaviour that is not all documented.** `updatedToolOutput` is
   documented in the SDK's types. That a job with a replaced output is not restored on resume,
   and that `validateInput` runs before hooks, are observed (two versions), not promised. The
   canary test of step 4 runs against every supported CLI; a version that changes either must
   fail the suite, not production.
2. **If a later CLI rejects the replaced object** (a new required output field), the model reads
   the CLI's original sentence with the inert job's id and the wrong cadence. The
   `additionalContext` line still names the server's id. The canary catches it.
3. **Inference, not verified:** which recorded field stops the restore (`durable: true` or the
   id). The design sends both.
4. **Inference, not verified:** croniter and the CLI agree on every cron the CLI accepts
   (day-of-month and day-of-week both set). A table test in step 2 should compare next-fire
   times for the docs' examples.
5. **The probes ran inside a hosted Claude Code session**, whose environment the probed CLI
   inherited (the 2.1.283 probe reused the outer session's id). Tool deferral and feature gates
   may differ in CI; the step 4 canary re-proves the seam there.
6. **Timezone.** The model is told the user's zone in the turn's context; the CLI's own notion
   of "local" is the server process's zone. A real-model run must show the cron is written in
   the user's zone. Setting `TZ` in the CLI's environment per conversation is a possible
   hardening, not in the plan.
7. **Kimi's built-ins stay visible and disabled** (today's behaviour). Hiding them needs
   `[tools] disabled` in a `config.toml` the harness would have to write; unsandboxed runs link
   the user's own file (`H:adapters/acp.py:1412`). Not in the plan.
8. **`RemoteTrigger`** is not disallowed today and is undeclared in `BUILTIN_SIDE_EFFECTS`
   (inventory `claude.md` row 211). Step 4 disallows it whatever D1 decides.

## 8. Method notes

- The project instructions ask for the gstack `/browse` skill for web pages. Its browser could
  not be installed here (the Playwright download host is blocked), so the one docs page
  (https://code.claude.com/docs/en/scheduled-tasks) was read with WebFetch, as the task brief
  allowed.
- Probe scripts and logs: `/var/tmp/cons/inventory-tmp/schedules/probe/` (`model.py`,
  `drive.py`, `show.py`, `test_kimi_cron.py`, `run*.jsonl`, `m*.log`). CLI extracts:
  `/var/tmp/cons/inventory-tmp/schedules/*.txt`. One process ran at a time.
- The harness, agent-runs and agent-contracts worktrees were read only; `git status` in the
  harness worktree was clean after the Kimi probe.
