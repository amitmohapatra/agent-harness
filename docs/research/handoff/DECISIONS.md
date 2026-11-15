# Decisions needed: places where a fix or the harness changes a library's behaviour

Your rule: libraries must work as they do on their own. Each item below is a place where
the harness does something different from the library, or where fixing a problem would.
Reply "accept all" or give the IDs you want changed.

Legend: **Rec** = my recommendation.

## Decided (user, 2026-11-14)
- **A1** Claude `verbatim_prompts`: turn on.
- **A2** Claude home: isolate per tenant+user, auto-memory off.
- **F2** Kimi: upgrade to 2.6.2 and handle the swarm-mode, plan-mode and permission-rule changes.
- **G1** Memory OpenAPI: add 504.
- General rule: library features work as they do on their own; harness middleware is opt-in; every
  collision resolved as (A) library works through harness, (B) harness replaces, or (C) adapter;
  prefer library code over our own; ReAct gets every capability the libraries have.
- Applied as "follow the library": B1–B5, C2–C5, D1(b), E1. Kept: C1, C6–C9, E2–E5. F1 and F3: yes.
- ReAct plan mode: same default as Claude and Kimi (both: entered on request) — being verified from source and built from LangChain approval + read-only rule.
- ReAct rewind / time travel: same default as LangGraph (keeps checkpoint history) — Claude SDK file checkpoints are opt-in.
- Handoffs: add `langgraph-swarm`.
- Cross-server interrupt: change cancel semantics so the holder stops first (approved contract change).
- Scheduling: investigated (design: /var/tmp/cons/inventory/DESIGN_library_schedules.md). Only Claude has a seam, for create and list only, on undocumented CLI behaviour, losing four harness schedule features; Kimi, OpenAI, LangGraph, Deep Agents have none. **User decision: keep the harness schedule tools for every library, as today.** Also block Claude's `RemoteTrigger` (cloud, outside governance).
- No public argument / contract changes unless the current one cannot support the need (cancel across servers: only if no existing route; memory 504 doc: yes; `step_limit` default: restored).
- Code standard (ARCHITECT_BRIEF.md): enterprise framework; no unused code; no duplication; SOLID; existing design patterns.
- Open: A3 (Kimi telemetry / auto-update env) — recommendation stands, applied unless you say otherwise.

## A. Security fixes that would change a library default

| ID | Library | What happens today | Option | Rec |
|---|---|---|---|---|
| A1 | Claude | A user's message containing `@/path` or `/command` makes Claude Code read that file or run that command, outside our approvals and records | Turn on the SDK's `verbatim_prompts` so end-user text is plain text (Claude Code's own option for this) | **Turn it on.** On a server, chat text comes from end users, not a developer at a terminal |
| A2 | Claude | All tenants and users on a worker share the machine's `~/.claude` settings, CLAUDE.md and Claude's auto-memory | Give each tenant+user its own Claude config folder; auto-memory off (it lives on one worker's disk, so it is lost on another server; our memory service is the memory) | **Isolate per tenant+user, auto-memory off** |
| A3 | Kimi | Kimi telemetry on; Kimi auto-updates itself | Set `KIMI_DISABLE_TELEMETRY` and `KIMI_CODE_NO_AUTO_UPDATE` on servers | **Set both.** Auto-update can move Kimi past the tested version mid-deployment; telemetry sends usage to Moonshot |
| A4 | OpenAI | SDK tracing sends tool inputs/outputs to OpenAI by default | Keep the SDK default, but warn loudly with the switch to turn it off | **Warn, don't change** (being done) |

## B. Approvals: follow what each library itself asks

| ID | Library | Today | Rec |
|---|---|---|---|
| B1 | Claude | Leaving plan mode is approved automatically | **Ask the person**, as Claude Code does |
| B2 | Claude | WebFetch, WebSearch, Skill run without asking where Claude Code would ask | **Ask, like Claude Code** (the catalog can allow them) |
| B3 | Kimi | Plans approved by nobody, first option picked | **Ask the person with all options** (Kimi requires review; being fixed) |
| B4 | Kimi | "Approve for this session" never chosen, so Kimi asks again each call | **Map our "allow for this chat" to Kimi's option** (one question) |
| B5 | OpenAI | "Allow for this chat" does not stick for the SDK's own `needs_approval` tools | **Make it stick** (being fixed) |

## C. Things the harness adds on every library

| ID | What | Libraries | Rec |
|---|---|---|---|
| C1 | Today's date appended to the system prompt | all | **Keep, documented, can be turned off with `without=`** (models need the date for schedules and "today") |
| C2 | Two identical calls in the same model step run only once | LangChain, Deep Agents, OpenAI | **Stop doing this for the library's own tools** (the library would run both); writes are already protected from double runs by idempotency keys |
| C3 | Large tool results (over 80k characters) and tables rewritten with a note, for the library's own tools too | LangChain, OpenAI | **Only where the harness runs the loop** (ReAct, `middleware()`); leave a plain `create_agent` / StateGraph / OpenAI agent's own tool results as the library returns them |
| C4 | Tools re-sorted by name | OpenAI | **Stop; keep your order** |
| C5 | When a tool name doesn't exist, the model is always told, even if you set `raise_error` | OpenAI | **Respect your explicit setting**; keep the friendly default only when you set nothing |
| C6 | Read tools retried 3 times when the library itself doesn't retry | OpenAI, LangChain | **Keep** (libraries' own retries now win when they retry) |
| C7 | A tool that declares no risk asks a person | all | **Keep** (your earlier decision, like Claude/ChatGPT) |
| C8 | Memory context, memory tools, harness tools, a memory-backed store/session added | all | **Keep** (that's the harness's job) |
| C9 | Graphs without their own checkpointer get one that keeps only the latest checkpoint (no time travel) | LangGraph, ReAct | **Keep, documented**: give your own checkpointer for time travel |

## D. What happens at the turn limit

| ID | Today | Option | Rec |
|---|---|---|---|
| D1 | Claude and OpenAI: hitting the turn limit makes one extra paid model call to get an answer, instead of the library's own "max turns" error | (a) keep the extra answer call, (b) return what the library produced as PARTIAL, no extra call | **(b)** — follows the libraries and costs nothing extra; the run is marked PARTIAL so the app can offer "continue" |

## E. Claude-specific harness changes

| ID | Today | Rec |
|---|---|---|
| E1 | A line about the outputs folder is appended to every system prompt; a `None` system prompt becomes a custom one | **Keep the outputs line only when the run has file delivery; leave `None` as Claude Code's default prompt** |
| E2 | A failed resume reruns the whole prompt | **Keep** (the journal stops tools running twice) |
| E3 | A run silent for 600 s is ended | **Keep** (stuck-process protection) |
| E4 | Cron/ScheduleWakeup turned off (harness schedules instead) | **Keep** (Claude's crons would outlive the run untracked) |
| E5 | MCP output cap raised, reader buffer raised to 100 MiB | **Keep** (needed for large results) |

## F. Versions

| ID | What | Rec |
|---|---|---|
| F1 | Claude Agent SDK 0.2.160 → 0.2.182 (CLI 2.1.325): adds "block the call if a hook fails", typed hook inputs, sandbox fix | **Upgrade** |
| F2 | Kimi Code 2.1.1 → 2.6.2: AgentSwarm only in a swarm mode ACP can't turn on; plan mode user-only; config permission rules would let team allow-rules skip our governance | **Stay on 2.1.x for now**; upgrade after we handle those three changes |
| F3 | OpenAI: use the Responses API through the gateway where it supports it (the SDK's own default) instead of forcing Chat Completions | **Yes** (being done; refuse clearly where the gateway can't) |

## G. Contract

| ID | What | Rec |
|---|---|---|
| G1 | Memory service answers 504 under load, but its published API doesn't list 504 | **Add 504 to the memory OpenAPI** (documents what already happens) |

## OpenAI Agents: the SDK's own `needs_approval` gate (architect, 2026-11-14, user: "use the library where possible, no duplicates in approval, do the best")
- **Keep the library's gate as the one pause mechanism.** Both kinds of approval already pause
  and resume through the SDK's own `needs_approval` / `RunState`: a call governance asks about
  is held at that gate with governance's question (`convert.openai_agents.held`); a tool with
  the user's own gate keeps it, unchanged.
- **One question per call holds today** (verified): a tool with its own gate that the catalog
  also asks about is asked once, at the user's gate
  (`test_your_own_gate_on_a_tool_the_catalog_asks_about_is_the_one_asked`); the SDK approval
  is counted by governance once (`Journal.approved`, `approved_in_framework`); an approval
  "for the conversation" sticks at the user's gate too (`bridge.unasked`).
- **Do NOT make the user's own gate carry governance's choices.** That would change how the
  library's gate behaves for the user's own tool and refuse the documented `answer` decision
  (docs/interrupts.md: approve / reject / answer / edit mapped onto `RunState.approve` and
  `RunState.reject(rejection_message=)`, the only two things the SDK takes).
- **Cleanup phase only (no behaviour change):** the two code paths that build and resume an
  SDK approval pause share one function, parameterised by who asks (governance's choices, or
  the five decisions for the user's own gate). Tests and the docs table stay as they are.

## The owner's working orders of the round (2026-11-14; from the retired job notes)

- Every bug found and every planned item is coded with its tests, except the performance work
  and the cleanup refactor (sections 7 and 8 of `REMAINING.md`); nothing is left open without
  a written reason. Priority order: P0, P1, the review items (SEC, ARCH, PERF), then P2, P3.
- No public argument, signature, endpoint or schema change unless the task says it is
  approved; no new setting; no refactor beyond the item.
- Test first: the test fails on the base for the expected reason, then passes; never skip,
  xfail or loosen a test to pass. Commit and push after every item, so a branch is always safe
  to merge at its head (a `WIP:` commit is finished or dropped before moving on).
- Shared machine: one pytest process per job; PostgreSQL only on 127.0.0.1:55957 in databases
  the job creates and drops (never `live57_*`, never port 55991); one memory-service process at
  a time (the `/var/tmp/cons/memstack.lockdir` lock). The gates are in `GATES_NOTE.txt`.

## The five owner decisions (2026-11-15, owner: "you decide what is best as we are doing it for an enterprise company; a sandbox may or may not be available")
- **Rewinding an ended run: yes, opt-in per agent.** agent-runs keeps an ended run's checkpoint
  only when asked (an additive flag on `finish`; a cancel takes the same final save), kept for
  the run-artifact retention like any artifact. Default off: keeping every ended run's state
  would multiply storage and the conversation data held. **Status: to build** (agent-runs, then
  `Agent.rewind` / `Agent.checkpoints` on ended runs).
- **Who rewinds, and who reads checkpoints: the run's own user only.** `Agent.checkpoints` now
  takes `reader=` and refuses anyone but the run's user, as `Agent.rewind` refuses any
  `reviewer` but them: a run's checkpoints hold its conversation, which a colleague of the same
  tenant must not read. Every rewind is on record (`metadata["rewound"]`: who, what, when). An
  admin path waits for a role model in the harness. **Status: done.**
- **An approval at a library's own gate after a rewind: it counts.** A rewound run's journal
  starts without the earlier run's approvals (`Journal.rewound`), so the only gate approval
  that can count is one a person gives in the rewound run itself — a fresh, explicit approval
  of the repeat; asking again through governance would ask the person the same question twice.
  Nothing allowed before the rewind carries over (`Journal.repeats` still asks). **Status: as is.**
- **Rolling upgrades: unknown journal fields are kept, never refused or dropped.** `Journal`,
  `Pending` and `Made` take `extra="allow"`: a server reads a journal a newer one wrote and saves
  its fields back as they were, so two versions share one run store. Ignoring them instead would
  let an older server drop them on its next save. This upgrade itself (from code that forbids
  unknown fields) needs a full stop or a drain of the old servers. **Status: done.**
- **Sandbox for Claude and Kimi: no default.** A sandbox is not always available. Configured,
  Claude and Kimi use it; not configured, they run in the run's own folder with governance
  asking before writes and shell commands. A `require_sandbox` setting (opt-in) refuses to start
  them without one, for regulated tenants: fail closed where the tenant asks for it.
  **Status: `require_sandbox` to build.**
