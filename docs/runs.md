# Runs, workers, schedules

This page is Way 1, wrapped: the harness records every run of an agent it wraps, and gives
you the inbox, schedules and workers. A team that keeps its own framework uses agent-runs
directly, with `trellis.runs.RunsClient` and `trellis.runs.Worker`:
[blocks/runs.md](blocks/runs.md) (Way 2).

Every run has a record (contracts `RunRecord`): in agent-runs when `RUNS_URL` is set
(`h.runs` is a `trellis.runs.RunsClient`), in process otherwise (`h.runs` is a `LocalRuns`:
same behaviour, nothing survives a restart). Both are the harness's `RunStore`
(`trellis/harness/runs.py`): the part of `RunsClient` the harness calls, with the same
signatures.

| Call | Record |
|---|---|
| `agent.run` / `agent.stream` | `RUNNING` (recorded in process), then `PAUSED` or an ending |
| `agent.start` | `QUEUED`; `RunHandle.result()` waits for a pause or an ending |
| `agent.resume` | the next attempt: `RUNNING` for an in-process run, `QUEUED` again for one that came from the queue; `CANCELLED` on cancel |
| `agent.cancel(run_id, reason=)` (a handle's: `agent.cancel(handle.id)`) | `CANCELLED`: at once when queued or paused; a run running here stops now — its attempt first, so its log in agent-runs ends with the call cut short and `RUN_FINISHED`, then agent-runs' cancel ends it, keeping the reason and who asked; one on a worker elsewhere is stopped by that worker at its next heartbeat |
| `agent.schedule(cron, user=, input=, tz=, tenant=, ...)` | a `Schedule`; each fire queues a run for `user`, with the run options `start` takes (below) |

`run`, `stream` and `start` take `run_timeout=` (the most working time, in seconds, across every
attempt — pauses and the queue not counted) and `deadline=` (when the run must have ended):
past either the run ends `TIMEOUT`, never retried. They are the record's `timeout_seconds` and
`deadline`, and agent-runs keeps the time worked (`worked_seconds`) across attempts and crashes;
`h.wrap(..., version=)` (else `service.version` of `OTEL_RESOURCE_ATTRIBUTES`) is its
`agent_version`
([reliability.md](reliability.md#run-time-limit-and-deadline)).

`start` returns a `RunHandle`: `id`, `await handle.status()` (the `RunRecord`; a run the
store does not have raises `ConfigurationError`) and `await handle.result(wait=None)`, which
reads the run every 0.5 s until it pauses or ends and returns a `Result` (`asyncio.timeout`
raises `TimeoutError` past `wait`). A queued run's input must be JSON (`start` refuses
anything else); a run started in process records its input as JSON where it can and as text
where it cannot, and so does its answer.

Every framework's runs are the same records, queued, claimed, paused and resumed the same way
([examples/06_scenarios/runs_every_framework.py](../examples/06_scenarios/runs_every_framework.py):
an agent of each queued, one worker for all, a schedule fired, a queued run cancelled).

## Resuming a paused run

A run that waits for a person — an approval, an `ask`, a framework's own pause — is `PAUSED`
with its question (an `Interrupt`, in the inbox). `agent.resume(interrupt_id, decision,
answer=, reviewer=, comment=, remember=)` answers it from any process: a run that was started
in process (`run`, `stream`) continues in the caller, now; one that came from the queue
(`start`, a schedule) is `QUEUED` again and any worker continues it. Either way the next attempt
reads the run's journal, so nothing asked or done before the pause happens again, and the
approved call runs once ([interrupts.md](interrupts.md)). After a crash nothing needs to be
called: the run's lease lapses and another worker continues it from its last progress
checkpoint ([below](#workers), [reliability.md](reliability.md#crash-behaviour-end-to-end)).

## Spending: a budget per run

`h.wrap(budget=Budget(...))` caps every run of the agent, `agent.run(budget=)` (and `stream`,
`start`, `schedule`) one run in its place: tokens, dollars as the gateway reports them, harness
tool calls. The budget is kept with the run's record, so a resumed run and a worker that
continues it go on from what it already spent; a sub-agent's run counts toward its parent's.
At a limit the run ends `PARTIAL` with `BUDGET_EXCEEDED` ([api.md](api.md#budget)). A tenant's
or a team's budget is Bifrost's.

```python
from trellis import Budget

agent = h.wrap(graph, id="triage", budget=Budget(max_tokens=50_000))
await agent.start(question, user="ada", budget=Budget(max_cost=0.50, max_tool_calls=20))
```

## Workers

```python
await h.worker([agent_a, agent_b]).run()  # until stopped; CPU count runs at a time
```

```bash
python -m trellis.harness.worker app.agents:h                  # every agent the Harness `h` wraps
python -m trellis.harness.worker app.agents:h --concurrency 8
```

The claim loop is agent-runs' SDK's, `trellis.runs.Worker`; the harness's worker
(`trellis.harness.worker`) runs it with your wrapped agents: each claimed run is its agent's
next attempt, and the harness's background writes start before the first claim and drain when
the loop ends. The same loop with a handler of your own: [blocks/runs.md](blocks/runs.md#workers).

**Your own worker or scheduler.** `await agent.execute(job)` is the claimed run's next attempt
— fresh, continued after an answer, or after a worker died — with its journal, governance,
memory and time limit: the one thing a worker does with a wrapped agent's run. Any loop can do
it: `trellis.runs.Worker(runs, agent.execute, [agent.id])`, or your own around
`runs.claim(worker_id, [agent.id])` (`job` is a `trellis.runs.Job`: the claimed record, the
worker's id, the lease's length). A job of another agent is refused (`ConfigurationError`).
[examples/05_features/own_scheduler.py](../examples/05_features/own_scheduler.py) runs one with a loop of its own;
[examples/05_features/priority_and_concurrency.py](../examples/05_features/priority_and_concurrency.py)
shows the queue order (`priority`, `concurrency_key`).

`h.worker(agents, *, concurrency=None)` needs at least one agent. `concurrency` — runs executed
at once — defaults to the machine's CPU count between 1 and 8. `await worker.run()` claims and executes until stopped: an idle worker asks again after
0.5 s, doubling the pause while the queue stays empty up to 10 s (half of it random, so a fleet
of idle workers does not ask in step), and asks at once again after it got work.
`await worker.run_once()` claims one run and executes it to its end or pause, and returns
`False` when nothing was queued. A claim that fails is logged and counts as no work; a run that
breaks the harness itself is logged and the worker goes on.

**Stopping.** `worker.stop()` — what `await worker.serve()` and so
`python -m trellis.harness.worker` call on `SIGTERM` or `SIGINT` — stops claiming and lets the
runs the worker holds finish, for up to 25 s (`trellis.runs`' `GRACE_SECONDS`). A run still
going then is *released* (cancelled with `trellis.runs.RELEASED`): stopped without writing
anything, and handed back to agent-runs (`POST /v1/runs/{id}/release`), which queues it at once
as its next attempt for another worker — no waiting for the lease to lapse, and no lapse counted
(the journal replays what its last checkpoint holds — its progress, below: every tool
call with side effects it completed). A second signal releases the runs at once. Then the
memory write queue drains (at most 10 s; what is left is spooled or counted lost —
[memory.md](memory.md)) and the process exits `0`. Give the container at least 40 s to stop
(e.g. a termination grace period of 45 s). Cancelling `worker.run()` instead stops the runs
it holds without writing anything: their leases lapse and agent-runs queues them again. A
worker's run ends `CANCELLED` only when someone asked (`agent.cancel`: its heartbeat says
`cancel_requested`), and `TIMEOUT` when its working time is used up. The attempt writes that,
as any run of it out of time does: its error (`run_timeout`, "the run worked past its time
limit of …"), its tool call cut short and its `RUN_FINISHED` (outcome `timeout`) on its events
— also when the worker's own clock for the working time, armed at the claim, stops it first.

A worker runs any target. Build it the same way in every worker process (at import, in the
module the worker loads); a LangGraph graph's own `interrupt()` (or HITL middleware) pause needs a
checkpointer every worker can reach — a harness approval or `ask` resumed by another worker is
answered from the journal even with an `InMemorySaver`
([langgraph.md](frameworks/langgraph.md#approvals-and-pauses)).

A worker claims a queued run under a 60 s lease and heartbeats it every 20 s (a failed
heartbeat is logged and retried at the next beat). It names itself
on the pause, the finish and an artifact upload, so a worker whose lease lapsed cannot write
over a run another worker has since claimed; a heartbeat refused (`409`) stops the run without
writing, and one that answers `cancel_requested` (someone cancelled the run) stops it and ends
it `CANCELLED`. Each lease says the working time the run has left (`remaining_seconds`: its
`timeout` or agent-runs' maximum, the lesser, less what it worked), and the attempt stops then,
`TIMEOUT`. A lapsed lease sends the run back to the queue as its next attempt (after 5 s,
doubling per lapse; the fifth ends it `ERROR`); a run its worker ends `ERROR` with an error that
may pass is queued again too, up to 3 times, after 10 s, 20 s, 40 s. A paused run
carries its journal as the run's checkpoint, so the worker that claims it after a resume — any
worker — repeats no question and no tool call made before the pause.

**Progress checkpoints: a worker that dies repeats no side effect.** While a worker runs it,
the run also saves its journal — the tool calls completed and their outputs, the answers it
was given, and the graph checkpoint of a `ReAct` or `middleware()` graph — as the run's checkpoint on a heartbeat
(`POST /v1/runs/{id}/heartbeat {worker_id, lease_seconds, checkpoint}`, which only the lease
holder may send): at once after every completed call that does more than read (a `write` or
`irreversible` tool, or one under a catalog rule), and after reads and model steps at most every
20 s (`PROGRESS_SECONDS`). When the worker dies — killed, out of memory, its machine gone — the
lease lapses, agent-runs queues the run again with that checkpoint, and the next attempt
replays the recorded calls instead of running them again: a payment made before the crash is
not made twice. A call that does more than read is also saved as *started* before it runs, so
what the attempt was doing *during* the crash is known: such a call is not run again blind —
the model reads that it was interrupted and may or may not have taken effect, and checks; a
tool that is idempotent runs again with the same idempotency key
([reliability.md](reliability.md#unknown-outcomes)). A read that was in flight simply runs
again.
What the run has spent (`RUN_FINISHED`'s `spent`) is saved as each model reply is accounted,
so the replies a dead worker was paid for since its last save stay counted: a heartbeat whose
checkpoint adds one small segment, the spending of the run and its sub-agents' runs, to the
journal's segments — the next spending save replaces it, the next progress save adds to it,
and the checkpoint stays segments alone; one save under way at a time, the replies accounted
meanwhile sent with the next (none when a progress save carried them). A call that only reads
waits, before it runs, for the one save that carries the replies accounted before it — not
for the saves of replies accounted while it waits, not at all when a save already carried
them, and not past a save that failed. A spending save that fails is a `warning` once in the
run (the next progress save carries the spending); one that finds the lease lost stops the
run at its next model call or read. Each spending save uploads one content-addressed segment:
a run writes one per model reply at most (none when the spending is already saved, or back at
what the last progress save holds), and one a later save replaced stays in agent-runs'
payload store until its sweep deletes it, as it deletes any object no record refers to, 24 h
after it was written. The attempt's end starts no
more spending saves: its pause or ending — or its worker handing the run back — is written
only once the spending save under way has its answer, so a heartbeat already sent cannot land
after it (after the same worker claimed the run again, it would put back the earlier
attempt's checkpoint). A model call still under way when the worker died is not counted: the
harness never saw its reply, though the provider may have charged it; nor is a reply a
framework reports only later (when its turn or query ends) before it does.
A save that fails is a `warning` event and a log line — the run goes on, and the next save
tries again; a save refused with `409` (the lease is gone) stops the run without writing
anything more. Runs started in process (`run`, `stream`) save no progress: nobody resumes them
after their process died.

**The journal is saved in segments.** Each save uploads only what changed since the last one
agent-runs took, as one payload object (zstd JSON, `POST /v1/runs/{id}/payloads`), and nothing
when nothing changed; the checkpoint lists the segments, `{"trellis_segments": [ref, ...]}`.
The attempt that continues the run (after a crash, or after a resume in any process) reads
them all at once and applies them in order; every 16 saves one segment holds the whole journal
again, so a resume reads at most 16. A save's size follows what the step added, not the run's
length, and the run's database row holds references only. Nothing to configure. A resume
reads the journal before it answers the run, so a journal that cannot be read leaves the run
waiting. A checkpoint that lists no segments is refused: segments are the only format. With
no `RUNS_URL` the store is this process's memory, and the
segments are files in a temporary folder (`LocalRuns(payload_root=)`), in the same format:
it bounds nothing, and the same reference is kept there with the run. A folder the store made
itself goes with it (closed, collected, or at the process's exit); one you pass is yours.

## Continuing where a run stopped

A run continues after a pause (`agent.resume`) or after its worker died (the lease lapses)
from what its journal and its framework kept — never from scratch where something kept more:

| Target | After a pause | After its worker died | What a long run keeps |
|---|---|---|---|
| `ReAct`; a `create_agent` or Deep Agents graph built with `middleware()` | in place, from its checkpoint in the run | from the latest step saved: the model is not asked again for the steps it holds | the latest checkpoint and the step under way's records |
| a graph with a checkpointer of its own | in place where the checkpointer holds the pause, else a harness pause from its input | a durable checkpointer: from the run's latest checkpoint; one the new process does not have (`InMemorySaver`): from its input | every call's record, and the checkpointer's own |
| a graph with none (no `middleware()`, a hand-built `StateGraph`, `checkpointer=False`), a function | from its input, the journal answering | from its input, the journal replaying | every call's record |
| OpenAI Agents | its own `RunState` (a `needs_approval` tool), else from its input | from its input, the journal replaying | every call's record |
| Claude Agent SDK, Kimi (ACP) | its own session resumed (Claude's transcript is in the journal; Kimi's on its machine) | the same, from the session as saved | the session's transcript (Claude), every call's record |

Wherever it continues from, every harness and framework tool call that finished replays from
the journal instead of running again, one that does more than read and was under way is not
run blind ([unknown](reliability.md#unknown-outcomes)), and nothing asked is asked again.

**A graph's checkpoint, in the run.** A `create_agent` or `create_deep_agent` graph built with
`middleware()` keeps its checkpoint in the run as `ReAct` does: compiled without a checkpointer,
`h.wrap` runs a copy that has `RunCheckpointer()` (the graph object is left as it is); there is
nothing to give. A checkpointer of its own (`InMemorySaver`, `AsyncPostgresSaver`) stays the
graph's, and so does `checkpointer=False` (no checkpoint: every attempt runs from its input).
With a checkpointer of its own, the run's checkpoints carry its id in their metadata
(`trellis_run_id`), and the attempt after a crash continues from the run's latest one — not a
new turn with the question again — when the checkpointer has it.

**A later step's call is a new call.** A graph continuing from a checkpoint does not make its
earlier steps' calls again, so the journal keeps each call's record with the step it was made in
(the checkpoint the step started from): after a pause or a crash, a later step's call identical
to an earlier one (`add_item("apple")` again, the same refund asked once more) runs, and is
approved, as a new call — only a call of the step under way replays. As the graph's checkpoint
in the run moves past a step, that step's records are dropped: the journal of a `ReAct` or
`middleware()` run is its checkpoints (each about 2 KB, and what each step wrote, once:
[react.md](frameworks/react.md#checkpoint-history-langgraphs-time-travel)) and one step's
records, however many steps it takes.
Each output on the run's [trajectory](evaluation.md) is kept as the event stream previews it
(2000 characters). The graph's state itself is the framework's: it grows with the conversation
unless the graph summarizes it (Deep Agents' summarization).

**A damaged checkpoint.** A graph checkpoint the journal holds that cannot be read back
(damaged, or written with types this release cannot load), or one missing the graph's own,
ends the run `ERROR` with `CHECKPOINT_UNREADABLE`, not retried: starting the graph over would
make again the calls its checkpoint had moved past. Nothing is run again; start a new run.

**A new version between the crash and the resume.** The run continues on the version that
claims it, with a warning naming both (`agent_version`): another prompt, model, tool or step
limit applies from the next step, the checkpoint as it was saved. A run started where its graph
kept no checkpoint (a release before this one, `checkpointer=False`) continues from its input on
one that keeps it, the journal replaying its calls. A change the saved checkpoint no longer fits
(a renamed state field the graph reads) is the graph's to migrate: let the runs under way finish
on the old version before you deploy it.

## Queue order and busy conversations

**What.** `agent.start(..., priority=0, concurrency_key=None)`: among the tenant's queued runs a
higher `priority` (-1000 to 1000) is claimed first, then the oldest; runs sharing a
`concurrency_key` run one at a time (agent-runs' limit, one unless its operator says
otherwise), the others wait `QUEUED`.

**When.** `priority` for an urgent run; `concurrency_key` for anything that must not run twice
at once (one customer's account, one repository).

**Automatic: a second message to a busy conversation waits.** A queued run of a conversation
(`thread=`) gets `concurrency_key="thread:<conversation>"` unless you name one: a second message
sent while the first run works is answered after it, never beside it (both would write the same
thread). A run of a conversation started with `run`, `stream`, `serve_chat` or `serve_a2a` (or
resumed) gets the same key and waits while another run of the same agent and conversation works,
in this process or any other sharing the run store: the conversation's messages are answered
one at a time, in the order they were sent, whichever replica each reaches. A run with no
`thread` is its own conversation.

The order is the conversation's line (`trellis.harness.line`): the run store keeps it, and it is
the one ordering of a conversation's turns, in a process and across them. A conversation's
messages sent to one process are recorded in the order they were sent; agent-runs creates
each run of a key in its place, after the key's runs already running or queued, in the order
the starts commit. A run's place is when it was created, or when its latest answer was recorded
(a run continued after a pause joins the line then, after the turns that arrived while it
waited); agent-runs gives it on the run as `placed_at`, and the harness orders by it without
recomputing it. It takes its turn once no `QUEUED` or `RUNNING` run of its agent and
conversation is placed before it, the line listed by its key (another user's thread with the
same name is never read). A queued run (`start`) holds its place while it waits for a worker, and a
worker claims it only once no run placed before it runs: a stream of messages cannot keep it
waiting. A run that pauses for a person, ends, or whose process dies leaves the line: a process
killed while it holds a turn frees it once its hold lapses (60 s) and agent-runs ends its run
`holder_lost`; one killed after recording a turn and before holding it, 60 s after the start.

A run waits for the run just ahead of it. One of this process: until it leaves the line here.
One elsewhere (another replica's, a worker's): by following that run's events in agent-runs
(its event stream says when the attempt ends or pauses), on a client of its own — one stream a
run ahead, shared by every run waiting behind it in the process, at most 32 at once. Nothing
polls: only where that stream cannot be followed (it dropped, or 32 are open) does a run read
its line again, after a pause that doubles from 0.05 s to 5 s (jittered), and at the latest
every 30 s. A run cancelled while it waits its turn (`agent.cancel`, from any process) takes
none: it ends `CANCELLED` without running.

**On failure.** A priority outside -1000..1000 or a key over 200 characters is refused by the
contract (`ValidationError`). A run waiting for its turn is held like any run this process
executes: it counts its working time while it waits, and a deadline or time limit ends it there.
An agent-runs older than this release sends no `placed_at`: a conversation's turn then fails with
a `ConfigurationError` naming the run, rather than run in a guessed order; upgrade agent-runs
before the harness. During a rolling
deploy, a turn started by a harness older than this release has no `concurrency_key`: it is
not in the line, so a message sent meanwhile to a new replica does not wait for it (and a turn
paused by an old release, continued on a new one, waits for nobody). Let the turns under way
end before the last old replica goes.

## A message to a busy conversation: mode

**What.** `agent.run(..., mode=)` and `agent.stream(..., mode=)` say what a message does when
another run of its conversation works, in this process or another ([ADR 0005](adr/0005-conversations-across-runs.md)):

| `mode` | What happens |
|---|---|
| `"queue"` (default) | it waits for the working run to end, then runs: what a second message always did |
| `"interrupt"` | the working run is cancelled (`agent.cancel`, reason `interrupted by a newer message of the conversation`), wherever it works; the message takes the turn and continues the conversation — the cancelled turn's question is in its history |
| `"steer"` | the message is given to the working run at its framework's next step, and that run's answer is the message's: its run ends `SUCCESS` with that answer, streamed under that answer's message id |

Where `"steer"` reaches the working run:

| Target | Steers | Its hook |
|---|---|---|
| `ReAct`, a graph built with `middleware()` (Deep Agents with `HarnessTools` too) | yes | `HarnessTools.before_model`: a user message before the next model call |
| OpenAI Agents SDK | yes | `RunConfig.call_model_input_filter` (after the run's own): a user item in each later model call's input |
| Claude Agent SDK | yes, after a tool call | a `PostToolUse` hook's `additionalContext` (a query that calls no tool takes none) |
| a function, Kimi (ACP), a graph without the harness's middleware | no (`Adapter.steers`) | none: the message waits as `"queue"`, said once per agent |
| any run working in another process | no: nothing carries a message to it | none: the message waits as `"queue"`, said once per agent |

A steered message the working run never took (it ended first, or ended without an answer)
runs as its own turn. A message sent with `"interrupt"` while the turn is held in another
process says so in its own run (its metadata names the run it interrupts) and waits in the
line; the process holding that run reads the line each time it renews the run's hold (every
20 s), stops the run — its call under way is cancelled — and ends it `CANCELLED`; only then
does the message take its turn. The two never work side by side, and the message waits at
most 20 s for the turn it interrupts to stop (a process that died: until its hold lapses). A
turn a worker runs (`start`) is not interrupted: the message waits its turn. **When.** `interrupt` for a chat whose user sends a correction; `steer`
for a "also…" while the agent works. **On failure.** `start` takes no `mode`: a queued run waits
by its `concurrency_key`, and a worker's run is not held in process.

## Earlier turns with memory off

A run of a conversation reads its earlier turns from the run store when nothing else keeps
them — memory off (`without={"memory"}`, or no `MEMORY_URL`), and a target with no thread of
its own (no checkpointer, no OpenAI `session`) and no session continued: each earlier run's
question and answer (a cancelled one's question), of the same agent, tenant, user and thread,
the newest 10, oldest first. `ReAct`, a graph and an OpenAI agent get them as their own
messages, a function as `agent.history`, Claude and Kimi in the context's text. With memory on,
memory's window keeps the conversation as before ([memory.md](memory.md)); where that window
comes as messages and lacks the conversation's latest turns (their writes still on their way),
those turns come from the run store after it.

A target that takes messages and fits each request to a known window (`ReAct` with
`context_window=` or a model profile that says it) gets each earlier turn this process ran
whole: its question, its tool calls and their results (one past its tool's age marked out of
date), and its answer — the figures behind an answer, for "why was that?" or "redo it with X",
as memory's window gives them with memory on. They are sized for the request they go into, as
memory's window is (`Agent.conversation_budget`: what the window leaves once the instructions,
the tools, the question, the front end's context and a reply are in): each turn's question
and answer from the newest back while they fit, then each of those turns whole, from the
newest back, while it fits. `ReAct`'s clearing and `FitWindow` fit the turn under way (older
results cleared first, `read_file` reads one back). The transcripts are
kept with the conversation's saved results (`agent.results`, in this process); a turn another
process ran, or one with no window known, is its question and answer.

## A Claude or ACP session across turns

A conversation's turn on the Claude Agent SDK or an ACP agent (Kimi) continues the session its
framework kept for the conversation's last turn in this process (`agent.sessions`, the most
recent 128 conversations): the journal's session path a resume after a pause takes, sent the
turn's own prompt. A turn another process runs starts a new session, with the earlier turns as
above (memory off) or memory's window (memory on).

## Child runs at once, and the task tree

**What.** Every child run the harness starts — a sub-agent's call in any framework, `agent.run`
inside a run — is admitted under one adaptive limit (`trellis.harness.admission`): a parent run
has at most that many of its own children working at once, the rest waiting for a slot. Each
child admits its own children, so nesting never waits on a parent's slots.

**Automatic: the limit adapts (AIMD).** It starts at 4 and stays between 1 and 16. Each child
that ends while the process is healthy raises it by `1/limit` (about one more per round of
children); a sign of pressure halves it, at most once every 2 s:

| Pressure | Read from |
|---|---|
| CPU over 85% of the container's quota | cgroup v2 `cpu.max`/`cpu.stat`, else v1 `cpu`/`cpuacct`, else the load average over the cores |
| Memory over 85% of the container's limit (the working set: inactive page cache left out) | cgroup v2 `memory.max`/`memory.current`, else v1 `memory`, else `/proc/meminfo` |
| The provider slowing down: its pace (seconds per output token, smoothed) over twice its fastest recent pace, and slower than 50 tokens a second | every LangChain model call in a run |
| A `429` from the provider or the gateway | every LangChain model call in a run |

The machine is read at most once a second. The budget and the deadline stay hard ceilings: a
child waits for a slot at most what is left of its parent's time, and its spending counts toward
its parent's budget ([below](#spending-a-budget-per-run)). Kimi Code is told the limit as its
swarm's concurrency (`KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY`, unless the agent's own `env` names
it); a framework that spawns its own helpers (Deep Agents' `task`, an OpenAI agent as a tool,
Claude's sub-agents) keeps its own parallelism.

**The task tree.** Each child is on its parent's stream (`agent.stream`, `agent.events`, AG-UI's
`CUSTOM` events) as it goes, every framework's alike:

| `CUSTOM` event `name` | Data |
|---|---|
| `helper_queued` | waiting for a slot: `run_id`, `agent_id`, `limit` |
| `helper_started` | working: `run_id`, `agent_id` (a framework's own helper: `framework`, `kind`) |
| `helper_finished` | ended: `status` (`success`, `partial`, `error`, `timeout`, `cancelled`, `paused`), `seconds`, `tokens` and `cost` (the child's spending, when the harness counted it), `error` |

A child's spans are in its parent's trace, so Langfuse shows the same tree.

**On failure.** A child that waits for a slot past its parent's time ends `timeout` (for a
sub-agent's call, the call's error). Cancelling the parent cancels its children, those waiting
for a slot included.

## A run's events from anywhere

**What.** With `RUNS_URL`, every attempt appends its events (the `RunEvent`s `stream` yields)
to the run's event log in agent-runs as they happen, so any replica streams any run:
`agent.events(run_id, *, after=0)` yields them — those past position `after`, then each as it
is appended, until the run ends (a paused run's stream stays open for its next attempt) —
wherever the run executes (a worker, another replica). `serve_chat`'s reconnect route reads a
run this process did not serve from there too ([surfaces.md](surfaces.md)).

**Automatic.** Batched (at most 500 an append), in order, one append at a time, in the
background: the run never waits for it except at its end, when its last events (`INTERRUPT`,
`RUN_ERROR`, `RUN_FINISHED`) are appended before the pause or the ending is recorded (the log
takes nothing after) — and whoever watches in process hears `RUN_FINISHED` only once it is
recorded. A worker's appends name it (fenced by its lease). Without `RUNS_URL` nothing is kept:
`agent.events` follows the run's attempts in this process from now on.

**With progress.** A worker's progress save (the heartbeat after a step with side effects)
carries the events not in the log yet, and agent-runs keeps them in the same transaction as the
checkpoint: a worker killed right after a step leaves both the step and the events that show
it, or neither — never a step the next attempt will not repeat that its log never shows. An
agent-runs from before heartbeats took events refuses them once; progress is then saved alone
and the events go as appends.

**On failure.** Best-effort: while agent-runs cannot be reached (a restart), an append is tried
again, in order, for up to a minute; events it refuses or still cannot take are dropped with one
`warning` event (`events_undelivered`, logged, counted `trellis.run_events.undelivered`); the
run goes on. A lost lease stops the log.

## Admission: agent-runs' rate limit

A call agent-runs refuses with `429` (the tenant's request budget) is retried by its SDK after
the `Retry-After` it sends (at most 3 times, each wait at most 30 s). When it still refuses,
`run`, `stream`, `start`, `resume` and `schedule` raise `trellis.harness.agent.Throttled`: a
contracts error (`RUNS_RATE_LIMITED`, category `RATE_LIMIT`, `retryable=True`,
`details["retry_after"]` the seconds asked for), counted (`trellis.runs.rate_limited`). The
harness does not retry it again (one retry layer, the SDK's); `serve_chat` answers `429`
(`RATE_LIMIT`) with `Retry-After`. A tenant over agent-runs' cap of running runs is not refused:
its queued runs wait.

## Schedules

`agent.schedule` is one `runs.schedules.create(spec)` (`POST /v1/schedules`): agent-runs upserts
on `(tenant, agent, on_behalf_of, cadence, sha256 of the canonical input)`, so scheduling the
same thing again (a redeploy) answers the schedule that exists, unchanged. A scheduled run
carries everything a started one can (contracts 0.6.1, ADR 0007): `schedule` takes the same
run options as `start`, and every fired run gets them — agent-runs and `LocalRuns` alike:

| Run option | `agent.start(...)` keeps it in | `agent.schedule(...)` keeps it in | Each fired run gets |
|---|---|---|---|
| `run_timeout=` (else the agent's `h.wrap(run_timeout=)`) | `RunStart.timeout_seconds` | `ScheduleSpec.timeout_seconds` | `timeout_seconds` |
| the agent's `version` | `RunStart.agent_version` | `ScheduleSpec.agent_version` | `agent_version` |
| `without=` | `RunStart.metadata["without"]` | `ScheduleSpec.metadata["without"]` | `metadata["without"]` |
| `framework_options=` (JSON) | `RunStart.metadata["framework_options"]` | `ScheduleSpec.metadata["framework_options"]` | `metadata["framework_options"]` |
| `priority=` | `RunStart.priority` | `ScheduleSpec.priority` | `priority` |
| `concurrency_key=` (`start`: by default `thread:<conversation>`) | `RunStart.concurrency_key` | `ScheduleSpec.concurrency_key` | `concurrency_key` |

A fired run is the one the contracts build for its schedule (`Schedule.run_start`), from
agent-runs and from `LocalRuns` alike: the schedule's `metadata` is copied into the run's
`RunStart.metadata`, the fire's own keys (`schedule_id`, `schedule_name`, `fire_time` in UTC,
`created_by`) laid over it, so a worker reads a scheduled run's choices exactly as a started
run's; its `idempotency_key` is `<schedule_id>@<fire_time in UTC>`; and its `user_id` is
`None` — nobody is present at a fire — while it acts for `on_behalf_of`, which is the run's
user to the harness (`Runtime.user`, memory, files). `priority=` and `concurrency_key=` need trellis-contracts 0.6.1 or later: with an
older one `schedule` refuses them (`ConfigurationError`) rather than drop them. In Way 2 the
same keys work without a harness: `runs.schedules.create(ScheduleSpec(..., priority=,
metadata={"without": [...], "framework_options": {...}}))`, and whatever executes the fired run
(a `trellis.runs.Worker` handler, `agent.execute`) reads them from `job.record`. The cadence is a cron
expression or one of `hourly`, `daily`, `weekly`, `weekdays`, `manual`. Pause and resume a
schedule with `RunsClient`: `await runs.schedules.update(schedule_id,
ScheduleUpdate(enabled=False))` (or `True`). In process, a due schedule fires when a worker asks
for work; in agent-runs its ticker queues the run.

### Schedules users set in chat

"Every Monday at 8 check stock and tell me if it's under 5", then "what are my schedules",
"change it to 9", "pause it", "cancel it": every agent has four built-in tools for this
(`trellis/harness/schedules.py`), on agent-runs' schedules (`RUNS_URL`) or `LocalRuns`'. Nothing
to configure; `without=["schedules"]` turns them off (for the agent or one run). A graph built
with its tools bound takes them when it is built, with the harness's other tools: `await h.harness_tools()`
([frameworks/langgraph.md](frameworks/langgraph.md#the-harnesss-tools-on-a-graph-you-build); `h.schedule_tools()` alone). A run a schedule fired is offered none: a graph's bound ones answer that they are not for it.

| Tool | What it does |
|---|---|
| `schedule_task(task, when=None, condition=None, once=False, report=None, end_at=None, in_minutes=None, choices=None, timezone=None)` | `when` is a 5-field cron the model writes in the user's zone (`0 9 * * *` daily at 9, `0 9,17 * * *` at 9 and 17, `0 9 * * 1-5` weekdays, `0 8 * * 1` Mondays, `0 9 1 * *` the 1st, `0 9 4 11 *` with `once` for one day), checked with `croniter`; `in_minutes` instead is once, that many minutes from now ("in 2 hours"), rounded up to the next whole minute; `once=True` ends the schedule after its run (a run that fails pauses it instead, said in the error, so the user can run it again); `end_at` when it ends, in the user's zone: their last day (`2026-12-31` for "until 31 December": its runs that day still happen, none after; agent-runs keeps it as the instant that day ends) or an exact `YYYY-MM-DDTHH:MM`, a past one said back; `report` for a condition: `until_clears` (default: when it holds, then not again until it has stopped holding), `every_time` (each run it holds) or `once` (the first time it holds, then the schedule ends); `timezone` only when the user names one; `choices` (a vague ask): 2-3 concrete schedules, the best first, the user picks one with a button, or gives any time of their own: the question carries an answer schema the app renders from the user's settings (`WHEN_ANSWER`: how often, the days, the time, a date, when it ends), and an answer that is none of the suggestions, in words or those fields, goes back to the model as the user's own (nothing is made in its place) |
| `list_schedules()` | the user's schedules set in chat with this agent: id, task, when it runs (its next two runs as dates in its zone, `next Mon 24 Nov, 09:00, then Mon 1 Dec, 09:00 (Europe/Berlin)`: never its cron, which stays inside), condition, active or paused; where there is one: when it `ends` (in words in its zone), its last run (in its zone), `failed_runs_in_a_row`, why a paused one is `paused_because`, what its last run is `waiting` for (a person's answer: its next tick starts no other run meanwhile), `fire_error` (why agent-runs could not fire it), and for a condition how it `reports` and whether it was `reported` (why a watch went quiet) |
| `update_schedule(id, ids=None, when=None, task=None, condition=None, enabled=None)` | `enabled=False` pauses, `True` resumes; `ids`: the schedules the user's words fit, when several do: the user picks one |
| `cancel_schedule(id, ids=None)` | deletes it; `id="all"` deletes them all once the user confirms; `ids` as `update_schedule`'s |

- **When, in whose zone.** A schedule is in the user's zone: the one they name, else the one
  their app sends (AG-UI `forwardedProps.timezone`, e.g. the browser's
  `Intl.DateTimeFormat().resolvedOptions().timeZone`: [surfaces.md](surfaces.md)), else UTC.
  `schedule_task`'s description says the day and the hour in that zone ("It is now Tue
  2026-11-03, 14:00-14:59 (Asia/Kolkata, the user's zone)"), so "tomorrow at 9" has a date; the
  hour, not the minute, so the tools' text stays the same for an hour, as a prompt cache keeps
  it. With no zone known the model may ask for it.
- **When it schedules.** On a request for something later or on a repeat, not on a question
  that only mentions a time or a habit ("do we restock every Monday?"); it asks nothing the run
  can work out itself (which figures a summary has; no time said is 9:00, said in the reply).
  A condition is checked once now with the agent's own tools and the user hears the value ("12
  now; I'll tell you when it is 5 or less"); one that holds already is said, and scheduled only
  if the user wants to hear again. Nothing its tools can check ("when a competitor changes
  price") is not scheduled: it says so and offers what its tools can watch. A fired run is
  offered none of the four tools: it does its task (it made schedules of its own when it was).
- **One question for a new schedule.** A schedule with anything open is confirmed in one
  question (`Runtime.ask`: Confirm, recommended, or Cancel, and a form the app renders), every
  open detail in it at once, never asked one after another:
  - what the model assumed rather than the user said (`schedule_task(inferred=)`: time,
    days, zone, condition, end): with no time said the model uses
    `TRELLIS_SCHEDULE_DEFAULT_TIME` and lists `time`, and the form offers it as a time picker;
  - a schedule the model judged like one the user has (`alike=`, the id it read in
    `list_schedules`; no word-overlap rule of code decides, handoff 2.5): update it (the
    default) or keep both;
  - the writes its runs make (`tools=`): allowed in its runs unasked (the default), or asked;
  - what its unattended runs do with a step nobody approved: skip it and say so (the default),
    or wait for the user.

  Confirm takes every default; answered in words, the model reads it and decides. A schedule
  said in full, with nothing new to allow, is made with no question. The very same again
  (task, time, zone, once and the condition's figures) is "Already set". Cancel all, a
  reference that fits several (`ids`) and a vague ask (`choices`) ask with buttons too.
- **Unattended runs never wait for nobody.** A run its schedule fired calls what the schedule
  allowed; a call nobody approved is skipped (`trellis_unapproved: skip`): the run goes on and
  its reply says what was not done and that the user can approve it. A schedule set to wait
  pauses on it instead, and waits a day for the user's answer (`runtime.UNANSWERED`, the
  interrupt's `deadline`): past it the run ends `TIMEOUT`, nobody answered (agent-runs'
  ticker, and `LocalRuns` as it). A schedule made before it was asked waits, as it did. An
  `irreversible` tool asks every call, so it is never run unattended.
- **Whose.** A schedule acts for the run's user, in the key's tenant (`on_behalf_of`); each user
  sees and changes only their own, and only those set in chat (a schedule your code makes with
  `agent.schedule` is not listed).
- **The same chat.** A fired run is the same agent, user and thread: the schedule's metadata
  names the thread (`thread_id`), and agent-runs fires the run into it. Its reply is saved into
  that conversation like any turn (memory's transcript, or this process's history with memory
  off), so it is there when the user opens the chat; an open chat hears it at once on the AG-UI
  thread stream ([surfaces.md](surfaces.md)). It takes the agent's own `run_timeout` and budget,
  and a write it makes goes through the same governance and approvals (an `ask` in the chat).
- **Only when it matters.** With a `condition`, the run is told to reply exactly `NO_REPORT`
  when it does not hold. Such a run is not delivered: its output is `NO_REPORT`, it is not
  written to the conversation, not grounded or judged, and the thread stream leaves it out. A
  condition that still holds is reported once: the runs after stay quiet (`NO_REPORT`) until
  it stops holding (`trellis_reported` in the schedule's metadata).
- **Limits.** `TRELLIS_SCHEDULE_*` ([configuration.md](configuration.md)), the handoff's
  defaults: at most one run an hour (`MIN_INTERVAL`, 3600 s), 5 schedules a user per agent
  (`MAX_PER_USER`), each fired run 10 minutes (`RUN_TIMEOUT`) and 30 tool calls
  (`RUN_MAX_TOOL_CALLS`: the run's `Budget(max_tool_calls=)`; an agent's own lower limits
  stay), paused after 3 failed runs in a row (`PAUSE_AFTER_FAILURES`), asked to be kept after
  90 days unchanged (`EXPIRE_DAYS`), and 09:00 when no time is said (`DEFAULT_TIME`).
  `schedule_task`'s description says the default time and the shortest repeat in force, as
  facts the model decides by. Per user or plan, `Harness(schedule_limits=async (user, tenant)
  -> ScheduleLimits | None)`: a `ScheduleLimits` field (the same names without `schedule_`:
  `min_interval=`, `max_per_user=`, `run_timeout=`, `run_max_tool_calls=`,
  `pause_after_failures=`, `expire_days=`) left `None` is the environment's
  (`schedules.limits(runtime)` resolves them; the shortest repeat is the longer of this and
  agent-runs' own floor, read from its `GET /v1/schedules/policy` at most once a minute per
  tenant). agent-runs' own floor, its per-tenant and
  per-day limits (`RUNS__SCHEDULES__*`), its rate limit and `max_running_per_tenant` still
  apply. A limit reached is a sentence the model tells the user ("You have 5 of 5 schedules:
  cancel one before adding another."; "That runs every 2 minutes, more often than every 1
  hour..."), never a cron or an error code; agent-runs' limits are said with their numbers too
  ("Your organisation has 50 of 50 schedules, the most it allows...", from its
  `SCHEDULE_LIMIT`, `CADENCE_TOO_FREQUENT` and `FIRES_PER_DAY` codes).

- **Unattended, safely.** Nothing to configure. agent-runs fires a schedule once after an
  outage (not once per missed tick), skips a tick while its last run is still queued, running
  or paused (Kubernetes `concurrencyPolicy: Forbid`), and fires each schedule a stable 0 to 59
  seconds after its tick (a hash of its id: "daily at 9" is still 9:00, and a thousand of them
  spread over the minute). After 3 failed runs in a row (`ERROR`, `TIMEOUT`; the setting above) the schedule is
  paused, and that run's error says so: in the thread, the run's record and its `run.finished`
  webhook (`trellis_failures` in the schedule's metadata; a success clears it). A schedule the
  user has not made or changed for 90 days (the setting above) is paused at its next run
  (`paused_because`: it waits for the user to keep it), and that run does its task and ends
  by asking, in its model's words and the user's language (handoff 2.6), whether they still
  need it; it is delivered even when its condition has nothing to report. The user keeps it
  in chat (`update_schedule`, which resumes and renews it: `trellis_touched`) or cancels it.
  A run that asked nothing (a bare `NO_REPORT`) leaves it on, and its next run asks again. A run that
  times out while no worker holds it (agent-runs ends it) is not counted. A fired run is told
  the day and hour in its schedule's zone (`trellis_zone`), never silently UTC.
- **A user who left.** Check them in an `on_run_start` hook: a fired run acts as its
  `on_behalf_of`, and a hook that raises ends the run before any tool is called; cancel the
  user's schedules there (`h.runs.schedules.list(on_behalf_of=)`, then `delete`) or when your
  directory says they left.

  ```python
  class StillHere(Hooks):
      async def on_run_start(self, run: Runtime) -> None:
          if run.scheduled is not None and await user_left(run.tenant, run.user):
              raise PermissionError(f"{run.user} has left: their schedules stop")
  ```

**Push and email** need nothing in Trellis: agent-runs' signed webhook (`run.finished`) reaches
your backend, which sends the notification. Its summary names the run's thread; a scheduled
run carries `schedule_id` in its metadata, and one with nothing to report has the output
`NO_REPORT`:

```python
from trellis.runs import RunsClient, WebhookEvent, parse_delivery, verify_signature
from trellis.runs.webhooks import SIGNATURE_HEADER

# once: await runs.webhooks.create("https://app.example/hooks/runs", [WebhookEvent.FINISHED])


async def hook(request: Request) -> Response:
    body = await request.body()
    if not verify_signature(SECRET, request.headers.get(SIGNATURE_HEADER), body):
        return Response(status_code=401)
    delivery = parse_delivery(body)
    record = await runs.get(delivery.data.run.run_id, tenant=delivery.tenant_id)
    if record and "schedule_id" in record.metadata and record.output != "NO_REPORT":
        await notify(record.on_behalf_of, record.thread_id, record.output)  # push, email
    return Response(status_code=204)
```

### Long-running work

A job that takes a while (a tool declared `long_running`, [tools.md](tools.md#a-tool-that-starts-a-long-job))
is followed up by the harness, not polled by the model (`trellis/harness/follow_ups.py`):

- **Durable and once.** Each check is a one-time schedule of the run's user in its chat
  (`trellis_follow_up` in its metadata), agent-runs' like any: it survives a restart, fires
  once, and holds no worker while it waits. It is not one of the user's schedules: no limit
  counts it.
- **No model while waiting.** A follow-up's run reads the status tool before any model call:
  still under way, the next follow-up is made (2, 5, 15, 30, then every 60 minutes) and the
  run says nothing (`NO_REPORT`).
- **One continuation.** Done, failed, or under way after `follow_ups.GIVE_UP` (a day), one run
  continues the task from what the status said: it reports, or does the task's next step (its
  unattended rule applies), and the chain ends. A user's every-Monday schedule that starts a
  job gets a chain per run; agent-runs starts no new run of it while one waits.

### Telling users: channels

One interface tells a user what happened while they were away, whatever carries it
(`trellis/harness/notifications.py`): a `Notification` — its `kind` (`Kind.RESULT` a fired
run's result, `FAILURE`, `PAUSED` a schedule paused, `APPROVAL` a run waiting for the user's
approval), its `tenant` and `user`, the `thread`, `run_id` or `schedule_id` it is about, a
short `text` of the facts, and a stable `key` — goes to the user's channels with
`await h.notify(notification)`, which returns the names of the channels that took it now.

- **Channels.** `Harness(channels=[...])`: the channels the company connected, set once in
  code; none given is the chat's own (`ChatChannel`), `[]` is nowhere. A channel is any object with a `name` (one per harness) and
  `async deliver(notification, harness)` that returns once it delivered (or cannot tell) and
  raises when it did not (`Undelivered`): email, Slack or Teams plug in here, with nothing of
  schedules or runs to change.
- **Whose channels.** `Harness(user_channels=async (user, tenant, notification) -> names |
  None)`: the names of the channels this user gets for this notification (by its `kind`, too),
  from the company's code, as `schedule_limits=` is — never the environment. No hook, or one
  that says `None`: the chat only (`CHAT`), never every channel. A name the harness has no
  channel of is logged and skipped; a hook that raises is logged and the chat alone is told.
- **At most once per channel.** `notify` gives a channel a tenant's `key` only once in the process: a
  second call (a retry, a resumed run, two at once) sends nothing again. A channel that raised
  is logged and tried again by the next call. The last 10,000 keys are kept (`KEPT`); a channel
  that reaches an outside system passes `key` on as its idempotency key, so another process
  does not send it twice either.
- **The chat.** A notification is about a run of the user's thread (a fired run is the same
  agent, user and thread): that run is the chat's delivery — the thread stream delivers its
  reply, pause or error to an open chat ([surfaces.md](surfaces.md)) and the conversation keeps
  its reply — so `ChatChannel` sends nothing of its own, which would say it twice. It checks the
  run is the user's, in that thread (`Undelivered` otherwise, or with no run).

```python
from trellis import Harness
from trellis.harness.notifications import ChatChannel, Kind, Notification


class Email:
    name = "email"

    async def deliver(self, notification: Notification, harness: Harness) -> None:
        await mailer.send(notification.user, notification.text, idempotency_key=notification.key)


async def routes(user: str, tenant: str, notification: Notification) -> list[str] | None:
    if notification.kind is Kind.APPROVAL:
        return ["chat", "email"]  # an approval waits: email it too
    return None  # the rest: the chat only


h = Harness(channels=[ChatChannel(), Email()], user_channels=routes)
said = Notification(
    kind=Kind.RESULT,
    tenant=tenant,
    user="ada",
    text=report,
    key=key,
    thread="chat-1",
    run_id=run_id,
)
await h.notify(said)
```

## The inbox

`await h.inbox("role:procurement")` — the paused runs waiting on that assignee (or, with no
argument, on anyone in the tenant), newest first, as `trellis.runs.RunSummary` (`run_id`,
`agent_id`, `status`, `awaiting` — the interrupt —, `assignee`, `deadline`, `updated_at`). An
`ask` with no `assignee` waits on the run's user (`user:<user>`). Answer one with
`agent.resume`. It is `runs.iterate(status=PAUSED, assignee=..., tenant=..., max_pages=10)`
with pages of 500 (`INBOX_LIMIT`, `INBOX_MAX_PAGES`): past 5000 runs it returns the newest and
logs a warning.

The assignee is a filter, not a lock: every key of the tenant reads every inbox. Answering is
checked in agent-runs: the application's key (it may act for anyone, the default) or an admin
key answers any run, and its `reviewer` is taken as given; a key restricted to listed people
answers only as one of them, a run assigned to that person or to nobody, never a group's run
(`AuthorizationError` otherwise). See [interrupts.md](interrupts.md#answering) and
[agent-runs' rule](https://github.com/amitmohapatra/agent-runs/blob/main/README.md#who-may-answer-a-paused-run).

Notifications (a run paused, escalated or finished) are agent-runs' tenant webhook
subscriptions (`RunsClient.webhooks.create`, `POST /v1/webhooks`), not a harness setting; a
receiver checks each delivery with `trellis.runs.webhooks.verify_signature`
([blocks/runs.md](blocks/runs.md#webhooks), [interrupts.md](interrupts.md#telling-people)).
Runs kept in process (no `RUNS_URL`) have no webhooks: nothing is told unless a run hook of
yours tells it.

## agent-runs wire

`RunsClient` sends `X-API-Key: TRELLIS_API_KEY` on every call. The harness calls, by operation
id: `runs.start` (`POST /v1/runs`, `RunStart` + `queue`), `runs.claim`
(`POST /v1/runs/claim` → `Claimed{run, lease}` or `204`), `runs.heartbeat`
(`POST /v1/runs/{id}/heartbeat`, with the progress `checkpoint` when there is one → `Lease`),
`runs.pause` (`POST /v1/runs/{id}/pause?worker_id=`, an `Interrupt` and the checkpoint),
`runs.resume` (`POST /v1/runs/{id}/resume`, an `InterruptResolution`), `runs.finish`
(`POST /v1/runs/{id}/finish?worker_id=`), `runs.get` (`GET /v1/runs/{id}`), `runs.list`
page by page (`GET /v1/runs?status=PAUSED&assignee=&limit=500&cursor=`, the inbox),
`artifacts.upload` (`POST /v1/runs/{id}/artifacts?worker_id=&checksum=`, an `ask` payload or
→ `ArtifactRef`), `artifacts.download` (`GET /v1/artifacts/{id}`), `artifacts.put_payload` /
`get_payload` (`POST /v1/runs/{id}/payloads`, `GET /v1/payloads/{sha256}`: a journal's segments),
`runs.cancel` (`POST /v1/runs/{id}/cancel {reason}`), `runs.release`
(`POST /v1/runs/{id}/release {worker_id, checkpoint?}`, a stopping worker's runs) and
`schedules.create` (`POST /v1/schedules`, `ScheduleSpec`).

**The tenant is explicit.** A call whose body names the tenant (a start, a pause, a schedule)
sends it as `X-Trellis-Tenant`; every other call the harness makes passes `tenant=` — the run's
own, from its record or its runtime — so a platform key (which has no tenant of its own) works
on every call, and nothing is remembered between calls. `h.inbox`, `h.feedback` and
`agent.resume` take `tenant=` for a platform key, like `agent.run`.

Every call is retried when it fails on the way — a transport error (a refused connection, a
timeout), `429`, `502`, `503` or `504` — up to 3 times, after the `Retry-After` agent-runs sent
(at most 30 s) or else with exponential backoff and full jitter (a random wait under a ceiling
of 0.25 s, doubled for each retry, at most 5 s). Every write is safe to repeat: a run start is
idempotent on its id, a pause or finish repeated by the same worker with the same status
answers the stored record, an artifact is stored once per checksum, a schedule is upserted. A
claim whose answer was lost leaves that run leased to this worker unworked until the lease
lapses (60 s), when agent-runs queues it again — late, never lost or run twice at once. A
pause or finish the store refuses as a conflict (`ConflictError`) is read back once: when the
run already is what was written (the first attempt landed, its answer did not), the run goes on
as recorded — it is not failed, queued again or executed again.

A refusal raises the SDK's errors (`trellis.runs`), read from agent-runs' problem document
(RFC 9457) by its `code`: every one is a `RunsError` with the problem's `code`, the `status` and
`retryable` (so a run that fails on it keeps whether it may be retried). `LEASE_LOST` is
`LeaseLostError` — the worker no longer holds the run: it stops and writes nothing more — and is
**not** a `ConflictError`, so the conflict read-back above never swallows a lost lease;
`CONFLICT` is `ConflictError` (a run id taken, an answer to another interrupt, an illegal
transition); `NOT_FOUND` is `NotFoundError` (a read by id answers `None` instead). A heartbeat
refused with `409`, whatever its code, is `LeaseLostError`. A call that still fails after its
retries raises `DependencyUnavailableError`, `retryable` true. `LocalRuns` raises the same
classes.
