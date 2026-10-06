# Flows

Each flow as a sequence diagram, written from the code it names: what calls what, in which
order, and which service each call reaches. The structure these flows run through — the
pipeline, the bridge, the journal, the adapters, the middleware — is
[architecture.md](architecture.md). Calls in a `Writes` lane are background writes: the run
does not wait for them.

* [A run](#a-run)
* [Memory: push, pull, record](#memory-push-pull-record)
* [A pause, an approval and a resume](#a-pause-an-approval-and-a-resume) — in place with a
  checkpointer, and by journal replay
* [A worker claims a run, crashes, and another resumes it](#a-worker-claims-a-run-crashes-and-another-resumes-it)
* [A schedule fires](#a-schedule-fires)
* [A gateway MCP call, and Code Mode](#a-gateway-mcp-call-and-code-mode)
* [Sub-agents](#sub-agents)
* [AG-UI chat](#ag-ui-chat) and [A2A](#a2a)
* [Evaluation, offline and online](#evaluation-offline-and-online)
* [ReAct: one model step through the middleware](#react-one-model-step-through-the-middleware)
* [Stopping a worker](#stopping-a-worker)

## A run

`agent.run` → `pipeline.attempt` (`agent.py`, `pipeline.py`). A wrapped agent answers with
memory recalled, calls an MCP tool, writes a memory through the `memory_remember` pull tool,
and a person's feedback arrives later.

```mermaid
sequenceDiagram
  autonumber
  actor User as Application / user
  participant Agent as Harness h · Agent (h.wrap)
  participant Runs as RunStore (RunsClient or LocalRuns)
  participant P as pipeline.attempt
  participant Mem as Memory service
  participant FW as Adapter + framework
  participant Br as tools.bridge
  participant GW as Bifrost
  participant W as Writes (background)
  participant LF as Langfuse (scores API / OTLP)

  User->>Agent: await agent.run(input, user=, thread=)
  Agent->>Mem: GET /v1/keys/self (tenant, kept 10 min, the last answer while memory is down)
  Agent->>Runs: start(RunStart: without=, framework_options= in metadata) → RUNNING
  Agent->>P: attempt(agent, record)
  P->>GW: MCP tools/list with the virtual key (definitions, kept 300 s)
  P->>Mem: GET /v1/tools?names= + If-None-Match (governance: risks, approve_when, every 30 s)
  P->>Mem: GET /v1/agent-tools (pull tools, kept 10 min)
  P->>Mem: POST /v1/context (memory recall: retrieve memory span)
  Mem-->>P: rendered, bundle_id, tools [name, confidence]
  P->>FW: prepare_input(input, context), invoke(native tools, framework_options)
  FW->>GW: chat completion (the team's model through Bifrost)
  FW->>Br: call erp-get_stock(sku)
  Br->>Br: replay? hooks, governance.check: read → run
  Br->>GW: POST /v1/mcp/tool/execute (execute_tool span)
  GW-->>Br: result
  Br-)W: memory.record_tool
  Br-->>FW: result text
  FW->>Br: call memory_remember(content)
  Br->>Mem: POST /v1/agent-tools/memory_remember (in the run's scope)
  Br-->>FW: stored
  FW-->>P: output → extract(answer, transcript)
  P->>Runs: finish(SUCCESS, output, tenant=)
  P-->>User: Result(SUCCESS, answer)
  W-)Mem: POST /v1/tools/invocations (the MCP call)
  W-)Mem: POST /v1/messages (transcript, one batch per attempt)
  W-)Mem: POST /v1/feedback (system: confirm)
  W-)Mem: POST /v1/verify (sampled: TRELLIS_GROUNDING_SAMPLE) → grounding score
  W-)LF: score grounding on the run's trace
  User->>Agent: await h.feedback(run_id, "correct", correction)
  Agent->>Runs: get(run_id, tenant=)
  Agent->>Mem: POST /v1/feedback (human, review pending)
  Agent->>LF: score feedback (POST /api/public/scores and a score span)
```

The memory tools' own calls are not recorded again (the service logs them); everything else
the bridge runs is. Runnable: [examples/02_way1_react/agent.py](../examples/02_way1_react/agent.py).

## Memory: push, pull, record

What the harness does with the memory service around one run (`agent.py`:
`Agent.push`, `Agent.record_tool`, `Agent.recorded_run`; `clients/memory.py`). Push is the
context before the framework runs; pull is the memory tools the model calls itself; record is
what the run leaves behind. `without=` turns each off (`memory_push`, `memory_pull`,
`records`; `memory` all three).

```mermaid
sequenceDiagram
  autonumber
  participant P as pipeline.attempt
  participant A as Agent (memory)
  participant Mem as Memory service
  participant FW as Framework (model)
  participant Br as tools.bridge
  participant W as Writes (background)

  Note over P,Mem: push (memory_push)
  P->>A: push(task, the run's tool names)
  A->>Mem: POST /v1/context {query, scope, budget, tools.available (from 5 tools)}
  Mem-->>A: rendered, bundle_id, evidence_status, tools [name, confidence]
  A-->>P: the context, the tools offered (hints, memory tools, those already used)
  P->>FW: the context as a system message (window off when the framework keeps the thread)
  Note over FW,Mem: pull (memory_pull)
  FW->>Br: memory_search(query)
  Br->>Mem: POST /v1/agent-tools/memory_search {args, scope}
  Mem-->>Br: items [id, kind, text]
  Br-->>FW: the result (journaled, not recorded again)
  FW->>Br: tool_search(task)
  Br->>Mem: POST /v1/agent-tools/tool_search
  Mem-->>Br: the tools that fit: offered from the next model call
  Note over Br,Mem: record (records)
  FW->>Br: refund(order, amount)
  Br-)W: record_tool(refund, args, outcome)
  W-)Mem: POST /v1/tools/invocations (Idempotency-Key: the call's key)
  P-)W: the run ended
  W-)Mem: POST /v1/messages (the transcript, source_message_id each: stored once)
  W-)Mem: POST /v1/feedback (system: SUCCESS confirm / ERROR reject)
  W-)Mem: POST /v1/verify {bundle_id, answer} (a sampled share: grounding)
```

A failed context read is a `warning` event and the run goes on without it; a write given up is
spooled with `TRELLIS_SPOOL_DIR` ([memory.md](memory.md)). Runnable:
[examples/02_way1_function/memory_documents_feedback.py](../examples/02_way1_function/memory_documents_feedback.py),
and with no harness [examples/03_way2_memory/memory_block.py](../examples/03_way2_memory/memory_block.py).

## A pause, an approval and a resume

A call governance asks about pauses the run; a person answers from the inbox; the run continues
and the approved call runs once. How it continues depends on the target
(`pipeline._paused`, `Agent.resume`, `Adapter.resume_input`).

### In place: a LangGraph graph with a checkpointer

The harness's `ask` *is* LangGraph's `interrupt`: the graph stops inside the tool node, its
checkpointer keeps where, and the resume is `Command(resume=...)` — the model is not asked
again ([examples/02_way1_deepagents/agent.py](../examples/02_way1_deepagents/agent.py)).

```mermaid
sequenceDiagram
  autonumber
  actor App as Application
  participant Agent as Agent
  participant P as pipeline.attempt
  participant G as LangGraph graph (checkpointer)
  participant Br as tools.bridge
  participant Gov as Governance
  participant AR as agent-runs (or LocalRuns)
  actor Lee as Approver

  App->>Agent: await agent.run(input, user="ada", thread="t-7")
  Agent->>AR: start → RUNNING
  Agent->>P: attempt 1
  P->>G: ainvoke(messages, {thread_id: t-7, ...framework_options})
  G->>Br: tool node: refund(o-7, 40)
  Br->>Gov: check(refund, args)
  Gov-->>Br: ask (irreversible)
  Br->>G: Runtime.approve → langgraph interrupt(question)
  G-->>P: __interrupt__ (the checkpointer holds the step)
  P->>AR: pause(Interrupt APPROVAL, checkpoint = journal) → PAUSED
  P-->>App: Result(PAUSED, interrupt)
  Lee->>Agent: await h.inbox("user:ada") → the RunSummary
  Lee->>Agent: await agent.resume(interrupt_id, "approve", reviewer="lee")
  Agent->>AR: get(run) — the interrupt it waits on, the decision checked
  Agent->>AR: resume(InterruptResolution) → RUNNING (it was started in process)
  Agent->>P: attempt 2 (last_resolution)
  P->>G: ainvoke(Command(resume={interrupt id: resolution}))
  G->>Br: the tool node again: refund(o-7, 40)
  Br->>Br: the pending approval is answered: approved
  Br->>Br: execute refund (once), journal, record
  G-->>P: the final message
  P->>AR: finish(SUCCESS)
  P-->>Lee: Result(SUCCESS)
```

### By journal replay: everything else

A function, an OpenAI Agents `Agent` (for the harness's own approvals), a graph with no
checkpointer: the pause ends the attempt, and the resume runs the target again from its input
against the journal ([examples/02_way1_openai_agents/agent.py](../examples/02_way1_openai_agents/agent.py),
[examples/02_way1_langgraph/agent.py](../examples/02_way1_langgraph/agent.py)).

```mermaid
sequenceDiagram
  autonumber
  participant Agent as Agent
  participant P as pipeline.attempt
  participant FW as Framework
  participant Br as tools.bridge
  participant J as Journal
  participant AR as agent-runs (or LocalRuns)

  Agent->>P: attempt 1
  P->>FW: invoke(input)
  FW->>Br: lookup(sku) — read
  Br->>J: outputs[lookup(sku)#1] = 3
  FW->>Br: create_po(...) — irreversible
  Br->>J: pending: approval of create_po(...)#1
  Br-->>FW: raises Paused (a framework that swallows it is still paused)
  P->>AR: pause(interrupt, checkpoint = journal) → PAUSED
  Note over Agent,AR: the approver answers: agent.resume(..., "approve")
  Agent->>AR: resume(resolution) → RUNNING, attempt 2
  Agent->>P: attempt 2: checkpoint + last_resolution
  P->>J: file the resolution under the pending question
  P->>FW: invoke(input) — the same input, again
  FW->>Br: lookup(sku)
  Br->>J: replay: 3 (nothing runs)
  FW->>Br: create_po(...)
  Br->>J: replay: the approval is answered
  Br->>Br: execute create_po (once), journal, record
  FW-->>P: the answer
  P->>AR: finish(SUCCESS)
```

Entries are keyed by content and consumed in order: a re-planned call nobody approved is asked
about again, never matched to another approval. A run that came from the queue goes back to it
on resume (`QUEUED`) and a worker runs attempt 2 the same way
([interrupts.md](interrupts.md#how-a-run-continues)).

## A worker claims a run, crashes, and another resumes it

`agent.start` → `trellis.runs.Worker` (`worker/`) → `agent.execute(job)`; the progress
checkpoint is `Runtime._saved` after every call with side effects
([examples/06_scenarios/worker_crash_resume.py](../examples/06_scenarios/worker_crash_resume.py)).

```mermaid
sequenceDiagram
  autonumber
  actor App as Application
  participant Agent as Agent
  participant AR as agent-runs
  participant A as Worker A
  participant B as Worker B
  participant Pay as pay (a write tool)

  App->>Agent: handle = await agent.start("invoice 7", user="ada")
  Agent->>AR: POST /v1/runs (queue: true) → QUEUED
  A->>AR: POST /v1/runs/claim (lease 60 s)
  AR-->>A: {run, lease} → RUNNING, attempt 1
  A->>A: agent.execute(job): pipeline.attempt
  loop every lease/3 while it runs
    A->>AR: POST /v1/runs/{id}/heartbeat (409 → LeaseLostError: stop, write nothing)
  end
  A->>Pay: pay(120) through the bridge
  Pay-->>A: paid 120
  A->>AR: heartbeat(checkpoint = journal): the progress
  Note over A: the process dies: no finish, no release
  AR->>AR: the lease lapses → QUEUED, attempt 2 (after a backoff)
  B->>AR: POST /v1/runs/claim
  AR-->>B: {run: checkpoint = the journal}
  B->>B: agent.execute(job): attempt 2 reads the journal
  B->>B: pay(120) → replayed: "paid 120" (not paid again)
  B->>AR: POST /v1/runs/{id}/finish?worker_id= (SUCCESS)
  App->>Agent: await handle.result()
  Agent->>AR: GET /v1/runs/{id} (polled every 0.5 s) → SUCCESS
```

A write that was *running* when its worker died (started, no outcome journaled) is not run
again: the model is told its effect is unknown, with its idempotency key
([reliability.md](reliability.md#unknown-outcomes)).

## A schedule fires

`agent.schedule` (`agent.py`) → agent-runs' schedules and ticker → a worker. The run options a
schedule takes are kept in its `ScheduleSpec.metadata` and copied into every run it fires
([examples/06_scenarios/scheduled_run_selection_limit.py](../examples/06_scenarios/scheduled_run_selection_limit.py)).

```mermaid
sequenceDiagram
  autonumber
  actor Dev as Deployment code
  participant Agent as Agent
  participant AR as agent-runs (schedules + ticker)
  participant Wk as Worker
  participant P as pipeline.attempt
  participant FW as Framework

  Dev->>Agent: await agent.schedule("0 7 * * 1-5", input, on_behalf_of="ada", tz=, without={"judges"}, timeout=120, framework_options={...}, priority=10)
  Agent->>Agent: check without= and framework_options= (JSON) now
  Agent->>AR: POST /v1/schedules ScheduleSpec{cadence, timezone, input, timeout_seconds, priority, concurrency_key, agent_version, metadata: {without, framework_options}}
  AR-->>Agent: Schedule (an upsert: same agent, person, cadence, input → the same one)
  Note over AR: at 07:00 Europe/Paris (or POST /v1/schedules/{id}/fire)
  AR->>AR: RunStart{on_behalf_of: ada, metadata: the schedule's + schedule_id, timeout_seconds, priority} → QUEUED
  Wk->>AR: claim (highest priority first)
  AR-->>Wk: the run
  Wk->>P: agent.execute(job)
  P->>P: without = run_without(record), options = metadata.framework_options, budget = timeout_seconds
  P->>FW: invoke(input, config with the options) — no judges for this run
  P->>AR: finish(SUCCESS)
```

Without `RUNS_URL` the in-process store keeps the schedule and fires it when a worker in this
process asks for work, or when `h.runs.schedules.fire(id)` is called ([runs.md](runs.md#schedules)).

## A gateway MCP call, and Code Mode

`tools/toolbox.py` lists and publishes; `clients/bifrost.py` executes through `bifrost-sdk`
([gateway.md](gateway.md),
[examples/06_scenarios/gateway_code_mode_governed_evals.py](../examples/06_scenarios/gateway_code_mode_governed_evals.py)).

```mermaid
sequenceDiagram
  autonumber
  participant P as pipeline.attempt
  participant TB as Toolbox
  participant Gov as Governance
  participant Mem as Memory service
  participant Br as tools.bridge
  participant GW as Bifrost gateway
  participant S as MCP servers
  participant W as Writes (background)

  P->>TB: tools(code_mode = not without code_mode)
  TB->>GW: POST /mcp tools/list (the virtual key: what it allows)
  GW-->>TB: tools + annotations, Code Mode clients as meta-tools
  TB->>GW: tools/call listToolFiles, readToolFile (the Code Mode clients' declarations)
  TB->>GW: GET /api/mcp/clients (tools_to_auto_execute: left out)
  TB-)Mem: PUT /v1/tools/catalog (publish the listing, background)
  TB->>Gov: rules for the tools: read-only Code Mode servers, 3+ or 20+ tools → Code Mode
  Note over P,S: one MCP call
  P->>Br: erp-refund(order, amount)
  Br->>Gov: check → ask (approve_when holds) ... approved on resume
  Br->>GW: POST /v1/mcp/tool/execute {id: idempotency key} + x-bf-mcp-include-clients: erp, x-trellis-identity (tenant, user)
  GW->>S: tools/call (the identity header forwarded where allowed)
  S-->>GW: result
  GW-->>Br: {role: tool, content}
  Br-)W: record_tool(erp-refund)
  Note over P,S: Code Mode: one script, many calls
  P->>Br: execute_tool_code(code)
  Br->>Gov: check → read (every server in it only reads)
  Br->>GW: POST /v1/mcp/tool/execute executeToolCode + x-bf-parent-request-id: run_id
  GW->>S: crm.customer(...), wiki.policy(...) (logged under run_id)
  GW-->>Br: what the script printed
  Br-)W: record_tool(execute_tool_code)
  P-)W: after the attempt: the scripts' nested calls
  W->>GW: GET /api/mcp-logs?llm_request_ids=run_id (every 2 s until two reads agree, ≤ 20 s)
  W-)Mem: POST /v1/tools/invocations (each nested call)
```

No model request ever gets the gateway's MCP tools: `bifrost-sdk` sends the deny-all scope
(`NO_GATEWAY_TOOLS`) on completions, and a framework's own model client gets it from
`h.model_headers()`.

## Sub-agents

`agent.as_tool()` (`subagents.py`): a planner calls a child agent; the child asks a person;
the answer goes back through the planner
([examples/06_scenarios/parallel_subagents_mixed_frameworks.py](../examples/06_scenarios/parallel_subagents_mixed_frameworks.py)).

```mermaid
sequenceDiagram
  autonumber
  actor Ada as User
  participant PA as planner (parent run)
  participant Br as tools.bridge (parent)
  participant SA as SubAgent tool (booker)
  participant CR as booker (child run)
  participant AR as agent-runs (or LocalRuns)

  Ada->>PA: planner.run("Plan a weekend", user="ada")
  PA->>Br: scout(...), visas(...), booker(...) — one step, all read: at once
  Br->>SA: booker(message)
  SA->>AR: start(RunStart{run_id: stable(parent, call key), parent_run_id, tenant, user, thread})
  SA->>CR: pipeline.attempt(child) — the parent's deadline and time left, its trace
  CR->>CR: ask("Which budget?") → the child pauses
  CR->>AR: pause(child)
  SA-->>Br: the parent pauses: interrupt{question, payload.subagent: {agent_id, run_id}}
  Note over Br: the other calls of the step finish first (journaled)
  PA->>AR: pause(parent, checkpoint = journal incl. the child's)
  PA-->>Ada: Result(PAUSED, "Which budget?")
  Ada->>PA: planner.resume(interrupt_id, "answer", answer="low")
  PA->>Br: attempt 2: scout and visas replayed, booker called again
  Br->>SA: booker(message)
  SA->>AR: the child's record: PAUSED → resume(child, the answer)
  SA->>CR: the child's attempt 2: ask returns "low", hotels(...) runs
  CR->>AR: finish(child, SUCCESS)
  SA-->>Br: the child's output
  PA->>AR: finish(parent, SUCCESS)
```

Cancelling the parent cancels its children; a child never works past its parent's time
([subagents.md](subagents.md)).

## AG-UI chat

`agent.serve_chat(app)` (`agui/`): a chat UI runs the agent, answers its question and
reconnects after a dropped connection
([examples/05_features/serve_agui_and_a2a.py](../examples/05_features/serve_agui_and_a2a.py)).

```mermaid
sequenceDiagram
  autonumber
  actor UI as Chat UI (AG-UI client)
  participant R as serve_chat routes
  participant H as Hub (buffered events per run)
  participant P as pipeline.attempt (background task)
  participant AR as agent-runs (event log, RUNS_URL)

  UI->>R: POST /agui/run {threadId, runId, messages} (+ your auth)
  R->>R: identity(request) → user, tenant from the key
  R->>H: open the run's buffer
  R->>P: start the run in the background (a client that leaves does not stop it)
  P->>H: RunEvents (numbered)
  P-)AR: the same events into the run's event log
  H-->>UI: SSE: RUN_STARTED, TEXT_MESSAGE_*, TOOL_CALL_* (id: n)
  P->>H: the run asks "Which evening?"
  H-->>UI: RUN_FINISHED {outcome: interrupt, the question, options, component}
  UI->>R: POST /agui/run {resume: [{interruptId, status: resolved, payload: "friday"}]}
  R->>P: agent.resume(...) — continues the run
  P->>H: events
  Note over UI,R: the connection drops
  UI->>R: GET /agui/runs/{runId}/events (Last-Event-ID: n)
  alt this replica served the run
    R->>H: events after n, then live
  else another replica (RUNS_URL)
    R->>AR: GET /v1/runs/{id}/events?after=n
  end
  R-->>UI: SSE: the rest, RUN_FINISHED {outcome: success}
```

## A2A

The calling side is `remote(url)` (`a2a/client.py`) — what the `a2a(url)` tool runs inside a
calling run; the serving side is another harness's `serve_a2a(app, url)` (or any A2A server).

```mermaid
sequenceDiagram
  autonumber
  participant P as Calling run (bridge)
  participant C as a2a(url) tool (RemoteAgent)
  participant S as Remote serve_a2a (DefaultRequestHandler)
  participant X as RunExecutor
  participant RP as Remote pipeline.attempt

  Note over C: resolve once per toolbox: GET {url}/.well-known/agent-card.json
  P->>C: call greeter(message)
  C->>S: SendStreamingMessage (A2A-Extensions: trusted-identity,<br/>x-trellis-identity: tenant + user, context id = the calling thread)
  S->>X: execute(context, event queue)
  X->>X: HeaderIdentity: the header's tenant must be the key's
  X->>RP: a run (run_id = task id)
  RP-->>X: RunEvents
  X-->>C: task SUBMITTED, WORKING (text, progress), artifact "result", COMPLETED
  C-->>P: the result artifact (or the text)
  alt the remote run asks something
    X-->>C: INPUT_REQUIRED + the question
    C->>P: on_input = runtime.ask(question): the calling run pauses
    C->>S: CancelTask (the remote task is not left waiting)
    Note over P: on resume the call is made again and ask returns the answer,<br/>which is sent on the new remote task as the next message
  else the remote run fails
    X-->>C: FAILED
    C-->>P: ToolError: the calling model reads "greeter failed: ..."
  end
```

From plain code the same client is `remote(url, tenant=, user=)`: a question goes to `on_input`,
or is raised as `InputRequired` and answered with `reply(task_id, answer)`
([examples/03_way2_a2a/remote.py](../examples/03_way2_a2a/remote.py)). The serving side also
sends signed push notifications to a client's webhook ([surfaces.md](surfaces.md)).

## Evaluation, offline and online

### Offline: `h.evaluate` over a dataset

`evals.evaluate` (`h.evaluate` delegates to it)
([examples/05_features/trajectory_evals.py](../examples/05_features/trajectory_evals.py)).

```mermaid
sequenceDiagram
  autonumber
  actor Dev as Developer / CI
  participant H as evaluate (h.evaluate)
  participant LF as Langfuse
  participant P as pipeline.attempt (per item)
  participant Mem as Memory service
  participant GW as Bifrost (judge key)
  participant W as Writes (background)

  Dev->>H: await h.evaluate(agent, "support-golden", [grounding(), called("lookup"), llm_judge(...)])
  H->>LF: GET /api/public/v2/datasets/support-golden
  loop every page
    H->>LF: GET /api/public/dataset-items?datasetName=&page=&limit=50
  end
  par concurrency items at a time
    H->>LF: POST /api/public/dataset-run-items {runName, datasetItemId, traceId} (v3)
    H->>P: attempt(...) inside telemetry.experiment: langfuse.experiment.* on every span (v4)
    P->>Mem: POST /v1/context (bundle_id)
    P-->>H: Result + the run's tool calls (the trajectory)
    H->>Mem: grounding: POST /v1/verify {bundle_id, answer}
    H->>H: called / tool_sequence: read EvalCase.trajectory
    H->>GW: llm_judge: POST /v1/chat/completions (TRELLIS_JUDGE_MODEL, temperature 0)
    GW-->>H: {"score", "reasoning"} (malformed → asked once more)
    H->>LF: POST /api/public/scores (each score, on the run's trace)
  end
  H->>W: drain, then export the spans
  H-->>Dev: EvalReport (items in dataset order, summary per evaluator)
```

A pausing item is cancelled and reported `interrupted`; a failing one `error` — never fatal.

### Online: judges on sampled runs

```mermaid
sequenceDiagram
  autonumber
  actor User as Application / user
  participant P as pipeline.attempt
  participant Runs as RunStore
  participant W as Writes (background)
  participant J as judges (Harness(judges=[...]))
  participant GW as Bifrost (judge key)
  participant LF as Langfuse

  User->>P: await agent.run(question, user=)
  P->>Runs: finish(SUCCESS, answer)
  P->>P: sampled(run_id, TRELLIS_JUDGE_SAMPLE)? and not without judges
  P-)W: submit judge.<name> (one per judge)
  P-->>User: Result(SUCCESS, answer): nothing waits for the judges
  W->>J: judge(EvalCase(question, answer, context, memory, run_id, trajectory), [one judge])
  J->>GW: POST /v1/chat/completions (TRELLIS_JUDGE_MODEL)
  GW-->>J: {"score", "reasoning"}
  J->>LF: POST /api/public/scores on the run's trace (and a score span)
  Note over W,J: a judge that fails is a warning event and a log line, never a failed run
```

## ReAct: one model step through the middleware

`ReAct(...)` builds a `create_agent` graph whose middleware wrap every model call and every
tool call (`react.py`, `middleware.py`; order in
[architecture.md](architecture.md#the-middleware)). One step: the model is asked, it calls two
tools, the calls go through the bridge.

```mermaid
sequenceDiagram
  autonumber
  participant G as create_agent graph (model node)
  participant FS as FilesystemMiddleware
  participant HT as HarnessTools
  participant SG as StallGuard
  participant SL as StepLimit
  participant CE as ContextEditing
  participant SU as Summarization
  participant RT as ReadTools
  participant MH as ModelHooks
  participant M as Chat model
  participant T as tools node
  participant Br as tools.bridge

  Note over G: once per invocation (before the agent): PatchToolCalls answers calls left unanswered, StepLimit counts from 0
  G->>FS: wrap_model_call(request)
  FS->>HT: + the read_file tool's guidance
  HT->>SG: tools = the run's offered tools (hints, memory, used), sorted
  SG->>SL: the last steps repeat a call max_repeats times, or 3 failed? → ModelError
  SL->>CE: step n of max_steps (past it: tool_choice none, one last answer)
  CE->>SU: over half the window: older tool results cleared (placeholder → read_result)
  SU->>RT: near the window's end: older turns summarized (one model call), history to a file
  RT->>MH: read_file / read_result offered only once there is something to read
  MH->>MH: before_model hooks, the pinned prompt, the task kept ahead of a summary
  MH->>M: the request (chat span, model_timeout and the run's time left)
  M-->>MH: AIMessage with tool_calls [lookup, refund]
  MH->>MH: after_model hooks, usage on the span, a model error → ModelError (retryable?)
  MH-->>G: the response, back out through every layer
  G->>MH: after_model: calls whose arguments are not JSON answered with what was wrong
  G->>T: the calls of the step (at once)
  T->>FS: wrap_tool_call(lookup)
  FS->>HT: wrap_tool_call(lookup)
  HT->>Br: bridge.call(lookup, args, call id, step) — reads run at once
  Br-->>HT: outcome
  T->>HT: wrap_tool_call(refund) — a write: after the earlier writes of the step
  HT->>Br: bridge.call(refund, ...) — governed, journaled, recorded
  Br-->>HT: outcome (or the run pauses)
  HT-->>FS: ToolMessages
  FS-->>T: a result over 20,000 tokens saved as a file, a preview in its place
  T-->>G: the next model step
```

The graph's checkpoint after every step goes into the run's journal (`RunCheckpointer`): a
resume — after a pause, a crash or a requeue — continues from the last step without asking the
model again ([frameworks/react.md](frameworks/react.md)). Middleware you add with
`ReAct(middleware=[...])` sits between `ReadTools` and `ModelHooks`
([examples/06_scenarios/planning_todolist.py](../examples/06_scenarios/planning_todolist.py)).

## Stopping a worker

`python -m trellis.harness.worker` serves the worker: `trellis.runs.Worker.serve()` turns
`SIGTERM`/`SIGINT` into `stop()`, and the harness worker drains the writes when the loop ends.

```mermaid
sequenceDiagram
  autonumber
  participant OS as Orchestrator
  participant CLI as python -m trellis.harness.worker
  participant HW as harness Worker
  participant Wk as trellis.runs.Worker
  participant R as Runs it holds
  participant AR as agent-runs
  participant W as Writes
  CLI->>HW: serve()
  HW->>W: start()
  HW->>Wk: serve()
  OS->>Wk: SIGTERM
  Wk->>Wk: stop()
  Wk--xAR: no more claims
  par within GRACE_SECONDS (25 s)
    R->>AR: finish / pause (as usual)
  end
  alt a run is still going
    Wk->>R: cancel(RELEASED): nothing written
    Wk->>AR: release(run): QUEUED at once, next attempt
  end
  Wk-->>HW: the loop ended
  HW->>W: drain ≤ DRAIN_SECONDS
  CLI->>W: h.aclose(): spool or count the rest
  CLI-->>OS: exit 0
```
