# Pauses: `ask`, approvals, `resume`

## Asking

```python
answer = await trellis.current().ask(
    "Which supplier?",
    options=["ACME", "Globex"],
    assignee="role:procurement",
    deadline=tomorrow,
    escalate_to="role:procurement-leads",
)
```

The run pauses (`Result.status == PAUSED`, `Result.interrupt` a contracts `Interrupt`) and, on
resume, the call returns the answer. What the person is shown follows from what is asked:

| Arguments | `ui` | `reason` | `payload` |
|---|---|---|---|
| `options=` | `choice` | `CHOICE` | |
| `table=rows` | `table` | `QUESTION` (`REVIEW` with `expects=`) | `{"table": rows}` |
| `diff=(before, after)` | `diff` | `QUESTION` (`REVIEW` with `expects=`) | `{"diff": {"before", "after"}}` |
| anything else | `form` (`expects=` its schema) | `QUESTION` | |

The question is the run's user's to answer (`assignee="user:<user>"`) unless `assignee` names
someone else (`user:…`, `role:…`); it is in their inbox (`h.inbox(assignee)`). `escalate_to`
needs a `deadline`: when it passes, agent-runs hands the question to `escalate_to` (once), or
ends the run `TIMEOUT` when nobody is named. Runs kept in process (no `RUNS_URL`) do neither.

The whole signature: `await trellis.current().ask(question, *, expects=None, table=None,
diff=None, options=None, assignee=None, deadline=None, escalate_to=None)`. An `expects` that is
not a JSON Schema is refused where it is asked (`ConfigurationError: cannot ask 'How many?':
expects is not a valid JSON Schema: …`), not when someone answers. The same question
asked again in one run (same text, same kind, same options) is the same entry in the journal:
a re-run gets the answer it was given, in order.

A payload up to 16 KiB of JSON travels in the interrupt. A larger one is stored as a run
artifact in agent-runs (`POST /v1/runs/{id}/artifacts`, up to 50 MiB, kept 7 days after the
run ends) and travels as `payload_ref`; `serve_chat` serves it at
`{path}/runs/{run_id}/artifacts/{artifact_id}` from whichever process is asked, while the run
waits on it. The run's checkpoint stays small.

An approval (an `irreversible` tool, a catalog `approve_when` that holds) is the same pause
with `reason=APPROVAL` and the tool call attached.

## Answering

```python
await agent.resume(interrupt_id, "answer", answer="ACME", reviewer="lee")
await agent.resume(interrupt_id, "approve", reviewer="cfo")
await agent.resume(interrupt_id, "edit", answer={"amount": 9000}, reviewer="cfo")
await agent.resume(interrupt_id, "reject", reviewer="cfo")
await agent.resume(interrupt_id, "reject", answer="over budget", reviewer="cfo")  # with a reason
await agent.resume(interrupt_id, "cancel", reviewer="cfo")
```

What `ask` returns: the answer; `True`/`False` for approve/reject; the edited value for edit;
cancel ends the run `CANCELLED`. A rejected tool call is not run and the model reads that it
was rejected — with the reviewer's reason when the reject carries one as `answer`
(`"refund was not run: the approver rejected it (over budget)"`). The interrupt id names its run, so nothing else is needed; a
resume must answer the interrupt the run currently waits on.

**Answers are checked.** Before anything is sent or recorded, `resume` refuses an answer that
does not fit its question, with a `ConfigurationError` saying why, and the run keeps waiting:

* an `answer` must fit `expects` (JSON Schema) — `not an answer to <id>: the answer does not
  fit what was asked: 'five' is not of type 'integer'` — and, with no `expects`, be one of the
  `options` when there are some (`the answer 'M' is not one of the options ['S', 'L']`);
* an `edit` of a question (a review) must fit `expects` the same way;
* an `edit` of a tool call must fit the tool's input schema — its required arguments, the
  basic types of the declared ones, no unknown ones where the schema allows none (`the edited
  arguments do not fit refund: amount must be of type number`) — when the agent's toolbox lists
  the tool at that moment. A tool it cannot list then (an MCP tool whose server is down) is
  left to the call: the tool checks its arguments when it runs, and a failure is a result the
  model reads.

`approve`, `reject` and `cancel` carry nothing to check. The check is the one agent-runs makes
(`trellis.runs.answers`), so a run kept in process (no `RUNS_URL`) is checked the same way.
`serve_chat` answers a refused resume `409` (`BAD_RESUME: …`), and `serve_a2a` keeps the task
`input-required` with the reason. The edits a framework's own approval takes are the
framework's to check (below).

**Who may answer.** With agent-runs, the harness's key decides. The application's key (it may
act for anyone, the default) or an admin key answers any run, and `reviewer` is recorded as
given: the application vouches for it. A key restricted to listed people
(`may_act_as=["user:priya"]`) answers only as one of them (`reviewer="priya"` is `user:priya`)
and only a run assigned to that person or to nobody; never a run assigned to a group
(`role:…`), which the application's key answers. A refusal raises
`trellis.runs.AuthorizationError` saying why, before anything changes. The rule is
[agent-runs' "Who may answer a paused run"](https://github.com/amitmohapatra/agent-runs/blob/main/README.md#who-may-answer-a-paused-run); setting up such keys is
[onboarding.md](onboarding.md#4-optional-a-key-per-person-for-an-approvals-ui).

## How a run continues

* **LangGraph with a checkpointer** (Deep Agents with one included): `ask` is LangGraph's
  `interrupt`, and the resume is `Command(resume=...)` — the graph continues in place. A
  graph's own `interrupt(value)` is surfaced as a question and resumed with the raw answer;
  without a checkpointer it cannot be resumed and the run fails saying so.
* **LangChain `HumanInTheLoopMiddleware`, Deep Agents `interrupt_on`**: the middleware's own
  pause (below).
* **OpenAI Agents `needs_approval` tools**: the SDK's own pause, continued from its `RunState`
  (below).
* **Everything else** re-runs from the input as the next attempt, with the **journal**:
  answers already given return where their question is asked, and tool calls already made
  return their recorded outputs instead of running again. Entries are keyed by content (the
  question; the tool and its arguments) and consumed in order.

Per framework — which pauses resume in place, what the model is asked again, the frameworks'
own gates — see the [framework pages](README.md#which-target). A checkpointed graph resumes in
place only where its checkpointer still holds the pause — with `InMemorySaver`, the process
that paused it. Resumed elsewhere (a worker, a replica), an approval or `ask` of the harness's
is answered from the journal (a re-run, as without a checkpointer), and a graph's own
`interrupt()` or middleware pause fails the run: those need a shared checkpointer
([langgraph.md](frameworks/langgraph.md#approvals-and-pauses)).

The journal is the run's checkpoint: the pause stores it with the run (`RunRecord.checkpoint`
in agent-runs, cleared when the run ends), and whichever process or worker resumes the run
reads it back with the resolution, so a resume elsewhere repeats no question and no tool
call. A worker also saves it as progress after every call with side effects, so a worker that
dies mid-run repeats none either ([runs.md](runs.md#workers)). A journal over agent-runs'
1 MiB checkpoint bound (tools that returned a lot) is stored as a run artifact the checkpoint
names, and read back the same way. `ReAct` journals its model
steps too: a resume replays the steps before the pause instead of asking the model again.

### Framework approvals: LangChain's middleware and OpenAI Agents' `needs_approval`

A framework that asks for approval itself pauses the run the same way: `Result.interrupt` is
an approval (`reason=APPROVAL`, `ui="approve"`, the call in `tool_call`), it is in the inbox,
and the same five decisions answer it.

**`create_agent(middleware=[HumanInTheLoopMiddleware(interrupt_on=...)])`, and Deep Agents'
`create_deep_agent(interrupt_on=...)`** (a checkpointer is needed, as for any graph pause). The
middleware batches the calls of one model message that need review into one request; the
interrupt shows the first (`"Approve email?"`, `"Approve email? (and 1 more call)"`) and carries
the whole request as its `payload` — `action_requests` (each call's `name`, `args`,
`description`) and `review_configs` (each call's `allowed_decisions`). The resume is the
middleware's `{"decisions": [...]}`, one per call, in order:

| `resume(...)` | Decisions sent |
|---|---|
| `"approve"` | `approve` for every call |
| `"edit", answer={...}` | `edit` of the first call with those arguments (the tool runs with them), `approve` for the others |
| `"reject"` (`answer="why"`) | `reject` for every call, the reason as its `message` (the model reads it) |
| `"answer", answer=...` | `respond` for every call: the tool does not run and the model reads the answer as its result |
| `"answer", answer={"decisions": [...]}` | sent as it is — one decision per call, for a batch decided call by call |

A decision that a call's `allowed_decisions` does not include is refused by `resume`
(`ConfigurationError`) before anything is recorded, and the run keeps waiting. An edit's
arguments go to the middleware as given (it and the tool check them), as do decisions sent as
`{"decisions": [...]}`. Leave the
tools the middleware covers out of the harness's own approvals (`side_effects="write"` or
`"read"`, no catalog `approve_when`), or the call is approved twice.

**OpenAI Agents `function_tool(needs_approval=...)`.** The SDK pauses with its `RunState`,
which the pause keeps as the run's checkpoint, and the resume continues that state. The SDK
approves or rejects a call, with a `rejection_message` the model reads — it takes neither
edited arguments nor an answer in place of the result (verified against `openai-agents` 0.22:
`RunState.approve(item)`, `RunState.reject(item, rejection_message=)`), so:

| `resume(...)` | On the `RunState` |
|---|---|
| `"approve"` | `approve(item)`: the tool runs |
| `"reject"` (`answer="why"`) | `reject(item, rejection_message=why)` (no reason: the SDK's own message) |
| `"answer", answer=...` | `reject(item, rejection_message=answer)`: the tool does not run and the model reads the answer |
| `"edit", answer={...}` | `reject(item, rejection_message="A reviewer changed the arguments of this <tool> call, so it was not run. Call <tool> again with exactly these arguments: {...}")`; when the model then calls the tool with exactly those arguments, that call is approved in the same attempt and runs once — with any other arguments it asks again |

Several `needs_approval` calls in one turn are asked about one at a time.

A run started with `run`/`stream` continues in the process that calls `resume`; a queued run
(`start`, a schedule) goes back to the queue and a worker continues it (`Result.status ==
QUEUED`).
