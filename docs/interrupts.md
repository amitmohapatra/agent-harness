# Pauses: `ask`, approvals, `resume`

## Asking

**What.** `trellis.current().ask(...)` asks a person and waits: the run pauses
(`Result.status == PAUSED`, `Result.interrupt` a contracts `Interrupt`) and, on resume, the call
returns the answer.

**When.** When only a person knows: a choice, a value, a correction, a sign-off. A tool call's
approval needs no `ask` (governance, `approval=`, below).

**Where.** Any tool or node of any adapter (`ReAct`, a function, LangGraph — `interrupt()`
in place with a checkpointer —, Deep Agents, OpenAI Agents, Claude); Way 2 builds the same
question (`Question`, below).

**How.**

```python
from trellis.contracts import Option

plans = await trellis.current().ask(
    "Which plans?",
    options=[Option(value="basic", label="Basic, 10 EUR", description="up to 3 seats"), "pro"],
    multiple=True,  # the answer is a list of values: ["basic", "pro"]
    component="plan-picker",  # your own screen, where a surface has it
    props={"customer": "acme"},  # its data, passed as it is
    assignee="role:sales",
    deadline=tomorrow,
    escalate_to="role:sales-leads",
)
address = await trellis.current().ask(
    "Where to?", form=Address, ui_schema={"street": {"ui:widget": "textarea"}}
)  # an Address: the form is Address's JSON Schema
```

What the person is shown follows from what is asked:

| Arguments | `ui` | `reason` | `payload` |
|---|---|---|---|
| `options=` (strings or `Option(value, label, description)`; `multiple=True` for several picks) | `choice` | `CHOICE` | |
| `table=rows` | `table` | `QUESTION` (`REVIEW` with `expects=`) | `{"table": rows}` |
| `diff=(before, after)` | `diff` | `QUESTION` (`REVIEW` with `expects=`) | `{"diff": {"before", "after"}}` |
| anything else | `form` (`expects=` its JSON Schema, or `form=` a pydantic model; `ui_schema=` its widget hints, react-jsonschema-form's `uiSchema`) | `QUESTION` | |

`component=` names your own screen and `props=` its data: a surface that has that screen
renders it with the props as they are (the reference inbox's `window.trellisComponents`, your
AG-UI client from the interrupt's `metadata`), any other renders `ui`. Everything goes on the
contracts' `Interrupt` as it is. An answer carries option values, never labels; with `form=`
it is read back into the model.

The whole signature: `await trellis.current().ask(question, *, expects=None, form=None,
table=None, diff=None, options=None, multiple=False, ui_schema=None, component=None,
props=None, assignee=None, deadline=None, escalate_to=None)`.

**Automatic.** The question is the run's user's to answer (`assignee="user:<user>"`) unless
`assignee` names someone else (`user:…`, `role:…`); it is in their inbox
(`h.inbox(assignee)`), and whoever waits is told ([below](#telling-people)). `escalate_to`
needs a `deadline`: when it passes, agent-runs hands the question to `escalate_to` (once), or
ends the run `TIMEOUT` when nobody is named. Runs kept in process (no `RUNS_URL`) do neither.
The same question asked again in one run (same text, kind and options) is the same entry in
the journal: a re-run gets the answer it was given, in order. A payload up to 16 KiB of JSON
travels in the interrupt; a larger one is stored as a run artifact in agent-runs (`POST
/v1/runs/{id}/artifacts`, up to 50 MiB, kept 7 days after the run ends) and travels as
`payload_ref`; `serve_chat` serves it at `{path}/runs/{run_id}/artifacts/{artifact_id}` from
whichever process is asked, while the run waits on it. The run's checkpoint stays small.

**On failure.** A question that cannot be asked as given is refused where it is asked
(`ConfigurationError`, saying why: an `expects` that is not a JSON Schema, `form=` and
`expects=` both, `props` without a `component`, two options with one value, `multiple` with
nothing to pick, `escalate_to` without a `deadline`). An answer that does not fit is refused at
`resume` (below). An answer that fits `form`'s schema but not its own validators fails the run
where `ask` returns, saying so.

**Way 2.** `trellis.harness.asking.Question` takes the same arguments and is what `ask` builds:
`question.interrupt(tenant=, run_id=)` is the `Interrupt` to `runs.pause`, and
`question.answer(record.last_resolution)` reads the answer back (into `form`); agent-runs checks
the answer when it is given.

An approval (an `irreversible` tool, a catalog `approve_when` that holds, an approval
function's `Ask`) is the same pause with `reason=APPROVAL` and the tool call attached.

## Approval rules in code: `tool(approval=fn)`

**What.** Your rule for each call of a tool, ahead of governance:
`fn(args) -> None | True | Ask(...)` (sync or async) — `None`: governance decides as for any
tool; `True`: approved, it runs without asking (announced unless it only reads); `Ask(question,
assignee=None, component=None, props=None)`: a person approves it first, asked that, on that
screen.

**When.** For a rule only your code can say: an amount, a customer, the time of day. A rule an
administrator owns is the catalog's `approve_when` ([governance.md](governance.md)); a
guardrail across tools is a hook ([hooks.md](hooks.md)).

**Where.** `@tool(approval=fn)` / `tool(fn, approval=fn)`, every adapter (the bridge) and
`ReAct`; Way 2: `governed(fn, gov, ..., approval=fn)`, whose `on_ask(decision)` gets the
decision with the `Ask`'s `assignee`, `component` and `props`.

**How.**

```python
def refund_rule(args):
    if args["amount"] <= 20:
        return True  # small: no person
    if args["amount"] > 500:
        return Ask(
            f"Refund {args['amount']} EUR?",
            assignee="role:finance",
            component="refund-review",
            props={"amount": args["amount"]},
        )
    return None  # governance: irreversible asks


@tool(side_effects="irreversible", approval=refund_rule)
def refund(order: str, amount: int) -> str: ...
```

**Automatic.** The rule's answer is journaled with the call: a resumed run reads it instead of
asking the function again. A hook's `Deny` or `Ask` comes first (the function is not asked).

**On failure.** A function that returns anything else, or raises, fails the call: the model
reads why (`the approval function of refund returned 'yes': return None, True or Ask(...)`).

## Comments, and approving for the rest of the run

**What.** `agent.resume(..., comment="Fine this once; over 500 needs finance")` keeps the
reviewer's remark with the decision; `remember="run"` on an `approve` of a tool call approves
that tool's later calls in the same run without asking.

**When.** A comment whenever a decision needs its why (it reaches the feedback approval
patterns are learned from). `remember="run"` when a reviewer trusts the rest of this run with
the tool (a batch of refunds of one customer).

**Where.** `agent.resume`, the AG-UI resume entry (`"comment"`, `"remember"`), an A2A answer's
data part (`{"comment": ..., "remember": "run"}`), the reference inbox; every adapter and
`ReAct`. Way 2: `InterruptResolution(comment=, remember=)`.

**Automatic.** The comment is on the run record (`last_resolution`, agent-runs' resolution
history), on the `decision` event and the span attributes of the attempt that goes on after
it, and in the feedback record; a rejection's comment is what the model reads as why, when the
reject carries no `answer`. A remembered approval is kept in the run's journal: every later
call of that tool in the run — after a pause, a crash, on another worker — runs without asking,
with a `decision` event (`remembered: true`); another run asks again. It covers the tool's
later approvals asked of the same person or role (governance's; an approval function's or a
hook's `Ask` with the same `assignee`): an `Ask` that names someone else — finance, for a large
amount — still asks.

**On failure.** `remember="run"` on anything but an approval of a tool call is refused
(`ConfigurationError`); a comment over 4000 characters too.

## External results: `tool(external=True)`

**What.** A tool whose call is done outside the run — a person signing in an e-signature
system, a batch job, a human operator: the run pauses with the call, and the result given
from outside is what the model reads as the tool's output.

**When.** When the work takes as long as it takes and nothing in the process can do it.

**Where.** `@tool(external=True)` on every adapter and `ReAct` (the bridge); Way 2: pause with
a `Question(..., expects=...)` whose interrupt your code builds with the call, and hand the
answer to your framework as the tool's result.

**How.**

```python
@tool(side_effects="write", external=True)
def sign(contract: str) -> str:
    """Have a contract signed (a person signs it)."""  # never runs


paused = await agent.run("get c-7 signed", user="ada")  # paused on the call
await agent.resume(paused.run_id, result="signed by ada, 10:42")
```

**Automatic.** The interrupt is a `QUESTION` with the call in `tool_call` and, from the
function's return annotation, `expects` (here `{"type": "string"}`); governance still decides
first (an `irreversible` external tool is approved, then asked for its result). The result is
journaled like any call's output. The answer also comes as agent-runs' `ANSWER` resolution of
the interrupt (any client), or as an AG-UI or A2A answer.

**On failure.** A result that does not fit `expects` is refused at `resume`; a `reject` is an
error the model reads (`sign was not done: its result was refused`); a `cancel` ends the run.

## Telling people

**What.** When a run pauses for a person, every notifier is told: the question (redacted),
whose it is, by when, the call under approval, and where to answer.

**When.** Whenever people should not have to watch an inbox.

**Where.** Automatic for every agent of the harness: Slack with `SLACK_WEBHOOK_URL` (an
incoming webhook), email with `SMTP_URL` and `SMTP_FROM` (to the assignee when it is an
address, `user:ada@example.com`, else to `SMTP_TO`); your own with
`Harness(notifiers=[...])`: anything with `async notify(interrupt, link)`. The link is
`TRELLIS_INBOX_URL#<interrupt id>` when set ([configuration.md](configuration.md)).

```python
class Pager:
    async def notify(self, interrupt: Interrupt, link: str | None) -> None:
        await page(interrupt.assignee, interrupt.question, link)


h = Harness(notifiers=[Pager()])
```

**Automatic.** After the pause is recorded, in the background (the run is not held). A
notifier gets the interrupt redacted as everything leaving the process is (the question, the
payload, the props, the call's arguments; not whose it is). A sub-agent's question is told
once, as its parent's run's. A `notified` event says who was told.

**On failure.** Best-effort: a notifier that fails or takes over 10 s is a `warning` event on
the run (`notify_failed`, logged, counted `trellis.notifications{outcome="failed"}`), never a
failed run, and is not retried. For delivery that is retried and signed, for every run of the
tenant (wrapped or not), use agent-runs' webhooks (`run.paused`, `run.escalated`,
`run.finished`: [runs.md](runs.md#the-inbox), [blocks/runs.md](blocks/runs.md#webhooks)).

## The reference inbox

`h.serve_inbox(app, *, path="/inbox", identity=None)` serves a small page (static HTML and
JavaScript, no dependencies) that lists the paused runs of the agents wrapped by the harness
and answers them: labelled options (checkboxes with `multiple`), a form built from `expects`
with `ui_schema`'s `ui:widget`/`ui:title`/`ui:help`/`ui:placeholder`/`ui:order`, a table or a
diff, an approval with the call's arguments (approve, approve edited, reject, and "approve for
the rest of this run"), a comment, and cancel. A question naming a `component` is rendered by
your screen when the page has it: `window.trellisComponents = {"refund-review": (element,
props, interrupt, submit) => ...}`. It is a reference: copy it, or answer from your own screens
through its routes — `GET {path}/runs?assignee=` (the paused runs with their interrupts) and
`POST {path}/runs/{run_id}/resume` (`{"interrupt_id", "decision", "answer", "comment",
"remember"}`: checked first, `409` with why when it does not fit; then `202`, and the run goes
on in the background). `identity(request)` names the reviewer (else `anonymous`, with a
warning). Point `TRELLIS_INBOX_URL` at it and notifications link to each question.

## Testing: `trellis.testing.Reviewer`

```python
from trellis.testing import Decide, Reviewer

reviewer = Reviewer(
    {
        "refund": "approve",
        "Which plans?": ["basic", "pro"],
        "sign": "signed",
        "plan-picker": Decide("cancel"),
    }
)
result = await reviewer.run(agent, "settle acme", user="ada")  # every pause answered
```

An entry is found by the tool the interrupt asks about, its `component`, its question, then
`"*"`. For an approval: `"approve"`, `"reject"`, `"cancel"`, `True`/`False`, or a dict (the
edited arguments); otherwise the answer itself (an external tool's result too).
`Decide(decision, answer=None, comment=None, remember="once")` says it all; a function of the
interrupt may decide. `reviewer.answer(agent, result)` answers one pause, `settle` all of them
(at most 50), and `reviewer.resolution(interrupt)` is the `InterruptResolution` for Way 2's
`runs.resume`. An interrupt the script does not cover raises `LookupError` naming it;
`reviewer.answered` lists what was answered.

**Example.** [`examples/approvals.py`](../examples/approvals.py): an approval function (a small
refund approved by the rule, a large one asked on finance's screen), labelled options with
several picks, an external tool's result, answered by a `Reviewer`. Tests:
`tests/integration/test_approvals.py`, `test_questions.py` and `test_inbox.py` (every adapter,
Way 2), and against the services `tests/live/test_live_hitl.py`.

## Answering

```python
await agent.resume(interrupt_id, "answer", answer="ACME", reviewer="lee")
await agent.resume(interrupt_id, "approve", reviewer="cfo")
await agent.resume(interrupt_id, "edit", answer={"amount": 9000}, reviewer="cfo")
await agent.resume(interrupt_id, "reject", reviewer="cfo")
await agent.resume(interrupt_id, "reject", answer="over budget", reviewer="cfo")  # with a reason
await agent.resume(interrupt_id, "cancel", reviewer="cfo")
await agent.resume(interrupt_id, "approve", reviewer="cfo", comment="ok today", remember="run")
await agent.resume(run_id, result="signed by ada")  # an external tool's result; the run's id
```

The run's id answers whatever it waits on now. `reviewer` is required for a decision (not for
a `result=`).

What `ask` returns: the answer; `True`/`False` for approve/reject; the edited value for edit;
cancel ends the run `CANCELLED` (so does `agent.cancel(run_id, reason=...)`, which needs no
interrupt and stops a run whatever it is doing: [reliability.md](reliability.md#cancel)). A
rejected tool call is not run and the model reads that it
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
