# Pauses: `ask`, approvals, `resume`

## Asking

**What.** `trellis.current().ask(...)` asks a person and waits: the run pauses
(`Result.status == PAUSED`, `Result.interrupt` a contracts `Interrupt`) and, on resume, the call
returns the answer.

**When.** When only a person knows: a choice, a value, a correction, a sign-off. A tool call's
approval needs no `ask` (governance, an approval rule in a hook, below).

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
renders it with the props as they are (your AG-UI client from the interrupt's `metadata`, your
own inbox from `h.inbox`'s interrupts: [below](#an-inbox-of-your-own)), any other renders
`ui`. Everything goes on the
contracts' `Interrupt` as it is. An answer carries option values, never labels; with `form=`
it is read back into the model.

The whole signature: `await trellis.current().ask(question, *, expects=None, form=None,
table=None, diff=None, options=None, multiple=False, ui_schema=None, component=None,
props=None, assignee=None, deadline=None, escalate_to=None)`.

**Automatic.** The question is the run's user's to answer (`assignee="user:<user>"`) unless
`assignee` names someone else (`user:…`, `role:…`); it is in their inbox
(`h.inbox(assignee)`), and agent-runs' webhooks tell whoever waits ([below](#telling-people)). `escalate_to`
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

An approval (an `irreversible` tool, a catalog `approve_when` that holds, a hook's `Ask`) is
the same pause with `reason=APPROVAL` and the tool call attached.

## Approval rules in code: a `before_tool` hook

**What.** Your rule for when a call needs a person, by its arguments: a `before_tool` hook
([hooks.md](hooks.md)) returns `Ask(question, assignee=None, component=None, props=None)` for
the calls that need one — a person approves it first, asked that, on that screen — and `None`
for the rest, which governance decides as for any tool.

**When.** For a rule only your code can say: an amount, a customer, the time of day. A rule an
administrator owns is the catalog's `approve_when` ([governance.md](governance.md)), which also
decides by the call's arguments (`amount > 20`).

**Where.** `h.wrap(..., hooks=[...])` or `Harness(hooks=[...])`, every adapter (the bridge) and
`ReAct`; Way 2: `governed(fn, gov, ..., hooks=[...])`, whose `on_ask(decision)` gets the
decision with the `Ask`'s `assignee`, `component` and `props`.

**How.**

```python
class RefundRule(Hooks):
    async def before_tool(self, call: ToolCall) -> Ask | None:
        amount = call.args.get("amount", 0) if call.tool == "refund" else 0
        if amount <= 20:
            return None  # small: governance decides (a write tool runs, announced)
        if amount > 500:
            return Ask(
                f"Refund {amount} EUR?",
                assignee="role:finance",
                component="refund-review",
                props={"amount": amount},
            )
        return Ask(f"Refund {amount} EUR?")


@tool(side_effects="write")
def refund(order: str, amount: int) -> str: ...


agent = h.wrap(target, id="refunds", tools=[refund], hooks=[RefundRule()])
```

A hook asks; it never approves a call governance asks about. So the tool's `side_effects`
decide what runs unasked: to approve small amounts without a person, declare the tool
`"write"` (governance runs it, announced) and let the hook ask above the threshold, as above —
or let the catalog's `approve_when` ask by the arguments. An `irreversible` tool, or one whose
`approve_when` holds, is always asked about.

**Automatic.** The hook's verdict is journaled with the call: a resumed run reads it instead of
asking the hook again. An approval remembered for the run (`remember="run"`, below) covers the
later calls a hook asks of the same `assignee`.

**On failure.** A hook that raises fails the call (the framework sees the error).

## Comments, and approving for the rest of the run

**What.** `agent.resume(..., comment="Fine this once; over 500 needs finance")` keeps the
reviewer's remark with the decision; `remember="run"` on an `approve` of a tool call approves
that tool's later calls in the same run without asking.

**When.** A comment whenever a decision needs its why (it reaches the feedback approval
patterns are learned from). `remember="run"` when a reviewer trusts the rest of this run with
the tool (a batch of refunds of one customer).

**Where.** `agent.resume`, the AG-UI resume entry (`"comment"`, `"remember"`), an A2A answer's
data part (`{"comment": ..., "remember": "run"}`), an inbox of your own; every adapter and
`ReAct`. Way 2: `InterruptResolution(comment=, remember=)`.

**Automatic.** The comment is on the run record (`last_resolution`, agent-runs' resolution
history), on the `decision` event and the span attributes of the attempt that goes on after
it, and in the feedback record; a rejection's comment is what the model reads as why, when the
reject carries no `answer`. A remembered approval is kept in the run's journal: every later
call of that tool in the run — after a pause, a crash, on another worker — runs without asking,
with a `decision` event (`remembered: true`); another run asks again. It covers the tool's
later approvals asked of the same person or role (governance's, or a hook's `Ask` with the
same `assignee`): an `Ask` that names someone else — finance, for a large amount — still
asks.

**On failure.** `remember="run"` on anything but an approval of a tool call is refused
(`ConfigurationError`); a comment over 4000 characters too.

## Results from outside: `ask` in the tool

**What.** A tool whose result comes from outside the run — a person signing in an e-signature
system, a batch job, a human operator: the tool asks (`trellis.current().ask(...)`), the run
pauses inside it, and the answer given from outside is what the tool returns and the model
reads as its output.

**When.** When the work takes as long as it takes and nothing in the process can do it.

**Where.** Any tool of any adapter and `ReAct` (as `ask` is, above); Way 2: pause with a
`Question(..., expects=...)` and hand the answer to your framework as the tool's result.

**How.**

```python
@tool(side_effects="write")
async def sign(contract: str) -> str:
    """Have a contract signed (a person signs it)."""
    return await trellis.current().ask(
        f"Signed {contract}?",
        expects={"type": "string"},  # what the tool returns: TypeAdapter(str).json_schema()
        component="e-signature",  # the system that answers, and what it needs
        props={"contract": contract},
    )


paused = await agent.run("get c-7 signed", user="ada")  # paused inside sign
await agent.resume(paused.run_id, "answer", answer="signed by ada, 10:42", reviewer="e-signature")
```

**Automatic.** The interrupt is a `QUESTION` with `expects` the result's schema and, when the
tool names them, the `component` that answers it and its `props` (the call's arguments it
needs). Governance decides on the call first (an `irreversible` tool is approved, then asks for
its result). The answer is journaled with the run like any `ask`'s, and the call's output like
any call's. The answer also comes as agent-runs' `ANSWER` resolution of the interrupt (any
client), or as an AG-UI or A2A answer.

**On failure.** A result that does not fit `expects` is refused at `resume`; a `reject` returns
`False` to the tool, which raises to make it an error the model reads
(`raise ToolError("the signature was refused", source="tools")`); a `cancel` ends the run.

## Telling people

**What.** When a run pauses for a person, whoever waits is told — the question, whose it is,
by when, the call under approval, and the run to answer — by agent-runs' webhooks: the tenant
subscribes once to `run.paused` (and `run.escalated`, `run.finished`), and every pause of every
run of the tenant, wrapped or not, is POSTed to its receiver, signed. The receiver posts to a
channel, mails, pages.

**When.** Whenever people should not have to watch an inbox.

**Where.** With agent-runs (`RUNS_URL`): `runs.webhooks.create(url, [WebhookEvent.PAUSED])`
([blocks/runs.md](blocks/runs.md#webhooks)). Without it (`LocalRuns`, no `RUNS_URL`) there are
no webhooks, and nothing tells anyone on its own: a run hook that sees the pause
(`on_run_end` with a `PAUSED` result, [hooks.md](hooks.md)) is the way in process.

**How.**

```python
from trellis.runs import WebhookEvent
from trellis.runs.webhooks import SIGNATURE_HEADER, parse_delivery, verify_signature

hook = await h.runs.webhooks.create("https://ops.example/hooks/trellis", [WebhookEvent.PAUSED])
keep_secret(hook.secret)  # shown in this answer only


async def trellis_hook(request: Request) -> Response:  # your receiver (Starlette or FastAPI)
    body = await request.body()
    if not verify_signature(SECRET, request.headers.get(SIGNATURE_HEADER), body):
        return Response(status_code=401)
    run = parse_delivery(body).data.run  # a RunSummary; awaiting is the Interrupt
    asked = run.awaiting
    await post(f"{asked.question} (for {run.assignee}) https://ops.example/runs/{run.run_id}")
    return Response(status_code=204)
```

In process, without agent-runs:

```python
class Telling(Hooks):
    async def on_run_end(self, run: Runtime, result: Result) -> None:
        if result.status is RunStatus.PAUSED and run.parent is None:  # not a sub-agent's
            await page(result.interrupt.assignee, result.interrupt.question, run.run_id)


h = Harness(hooks=[Telling()])
```

**Automatic.** agent-runs writes the delivery with the pause (one transaction) and its ticker
sends it within a tick (5 s), retried, with the same `event_id` on every retry (drop repeats by
it). It carries the run's summary as agent-runs keeps it: `run_id`, `agent_id`, `assignee`,
`deadline` and the interrupt itself (`awaiting`: the question, `options`, `expects`, `component`
and `props`, the call under approval in `tool_call`) — as asked, not redacted: redact in the
receiver what must not reach a channel. A sub-agent's question pauses its own run and its
parent's, and both are delivered: answer the parent's (a receiver that reads the run with
`runs.get` skips one with a `parent_run_id`; a run hook skips `run.parent`).

**On failure.** A receiver that answers `408`, `429` or `5xx`, or cannot be reached, is retried
(7 attempts, at most 10 min apart), then kept as a dead delivery to redeliver
(`runs.webhooks.redeliver`); any other answer is final ([blocks/runs.md](blocks/runs.md#webhooks)).
A run hook runs in process and is not retried: what it raises is logged, and the run's outcome
stands.

## An inbox of your own

**What.** The paused runs waiting on a person, listed and answered from your own screen with
two calls: `await h.inbox(assignee)` (the paused runs of the tenant waiting on that `user:…` or
`role:…`, or on anyone with no argument — each a `RunSummary` with the interrupt it waits on in
`awaiting`: the question, `options`, `multiple`, `expects`, `ui_schema`, `component` and
`props`, the call under approval in `tool_call`) and `agent.resume(...)` ([below](#answering)).

**When.** Whenever people answer outside the chat that started the run: an approvals queue, a
back office, a review screen of your design.

**Where.** Any process of the deployment (with agent-runs, any replica: `h.inbox` is
`runs.iterate(status=PAUSED, assignee=...)`); Way 2: `RunsClient.iterate(...)` and
`runs.resume(...)` ([runs.md](runs.md#the-inbox), [blocks/runs.md](blocks/runs.md#the-inbox)).
A chat client answers its own runs with the AG-UI resume entry, `comment` and `remember`
included ([surfaces.md](surfaces.md)).

**How.** A route of your own on any web framework (FastAPI here):

```python
@app.get("/inbox")
async def listing(assignee: str | None = None) -> list[dict]:
    return [
        {"run_id": s.run_id, "interrupt": s.awaiting.awaiting()}  # render it as asked
        for s in await h.inbox(assignee)
        if s.agent_id in h.agents and s.awaiting is not None
    ]


@app.post("/inbox/{run_id}")
async def answer(run_id: str, request: Request) -> dict:
    given = await request.json()
    record = await h.runs.get(run_id)
    if record is None or record.agent_id not in h.agents:
        raise HTTPException(404, f"no run {run_id}")
    try:
        result = await h.agents[record.agent_id].resume(
            run_id,  # the run's id answers what it waits on now
            given["decision"],
            answer=given.get("answer"),
            comment=given.get("comment"),
            remember=given.get("remember", "once"),
            reviewer=await who_is(request),  # your identity
        )
    except ConfigurationError as exc:  # it does not fit: the run keeps waiting
        raise HTTPException(409, str(exc)) from exc
    return {"status": result.status.value}
```

**Automatic.** The answer is checked before anything is recorded (as any `resume`), and a run
of the queue goes back to it for a worker (`QUEUED`) while one started in process continues in
the call (run it in a background task when the reviewer should not wait). A sub-agent's
question is listed once, on its parent's run. With agent-runs, who may answer is its rule
([below](#answering)).

**On failure.** An answer that does not fit is a `ConfigurationError` saying why; an answer to
a run that waits on another interrupt, or is not paused, too.

## Testing: `trellis.testing.Reviewer`

```python
from trellis.testing import Decide, Reviewer

reviewer = Reviewer(
    {
        "refund": "approve",
        "Which plans?": ["basic", "pro"],
        "e-signature": "signed by ada",  # a result from outside, by its component
        "plan-picker": Decide("cancel"),
    }
)
result = await reviewer.run(agent, "settle acme", user="ada")  # every pause answered
```

An entry is found by the tool the interrupt asks about, its `component`, its question, then
`"*"`. For an approval: `"approve"`, `"reject"`, `"cancel"`, `True`/`False`, or a dict (the
edited arguments); otherwise the answer itself (a result from outside the run too).
`Decide(decision, answer=None, comment=None, remember="once")` says it all; a function of the
interrupt may decide. `reviewer.answer(agent, result)` answers one pause, `settle` all of them
(at most 50), and `reviewer.resolution(interrupt)` is the `InterruptResolution` for Way 2's
`runs.resume`. An interrupt the script does not cover raises `LookupError` naming it;
`reviewer.answered` lists what was answered.

**Example.** [`examples/approvals.py`](../examples/approvals.py): an approval rule in a hook (a
small refund runs unasked, a large one is asked on finance's screen), labelled options with
several picks, a result from outside the run (an `ask` in the tool), answered by a `Reviewer`. Tests:
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
await agent.resume(run_id, "answer", answer="signed by ada", reviewer="e-signature")  # the run's id
```

The run's id answers whatever it waits on now. `reviewer` is required.

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
