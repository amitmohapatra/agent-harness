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
resume, the call returns the answer. The UI hint and the reason follow from the arguments:

| Arguments | `ui` | `reason` |
|---|---|---|
| `options=` | `choice` | `CHOICE` |
| `table=` | `table` (`REVIEW` with `expects=`) | `QUESTION` |
| `ui="diff", expects=` | `diff` | `REVIEW` |
| anything else | `form` | `QUESTION` |

A table of up to 50 rows travels in `payload`; a larger one is stored as an artifact and
referenced by `payload_ref` (served by `serve_chat`). `escalate_to` needs a `deadline`; agent-runs
escalates or times out the run when it passes.

An approval (an `irreversible` tool, an `approve` rule) is the same pause with
`reason=APPROVAL` and the tool call attached.

## Answering

```python
await agent.resume(interrupt_id, "answer", answer="ACME", reviewer="lee")
await agent.resume(interrupt_id, "approve", reviewer="cfo")
await agent.resume(interrupt_id, "edit", answer={"amount": 9000}, reviewer="cfo")
await agent.resume(interrupt_id, "reject", reviewer="cfo")
await agent.resume(interrupt_id, "cancel", reviewer="cfo")
```

What `ask` returns: the answer; `True`/`False` for approve/reject; the edited value for edit;
cancel ends the run `CANCELLED`. The interrupt id names its run, so nothing else is needed; a
resume must answer the interrupt the run currently waits on.

## How a run continues

* **LangGraph with a checkpointer** (Deep Agents with one included): `ask` is LangGraph's
  `interrupt`, and the resume is `Command(resume=...)` — the graph continues in place. A
  graph's own `interrupt(value)` is surfaced as a question and resumed with the raw answer;
  without a checkpointer it cannot be resumed and the run fails saying so.
* **OpenAI Agents `needs_approval` tools**: the SDK's own pause; the resume approves or rejects
  on its `RunState` and continues it.
* **Everything else** re-runs from the input as the next attempt, with the **journal**:
  answers already given return where their question is asked, and tool calls already made
  return their recorded outputs instead of running again. Entries are keyed by content (the
  question; the tool and its arguments) and consumed in order.

The journal is the run's checkpoint: the pause stores it with the run (`RunRecord.checkpoint`
in agent-runs, cleared when the run ends), and whichever process or worker resumes the run
reads it back with the resolution, so a resume elsewhere repeats no question and no tool
call.

A run started with `run`/`stream` continues in the process that calls `resume`; a queued run
(`start`, a schedule) goes back to the queue and a worker continues it (`Result.status ==
QUEUED`).
