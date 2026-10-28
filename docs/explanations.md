# Explanations: questions about an earlier answer

"What steps did you take?", "where did this come from?", "how do you know?", "how sure are
you?", "did anyone approve this?", "how long did it take?", "which tools did you use?" — a
follow-up about an answer the agent already gave is answered from that run's **record**, which
the harness keeps, and each audience is shown only what it may see of it.

- [What a run's record holds](#what-a-runs-record-holds)
- [Sources: where a result came from](#sources-where-a-result-came-from)
- [What each audience sees](#what-each-audience-sees)
- [The two arms](#the-two-arms)
- [The model's note about memory](#the-models-note-about-memory)
- [Limits](#limits)

## What a run's record holds

Nothing new is collected: the record is what the harness keeps already, put together when the
run ends.

| Part | Where it comes from |
|---|---|
| the steps: each tool call, its arguments, how it ended, what it returned (a preview) and how long it took | the run's trajectory (`Journal.trajectory`), the same one the evaluators read |
| each step in plain words | the tool's own description, its first sentence, without any tool's name |
| the sources each step's result came from | the tool, [below](#sources-where-a-result-came-from) |
| the approvals: the tool, the decision, whom it was asked of, who decided, their comment, when | each person's answer to an approval, filed in the journal as the run resumes |
| the memories its context showed, and how far they covered the question | the pushed memory context (`memory_ids`, `evidence_status`) |
| what it spent | the run's budget accounting |
| the question, the answer, how it ended, how long it took (working time, and in all) | the run store's record of the run (`RunRecord`) |

The journal is the run's checkpoint, so the record survives a pause and a worker crash with it.
When the run ends it is kept: with agent-runs (`RUNS_URL`) in the run's event log, as a `record`
custom event any replica reads; in process, in the agent, as long as the in-process store keeps
finished runs.

## Sources: where a result came from

A tool says where its result came from the way LangChain's tools do: `content_and_artifact` —
it returns the content the model reads and an artifact, its `trellis.Citation` (`name`, `date`,
`link`) or a list of them. The model never reads the artifact; the run keeps it with the call.

```python
import json

from trellis import Citation, tool


@tool(side_effects="read", response_format="content_and_artifact")
def stock_quote(symbol: str) -> tuple[str, Citation]:
    """The latest closing quote of a stock symbol."""
    quote = {"symbol": symbol, "close": 123.45}
    return json.dumps(quote), Citation(
        name="Nasdaq Last Sale feed", date="2026-10-27", link="https://www.nasdaq.com/acme"
    )
```

A LangChain tool of a `ReAct` or `create_agent` graph (`@tool(response_format=
"content_and_artifact")`) is read the same way: an artifact that is a `Citation` (or a list, or
their JSON) is its source; any other artifact names none. A `ReAct` run's answer carries its
sources — `result.sources`, those its tools returned, each once — and never a source the model
named itself: a model that writes "says Reuters" when no tool returned Reuters adds nothing to
`result.sources`, and the record never lists it.

## What each audience sees

The view is rendered by the harness from the record, never written by the model. Which one is
the harness key's role, as agent-runs and the memory service read it: an `admin` or `platform`
key administers the tenant and gets the administrator's view; any other key (a `service` key,
what a deployment acts for its users with) gets the end user's.

Nothing is said twice in the prompt: when the run's conversation window already carries the
earlier turn (its question, its answer, its tools' results), the view leaves them out — the
end user's its "Earlier question" and "Answer given" lines, the administrator's the run's input
and output and each step's output.

**The end user (client view)** — the steps in plain words, the sources with their name, date and
link (only those the run's tools returned) and how old each was at the answer, the answer tied
to the steps it rests on, the honest limits, how far their saved notes covered the question, the
notes of **theirs** the run was shown, how long it took, and the approvals by role:

```text
Earlier question: "What did ACME close at yesterday?"
Answer given: "ACME closed at 123.45 USD."
When: 2026-10-28 18:04 UTC. It took 2.1 seconds of work.
What I did (1 step(s)):
1. The latest closing quote of a stock symbol. — from Nasdaq Last Sale feed, 2026-10-27, https://www.nasdaq.com/acme
Sources:
- Nasdaq Last Sale feed, 2026-10-27, https://www.nasdaq.com/acme (1 day(s) before the answer)
Basis: the answer rests on what step(s) 1 returned; step(s) 1 named their source.
Your saved notes I used:
- I hold 40 ACME shares in my retirement account.
Memory: your saved notes and history covered the question.
Approvals:
- a reviewer (finance) approved step 1.
Limits: it rests on a single source.
```

Never in it, whatever the record holds: a tool's or function's name (one named as a plain word,
`order`, reads as the word) or its arguments, a raw output, another person's or tenant's data
(another person's memory, even one shared with the workspace, is left out), the system prompt,
a governance rule, a model's name, tokens, cost, an id, an error trace, a score — and the
reviewer's identity (their role, when the approval was asked of a `role:`, is said). A test
holds every one of these against a record full of them (`tests/unit/test_explain.py`).

**The administrator (admin view)** — everything, as JSON: the run's record from the run store
(ids, user, input, output, status, timing), every step with its tool, arguments, output preview,
error class and seconds, the sources, the approvals with the reviewer and their comment, what
the run spent, and every memory it was shown with its owner and visibility.

### How the agent reads it

In the [`lookup` arm](#the-two-arms) a run of a conversation is offered one more tool,
`earlier_answers` (`back`: 1 for the latest answer before this question, 2 the one before):

- read-only, and governed as a read (the catalog's word on it applies as on any tool);
- scoped to the run's own conversation: the same tenant, user, agent and thread, runs that
  ended, never a child run's — another person's runs in a thread of the same name are not
  "earlier answers";
- never narrowed away by the tool hints, and not one of their candidates;
- what it returns is the view above; with no earlier answer, or no record of it, it says so —
  "I have no record of the steps behind that answer, so I do not know them."

A run with no conversation (no `thread=`) is not offered it.

## The two arms

A follow-up about an earlier answer needs that answer's calls and results. Two ways give them to
the model, and each costs differently, so both are measured — against each other and against
neither — and **neither is on until the measurements decide**. They exist only to be measured:
the release keeps exactly one of them, whichever the data supports, and the other's code goes.

| Arm | The conversation window | What a follow-up reads |
|---|---|---|
| neither (what an agent does) | the memory service's own: whole turns, newest first, within its default budget (2,000 tokens), the newest turn whole even past it | the window |
| `sized` | asked at a share of the model's window, capped (`conversation_token_budget`, trellis-memory 0.6); a request over the model's window keeps the newer older turns that fit (LangChain's `trim_messages`: the last messages from a user's message on, so a call is never parted from its result, the leading instructions kept, the oldest kept message cut to fit) instead of none; older tool results cleared by `ContextEditingMiddleware` as always | the window |
| `lookup` | the service's own, as neither | the window, and the record on demand (`earlier_answers`) |

An arm is taken by agents wrapped (and `ReAct` graphs built) inside
`trellis.harness.explain.measured(...)` — a measurement, as `react.defaults_on` is, not an option
of `h.wrap`:

```python
from trellis.harness.explain import measured

with measured("lookup"):
    agent = h.wrap(ReAct("You answer stock questions.", "gpt-4o", tools=[stock_quote]), id="stock")
```

A memory service older than the `conversation_token_budget` field refuses it; the harness then
asks for the window as before, from then on in that process (said once in the log).

The benchmark's `explain` suite measures the three (`python -m benchmark.cost --suite explain`,
[evaluation.md](evaluation.md#cost-benchmark)): conversations whose first turn runs tools with
distinctive sources — a short result, a long one (about 10,000 characters), twelve pages of a
tape (twelve calls) — and whose second asks each kind of question above, scored against the
first run's actual record (the source named, the step count, the time, the approvals, the
user's own note, "I do not know" where the record holds no answer), the end user's view and
answer checked for every never-see item; and ordinary follow-ups as controls. Per arm it reports
that accuracy, how often the record was looked up when it should have been (and, on a control,
when it should not), and the tokens, cached share, model calls and seconds of every turn — what
the controls must not raise. Run it with a service key (the end user) and with an admin key
(`--admin`, `TRELLIS_ADMIN_API_KEY`, memory on).

## The model's note about memory

The memory service's `evidence_status` says what **long-term memory** holds for the question —
never the conversation. When it holds nothing (`INSUFFICIENT`) or only part (`INCOMPLETE`), the
model is told so, and told that this conversation — its own earlier tool calls and their results
included — is its to use; if the answer depends on something the user said that is not there, it
says it does not know. (Told before that memory held nothing, a model asked where its last answer
came from answered "I do not know" with the call and its result in front of it.)

## Limits

- A source is what a tool says: a tool that returns no artifact (an MCP tool, an OpenAPI
  operation, a sub-agent) names none, and its step reads "no step returned a source".
- `result.sources` is a run's answer in process (`agent.run`, `agent.resume`); a queued run's
  `RunHandle.result()` reads the run store's record, which carries none — its record holds them.
- A run that ended before this release has no record: its question, answer and time are still
  said, its steps are not.
- The plain words of a step are its tool's description: a description written for the model
  reads as written.
- "Basis" ties the answer to the steps whose results it rests on; the model's own reasoning
  between them is not recorded.
