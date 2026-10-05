# A2A: `trellis.harness.a2a.remote` (Way 2)

`remote(url, tenant=, user=)` calls another agent over A2A from any code, on any framework:
a `RemoteAgent` is an async callable, a message in and the remote agent's answer out. The
remote side is any A2A server: a wrapped agent's `serve_a2a`, or another vendor's.

Only calling is a block. *Serving* an agent over A2A (or AG-UI) is Way 1: the server records
the run, streams its events, replays its journal on resume and reads a person's answer as a
decision, which is the harness's pipeline. To serve an agent built on your own framework, wrap
the function that calls it ([surfaces.md](../surfaces.md); any async function is a target,
[functions.md](../frameworks/functions.md)).

## Install and set up

`remote` ships in the harness distribution with the `a2a` extra:

```bash
pip install -e '../agent-harness[a2a]'    # the A2A SDK; no Harness needed
```

It reads no environment: who the call is for is an argument.

```python
remote(url, *, tenant, user, thread=None, on_input=None, name=None, headers=None,
       timeout=120.0, client=None) -> RemoteAgent
```

| | |
|---|---|
| `url` | the remote agent's base URL; its card is at `{url}/.well-known/agent-card.json` |
| `tenant`, `user` | who the call is for, sent on the trusted-identity header (a harness-served agent takes the call when `tenant` is the one its key speaks for) |
| `thread` | the A2A context id: one conversation on both sides. `None`: each call its own |
| `on_input` | `on_input(question) -> answer`, sync or async: answers a remote question on the same task. Without it, a question is raised as `InputRequired` |
| `name` | the tool name in `spec` (else the card's; unsafe characters become `_`, at most 64) |
| `headers` | sent on every request (an edge's `Authorization`, say); they never replace the identity |
| `timeout`, `client` | the HTTP client a `RemoteAgent` opens (`timeout` seconds per exchange), or an `httpx.AsyncClient` of yours, which it never closes |

## The API

```python
from trellis.harness.a2a import InputRequired, remote

async with remote("https://planner.example/a2a", tenant="acme", user="ada") as planner:
    try:
        plan = await planner("deploy the shop")
    except InputRequired as asked:  # the remote agent asks something
        plan = await planner.reply(asked.task_id, input(asked.question))
```

* `await agent(message, *, message_id=None)`: a new task with `message` (text, or a JSON value
  as a data part); the answer is the task's `result` artifact (several artifacts as a list), or
  its text. `message_id` names the message (the same id again is the same message to a server
  that deduplicates; the harness's `a2a(url)` tool sends its call's idempotency key).
* `await agent.connect()` / `async with remote(...) as agent`: reads the card once (a call
  connects first when nothing has); `aclose()` closes the HTTP client it opened.
* `agent.card` is the remote `AgentCard`; `agent.spec` a contracts `ToolSpec` (`write`,
  `{"message": string}` in), for exposing it as a tool in any framework. Both after `connect`.
* A remote question: `on_input(question)` answers it and the call goes on. Without
  `on_input`, `InputRequired` is raised (`question`, `task_id`); the remote task keeps
  waiting, and `await agent.reply(task_id, answer)` continues it: the task's answer, or the
  next question. If `on_input` raises (a graph's `interrupt`), the remote task is cancelled and
  the exception goes on, so a resumed caller calls again.
* A task that ends `failed`, `rejected` or `canceled`, and a remote agent that cannot be
  reached, are a contracts `ToolError` (`source="a2a"`).

## As a tool of your framework

A LangGraph (LangChain) tool: the remote question becomes the graph's own `interrupt`, and
`Command(resume=...)` answers it (the tool runs again, and the interrupt returns the answer):

```python
from typing import Any

from langchain_core.tools import tool
from langgraph.types import interrupt

planner = remote(url, tenant="acme", user="ada", on_input=interrupt)


@tool
async def plan(message: str) -> Any:
    """Plan a deployment with the remote planner agent."""
    return await planner(message)
```

An OpenAI Agents `function_tool` or a Claude in-process MCP tool is the same: an async
function that awaits the `RemoteAgent`; `planner.spec` carries the card's name, description and
schema for a framework that builds tools from one. Wrap the tool with `governed` when the call
should be governed ([governance.md](governance.md)): a remote agent is a `write` tool.

## Behaviour

* **Identity.** Every request carries `tenant` and `user` on the trusted-identity header
  (`x-trellis-identity`) and declares the extension. A harness-served agent refuses a tenant
  that is not its key's.
* **One thread.** With `thread`, every call is in that A2A context: the remote agent sees one
  conversation, and so does its memory.
* **Errors.** A failed remote task is a `ToolError` your framework's model can read; an
  unreachable agent too. Nothing is retried for you: a task is not idempotent.

## With Way 1

A wrapped agent calls remote agents with `a2a(url)` in its tools, which is built on `remote`:
the identity and the thread are the calling run's, and `on_input` is the run's own `ask`, so a
remote question pauses the calling run in agent-runs and the resumed run answers it
([surfaces.md](../surfaces.md#calling-a2a-agents-a2aurl--namenone)). A wrapped agent served
with `serve_a2a` is a remote agent your code calls with `remote` ([mixing.md](mixing.md)).

[`examples/a2a_agents.py`](../../examples/a2a_agents.py) serves a wrapped agent over A2A, calls
it as another agent's tool, and calls it from plain code with `remote()`.
