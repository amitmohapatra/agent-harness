# Memory: `trellis.memory` (Way 2)

`trellis.memory` is the memory service's Python SDK (`MemoryClient`). The service keeps what
agents and people said, stated and uploaded, and pushes back the context a turn needs. As a
block, your framework owns the prompt and the loop, and each turn your code does three things:

1. reads the **context** for the question into the prompt;
2. **records** the turn: the conversation, each tool call;
3. sends **feedback** once it knows how the run went.

Way 1 makes these calls on every run of a wrapped agent, plus the pull tools, the tool hints,
the grounding check and approvals learned ([memory.md](../memory.md)).

## Install and set up

```bash
pip install -e ../agent-memory-service/sdk/python    # pip trellis-memory; imports as trellis.memory
```

It depends on `httpx` and `pydantic` only: no framework, no harness.

```python
from trellis.memory import MemoryClient

memory = MemoryClient()  # MEMORY_URL (default http://localhost:8080) and TRELLIS_API_KEY
...
await memory.aclose()  # or: async with MemoryClient() as memory
```

Use one client per process. Against the local stack's development key (`dev-key`) no tenant is
needed: it acts in the development tenant, `default`.

## One turn

Every call is made in a **scope**: who the turn is for and which agent run it is.
`bind(...)` names the tenant (for a platform key), the user and the thread; `.agent(id,
agent_run_id=...)` makes it one run of one agent:

```python
scope = memory.bind(tenant_id="acme", user_id="ada", thread_id="chat-42").agent(
    "support", agent_run_id=run_id
)

# window=False: your framework keeps the history
pushed = await scope.context(question, window=False)
answer = await my_agent(system=pushed.rendered, user=question)  # a node, an Agent, a query()
await scope.history.add([("USER", question), ("ASSISTANT", answer)])

await scope.feedback("run", run_id, "confirm", source="system")  # or "reject"
```

* `context(query, *, token_budget=None, tools=None, window=True)` returns a `PromptContext`:
  `rendered` (prompt-ready text, memories cited by handle, `[m1]`), `bundle_id` (what the
  answer is graded against), `evidence_status` (`INSUFFICIENT` means: say you do not know) and,
  with `tools=` (your tool names), the tools that fit the task, each with a confidence.
  `window=False` leaves the recent conversation out when your framework keeps the thread
  (a LangGraph checkpointer, an OpenAI Agents session, a Claude session).
* Where the context goes is your framework's: a leading system message for a LangGraph graph,
  a `system` input item for OpenAI Agents' `Runner.run`, appended to Claude's `system_prompt`.
* `history.add([...])` appends to the thread's transcript, `(role, text)` pairs or messages.
* `feedback("run", run_id, verdict, source=...)` is how the run went: `confirm` or `reject`
  from your code (`source="system"`), a person's verdict (`source="human"`, the default) waits
  for the tenant administrator's review before it changes what memory learned.

## Recording tool calls

`record_tool(tool, args, output=, status=)` records one call the run made; the service learns
which tools fit which task from it (the tool hints). `async with scope:` makes the scope the
current one, so a tool can record itself without being handed the scope:

```python
from trellis.memory import current_context


async def create_po(sku: str, qty: int) -> str:
    po = await erp.order(sku, qty)
    if (scope := current_context()) is not None:  # the scope `async with scope:` entered
        await scope.record_tool("create_po", {"sku": sku, "qty": qty}, output=po)
    return po


async with scope:
    state = await graph.ainvoke(...)  # every recorded call lands in this run's scope
```

The recipes wrap each tool in a small `recorded(fn)` decorator that does this for any function.

## More than one turn

| Verb | |
|---|---|
| `remember(content, memory_type=, visibility=)`, `update(id, content)`, `forget(id)` | state, supersede or forget one memory |
| `search(query, kinds=, limit=)` | ranked evidence, without the rendered bundle |
| `verify(answer, bundle_id=, run_id=)` | per-claim grounding of an answer against the context it was given (what `grounding()` scores, [evaluation.md](evaluation.md)) |
| `tool_hints(task, available=)` | the tools that fit, best first, with the arguments found and the missing ones |
| `agent_tools()`, `call_agent_tool(name, args)` | the memory tools a model calls itself (pull mode): give them to your framework as tools |
| `profile()`, `profile.edit(block, new, ...)` | the pinned profile blocks |
| `feedback(record)` | a contracts `Feedback` as it is, e.g. `resolution.to_feedback(interrupt, ctx)` |

The SDK's [README](https://github.com/amitmohapatra/agent-memory-service/blob/main/sdk/python/README.md)
has every verb and the service's
[USAGE](https://github.com/amitmohapatra/agent-memory-service/blob/main/docs/USAGE.md) which
call fits which job.

## Behaviour

* **Retries.** Reads, and writes that carry an idempotency key (the verbs that write send one),
  are sent again on `429`, `502`, `503`, `504`, timeouts and dropped connections, after the
  service's `Retry-After` (at most 30 s) or a full-jitter backoff.
* **Circuit breaker.** After 5 calls in a row fail for want of the service, the client fails
  fast with `CircuitOpenError` for 30 s, then probes. A turn should answer without memory
  rather than fail:

  ```python
  from trellis.memory import DependencyUnavailableError

  try:
      pushed = await scope.context(question)
  except DependencyUnavailableError:  # CircuitOpenError is one
      pushed = None  # answer without memory this turn
  ```

* **Errors.** RFC 9457 problems, one exception per `code`: `AuthenticationError`,
  `AuthorizationError`, `NotFoundError`, `ConflictError`, `ValidationError`,
  `RateLimitedError`, `DependencyUnavailableError`, `TimeoutError`, each with `status`,
  `retryable`, `request_id`.
* **Tenancy.** A tenant key names its tenant; a platform key needs `bind(tenant_id=...)`. The
  service decides per scope what a write may touch: a refused write is an
  `AuthorizationError`.
* **Not queued.** Each call is awaited where you make it. Record in the background (your
  framework's background task) when the turn must not wait; the harness's spool and replay are
  Way 1 only.

## With Way 1

A wrapped agent's memory is the same service, through the same SDK, in the same scopes
(tenant, user, agent, run, thread): the harness pushes the context into the framework's input,
adds the pull tools, records the transcript, every tool call and the outcome, and queues those
writes in the background ([memory.md](../memory.md)). Inside a wrapped run,
`trellis.current().memory` is this SDK's `MemoryContext` already bound to the run. Your code
and wrapped agents share what the service learns: the same user's profile, memories and
documents, and the tool catalog governance reads ([mixing.md](mixing.md)).

Recipes: [LangGraph](langgraph.md), [OpenAI Agents SDK](openai-agents.md),
[Claude Agent SDK](claude-agent-sdk.md).
