# Surfaces

Two ways in, one way out. `serve_chat` puts an agent in front of a person's chat UI (AG-UI over
server-sent events); `serve_a2a` publishes it to other agents (A2A JSON-RPC); `a2a(url)` makes
another agent one of this agent's tools. Each surface runs the agent through the same pipeline
as `agent.run`: the run record, memory, tools, approvals and traces are the same whichever way
a run came in — and whichever framework the agent is built with ([framework pages](README.md#which-target)). Both need an extra: `[agui]`
(FastAPI) or `[a2a]` (the A2A SDK and FastAPI).
The code is `trellis.harness.agui` (the AG-UI server) and `trellis.harness.a2a` (`server`, and
`client`, the A2A client the `a2a(url)` tool is built on).

**The servers are Way 1.** `serve_chat` and `serve_a2a` serve a *wrapped* agent: `h.wrap(x)`
first, then `agent.serve_chat(app)` / `agent.serve_a2a(app, url)`. A server has to record the
run's start, pause and finish, stream its events in the contracts grammar, replay its journal
on resume and read a person's answer as a decision — that is the harness's pipeline, and an
interface that let unwrapped code do it would be a second runtime. So a team that keeps its
own framework and wants these servers wraps the part it serves: any async function is a target
([frameworks/functions.md](frameworks/functions.md)), so the function that calls your graph,
your OpenAI Agents runner or your own loop is enough. Otherwise use the protocols' own SDKs
(AG-UI's, the A2A SDK) with the Trellis blocks inside. *Calling* another agent needs none of
this: `remote(url, tenant=, user=)` works from any code ([blocks/a2a.md](blocks/a2a.md), Way 2).

## AG-UI: `agent.serve_chat(app, *, path="/agui", identity=None)`

| Route | |
|---|---|
| `POST {path}/run` | An AG-UI `RunAgentInput`. A new run (the client's `runId`, or one the harness names) executes in the background and its events stream as SSE, each with an `id:` numbered per run. A `resume` entry answers the interrupt a paused run on the thread waits on: `payload` is the answer (`true`/`false`/edited arguments for an approval), `status: "cancelled"` cancels, `decision` names a contracts decision outright. |
| `GET {path}/runs/{run_id}/events` | Reconnect: the run's events after `Last-Event-ID` (or `?after=`), then live until it finishes. Only the run's own user sees it. |
| `GET {path}/runs/{run_id}/artifacts/{artifact_id}` | Data the interrupt the run waits on carries by reference (`payload_ref`: a large `ask` table or diff), read from agent-runs. |

What the agent is asked: the latest `user` message's text, or the input's `state` when there is
none. The run keeps going when the client disconnects. Each run's events are buffered (its last 2048; past 256
runs the least recently used finished run is dropped, never a live one), and a resumed run keeps
counting where its first attempt stopped, so warnings from background writes after the finish
are still there on replay. A pause arrives as `RUN_FINISHED` with an `interrupt` outcome
(`interrupts[0]`: `id`, `reason` lower-cased, `message` the question, `toolCallId`,
`responseSchema` = `expects`, and `payload`/`payload_ref`/`tool_call` under `metadata`); a
failure as `RUN_ERROR`; a cancellation as `RUN_FINISHED` with a `cancelled` outcome.

How a resume entry becomes a decision: `decision` wins when present; `status: "cancelled"` is
`CANCEL`; a question is `ANSWER` with `payload`; an approval is `APPROVE` for `true`, `REJECT`
for `false`, `EDIT` for an object (the edited arguments), and anything else is refused. With
`decision: "reject"`, a text `payload` is the reviewer's reason, which the model reads.

The reviewer is the authenticated user (`identity`), and agent-runs records it as given when
the harness holds the application's key (it may act for anyone, the default). A harness whose
key is restricted to listed people answers only as one of them, a run assigned to that person
or to nobody: any other resume is refused by agent-runs, the stream ends with a `RUN_ERROR`
carrying the refusal, and the run stays paused ([interrupts.md](interrupts.md#answering)).
Under `serve_a2a` the same refusal fails the task (the run stays paused). Serve with the
application's key unless every person's runs are assigned to that person.

| Response | When |
|---|---|
| `401` `AUTHENTICATION` | `identity(request)` named nobody |
| `404` `NOT_FOUND` | a resume names no interrupt of this thread; a reconnect to a run this caller does not own or this process never served; an artifact the awaited interrupt does not reference, or agent-runs no longer has |
| `409` `CONFLICT` (`detail` `BAD_RESUME: …`) | a resume that cannot be read as a decision (an approval answered with neither `true`, `false` nor arguments), or that the run refuses (it waits on another interrupt, it is not paused) |
| `422` `VALIDATION` | a `runId` that is not a fresh identifier (letters, digits, `-_.:`; one already used here) |
| `422` (FastAPI's validation error) | a body that is not a `RunAgentInput`: a `decision` outside `answer`/`approve`/`reject`/`edit`/`cancel` (any case), a message `role` outside AG-UI's (`developer`, `system`, `assistant`, `user`, `tool`, `activity`, `reasoning`) |
| `RUN_ERROR` `RUN_ABORTED` | the run ended without telling its client (the harness itself failed, e.g. agent-runs refused a write) |

Every refusal of the surface's own is an RFC 9457 problem document (`application/problem+json`:
`type`, `title`, `status`, `detail`, `instance`, `code`, `retryable`), the platform's shape.

**OpenAPI.** The routes are in the app's OpenAPI document under the tag `agui`: both streams as
`text/event-stream` (each `data:` an AG-UI event, each `id:` its number), the artifact as the
media type it was stored with, every refusal with the problem schema, and `RunAgentInput`,
`Resume` (its `decision` the contracts `InterruptDecision`) and `Role` as schemas. An app that
has not named itself (FastAPI's default title) is titled `"<agent id> agent"`, with a
description and the harness's version.

**Several replicas.** A run's events are buffered in the process that serves it
(`agui/hub.py`), so a reconnect (`GET …/events`) must reach that replica: route a thread's
requests — keyed by the `threadId`, or the session cookie — to one replica (sticky sessions at the load balancer). A
resume may land anywhere (it reads the run from agent-runs), and so may an artifact read; a
reconnect that lands elsewhere is a `404`, after which the client can read the run's outcome
from agent-runs or start the next turn.

Harness events map one to one onto AG-UI events (the contracts already use AG-UI's names):
text and tool-call events carry their message and call ids, `CONTEXT_LOADED` becomes a
`CUSTOM` event named `context_loaded`, `CUSTOM` events (`tool_notice`, `log`, `warning`) keep
their name and carry the rest as `value`, `INTERRUPT` and `RUN_ERROR` are folded into the
finishing event, and an `ERROR`/`REJECTED`/`TIMEOUT` ending is `RUN_ERROR` with the error's
code and message.

`identity(request)` (sync or async) returns the user; the tenant is the one `TRELLIS_API_KEY`
speaks for. Nothing the client sends (`forwardedProps`, the thread...) decides who is calling.
Without `identity`, every request runs as `anonymous` (a warning is logged at mount).

## A2A: `agent.serve_a2a(app, url, *, identity=None)`

JSON-RPC at `url`'s path, the agent card at `{url}/.well-known/agent-card.json` (or the
origin's well-known path when `url` has none). The card names the agent by its id and
describes it with the target's own `description` (or `handoff_description`, or a function's
docstring), one JSON-RPC interface, streaming, push notifications and the trusted-identity
extension (`https://trellis.dev/a2a/extensions/trusted-identity/v1`); no credential is ever
part of it.

The task id is the run id, the context id the thread. Task states follow the run: `QUEUED` is
`submitted`, `RUNNING` `working` (text deltas and progress — tool calls, context, custom events
— arrive as working updates), `PAUSED` `input-required` with the question, `SUCCESS`
`completed` with the answer as the artifact `result`, `ERROR`/`TIMEOUT` `failed`,
`CANCELLED` `canceled`. A task this process does not hold (a restart, another replica) is
rebuilt from the run record when its own user asks for it.

The next message on an `input-required` task resumes it, as the person it belongs to (another
user is told "this task is not yours to answer" and the task keeps waiting). The decision comes
from a data part `{"decision": "approve"}` when present; else the words `cancel`, `abort`,
`stop` cancel; an approval reads `approve`/`approved`/`yes`/`ok`/`allow` and
`reject`/`rejected`/`no`/`deny`/`denied`, or a data object as the edited arguments (its
`payload` field when it has one); anything else answers the question (a data part's `answer`,
or the text). An answer that cannot be read keeps the task waiting and says why. A message to
a task that has ended, or is still working, is refused (`InvalidRequestError`), and a new task
never takes the id of an existing run. `CancelTask` cancels a working run, or ends a paused
one `CANCELLED`; the terminal state is sent once, whichever of the stream and the cancel gets
there first.

Push notifications go only to public https addresses — no credentials or fragment in the URL,
never `localhost` or a `.local`/`.internal`/`.localhost` name, every address the host resolves
to public (checked at registration and again at each delivery). The body is the protocol's
`StreamResponse`, signed `X-Trellis-Signature: t=<unix>,v1=<hmac-sha256 of "t.body">` with the
token the caller registered (agent-runs' webhook scheme), with `X-Trellis-Event:
a2a.task_update`, `X-Trellis-Delivery` and the token itself in `X-A2A-Notification-Token`. A
config without a token is not delivered to; a failing receiver is tried up to 3 times (0.2 s, then
0.4 s apart), logged, and never raised into the task. The signature is agent-runs' webhook
scheme, signed with `trellis.runs.webhooks.sign`; a receiver checks a delivery with
`trellis.runs.webhooks.verify_signature(secret, header, body, *, now=None, tolerance=300) -> bool`
(pip `trellis-runs`; a signature more than 300 s old or ahead is refused, a malformed header is
`False`).

**OpenAPI.** The A2A SDK serves its routes as plain Starlette routes, which FastAPI leaves out of
the app's OpenAPI document; on a FastAPI app the surface describes both under the tag `a2a` —
`GET {path}/.well-known/agent-card.json` (the `AgentCard`) and `POST {path}` (a JSON-RPC 2.0
request whose `method` is one of the protocol's: `SendMessage`, `SendStreamingMessage`,
`GetTask`, `ListTasks`, `CancelTask`, `SubscribeToTask`, the push-notification config methods,
`GetExtendedAgentCard`; answered with a JSON-RPC response, or server-sent events for the
streaming ones). The SDK's routes still answer every request; the descriptions never do.

Identity: `identity(context)` returns the user; by default the trusted `x-trellis-identity`
header — JSON `{"tenant_id": ..., "user_id": ...}`, set by the deployment's authenticating
edge, at most 4096 characters, its tenant (when it names one) the key's — else `anonymous`
(warned once per server). A header that is too long, not a JSON object, names another tenant
or no user is refused.

## Calling A2A agents: `a2a(url, *, name=None)`

A remote agent is one tool (`write`), named after its card (or `name`; unsafe characters become
`_`, at most 64), described by the card, taking `{"message": string}`. The card is read once;
each call is a `RemoteAgent` ([blocks/a2a.md](blocks/a2a.md)) as the calling run.
The message goes with the calling run's identity on the trusted-identity header (and the extension header) and its
thread as the context id, so a conversation between two agents is one thread on both sides.
The answer is the remote task's `result` artifact (several artifacts as a list), or its text.
A remote task that ends `failed`, `rejected` or `canceled` is a tool error the calling model
reads (`"<tool> failed: …"`), as is a remote agent that cannot be reached.

When the remote agent asks something (`input-required`), the calling run asks the same question
itself (`ask` is the `RemoteAgent`'s `on_input`), and the answer goes back on the same remote
task. If that pauses the calling run, the remote task is cancelled; the resumed run calls again and the journal answers the
question, so the remote agent gets the answer on its new task.
