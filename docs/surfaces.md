# Surfaces

## AG-UI: `agent.serve_chat(app, *, path="/agui", identity=None)`

| Route | |
|---|---|
| `POST {path}/run` | An AG-UI `RunAgentInput`. A new run (the client's `runId`, or one the harness names) executes in the background and its events stream as SSE, each with an `id:` numbered per run. A `resume` entry answers the interrupt a paused run on the thread waits on: `payload` is the answer (`true`/`false`/edited arguments for an approval), `status: "cancelled"` cancels, `decision` names a contracts decision outright. |
| `GET {path}/runs/{run_id}/events` | Reconnect: the run's events after `Last-Event-ID` (or `?after=`), then live until it finishes. Only the run's own user sees it. |
| `GET {path}/artifacts/{artifact_id}` | Data an interrupt carries by reference (`payload_ref`), from the paused run's checkpoint. |

The run keeps going when the client disconnects. Each run's events are buffered (2048 per run,
256 runs, least recently used dropped), so warnings from background writes after the finish
are still there on replay. A pause arrives as `RUN_FINISHED` with an `interrupt` outcome; a
failure as `RUN_ERROR`.

`identity(request)` (sync or async) returns the user; the tenant is `TRELLIS_TENANT`. Nothing
the client sends decides who is calling. Without `identity`, every request runs as
`anonymous` (a warning is logged at mount).

## A2A: `agent.serve_a2a(app, url, *, identity=None)`

JSON-RPC at `url`'s path, the agent card at `{url}/.well-known/agent-card.json` (or the
origin's well-known path when `url` has none). The task id is the run id, the context id the
thread. A pause is `input-required` with the question; the next message on the task resumes it
(the decision from a data part, from approve/reject/cancel words, or an object as an edit).
Push notifications go only to public https addresses and are signed
`X-Trellis-Signature: t=<unix>,v1=<hmac>` with the token the caller registered (the memory
service's webhook scheme); a config without a token is not delivered to.

Identity: `identity(context)` returns the user; by default the trusted
`x-trellis-identity` header (JSON; its tenant must be `TRELLIS_TENANT`), else `anonymous`.

## Calling A2A agents: `a2a(url)`

A remote agent is one tool. The message goes with the calling run's identity header and its
thread as the context id. When the remote agent asks something, the calling run asks the same
question itself (`ask`), and the answer goes back on the same remote task.
