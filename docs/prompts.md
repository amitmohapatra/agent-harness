# Prompts: in code, or stored in the gateway

A prompt in code is a string. A prompt the team changes without a redeploy is stored in the
Bifrost gateway's Prompt Repository — versioned, rolled out there — and named in code with
`stored(...)`.

```python
from trellis import Harness, ReAct, stored

h = Harness()  # BIFROST_URL: the gateway that keeps the prompts
agent = h.wrap(ReAct(system_prompt=stored("triage", team="EU"), model=MODEL), id="triage")
prompt = await h.prompt("triage@3", team="EU")  # any framework: prompt.text, prompt.messages
```

| You write | Automatic |
|---|---|
| the prompt's name: `"triage"` (the version the gateway serves now) or `"triage@3"` (committed version 3) | the version pinned per run and journaled — a resumed run reads the same one, even after a newer was committed; the run's `prompt` event and the `trellis.prompt.*` span attributes; the last good copy while the gateway cannot be reached |
| its variables (`stored("triage", team="EU")`, `h.prompt(ref, team="EU")`) | `{{variable}}` filled in; a variable left out is an error, not a literal `{{team}}` sent to the model |

## What

`stored(ref, **values)` names a stored prompt and the values its `{{variables}}` take; it is a
`ReAct`'s `system_prompt` or `h.prompt`'s reference. `await h.prompt(ref, **values)` reads it:
`.text` (its messages' text, in order) and `.messages` (its chat messages, filled).

A `ReAct` whose stored prompt has no variables to fill, its model named and routed through
Bifrost, has each model call *select* it (the gateway prepends the pinned version: its
`x-bf-prompt-id` and `x-bf-prompt-version` headers); any other — variables to fill, a model
object of yours — is rendered into the instructions. A graph of your own takes it with the
harness's middleware: `ModelHooks(prompt=stored(...))` ([gateway.md](gateway.md#prompts)).

## Native or ours

Of the frameworks, only the OpenAI Agents SDK keeps prompts of its own:
`Agent(prompt=Prompt(id=..., version=...))` (`agents.Prompt`, a `TypedDict` of `id`, `version`,
`variables`), a prompt stored at OpenAI and applied by OpenAI's Responses API — the SDK says it
is usable only with OpenAI models through that API. The harness passes it through (it runs a copy of your `Agent`) but
does not read, pin or journal it. Deep Agents (`create_deep_agent(system_prompt=)`), LangChain
(`create_agent(system_prompt=)`) and the Claude Agent SDK (`ClaudeAgentOptions(system_prompt=)`)
take text.

* **Theirs:** an OpenAI Agents agent on OpenAI's models whose prompts the team already manages
  at OpenAI.
* **Ours:** the gateway's Prompt Repository the deployment shares, a version pinned per run and
  the same text on every replay — for a `ReAct` (`stored(...)`) and for any framework that
  takes text (`await h.prompt(...)`).

## When

The instructions change more often than the code, or someone else owns them (a support lead's
tone, a legal rubric): keep them in the gateway and version them there. Otherwise a string in
code is reviewed with it.

## On failure

| What | Then |
|---|---|
| a name (or `name@version`) the gateway does not have | `ConfigurationError` (`no committed prompt named 'triage'`): a `ReAct` run ends `ERROR` with it |
| no gateway (`BIFROST_URL` unset) | `ConfigurationError`: set `BIFROST_URL` (a prompt in code is a string) |
| the gateway unreachable, read before | the last copy read (a warning in the log) |
| a variable not given | `ConfigurationError` naming it |
| a malformed reference (`"triage@"`), a version that is not a number | `ConfigurationError` where `stored(...)` is written; the run ends `ERROR` |
| a stored prompt outside a run, through the middleware | `ConfigurationError`: render it yourself (`await h.prompt(...)`) |

## Example

[`examples/05_features/stored_prompts.py`](../examples/05_features/stored_prompts.py): a stored
prompt as a `ReAct`'s instructions, pinned per run, and read with `h.prompt`, on a scripted
gateway.
