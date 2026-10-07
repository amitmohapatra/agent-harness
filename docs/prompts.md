# Prompts: from code, `.md` files, Langfuse or the gateway

A prompt is named once — `"triage"`, `"triage@3"` — and read the same way wherever it is kept:
in code, in a folder of Markdown files, in Langfuse's prompt management or in the Bifrost
gateway's Prompt Repository. The deployment says where to look (its environment); the code
says only which prompt and its variables.

```python
from trellis import Harness, ReAct

h = Harness()  # PROMPTS_DIR=prompts, LANGFUSE_* keys, BIFROST_URL: whichever are set
agent = h.wrap(
    ReAct(system="", model=MODEL, prompt="triage@3", prompt_vars={"team": "EU"}), id="triage"
)
instructions = await h.prompt("triage", team="EU")  # any framework's own instructions
```

| You write | Automatic |
|---|---|
| the prompt's name (`"triage"`, `"triage@3"`, `"triage@staging"`) or a `Prompt(...)` | where it is looked up, in a fixed order; the version pinned per run and journaled; the run's `prompt` event and span attribute; the last good copy of a remote source |
| its variables (`prompt_vars=`, `h.prompt(ref, team="EU")`) | `{{variable}}` filled in; a variable left out is an error, not a literal `{{team}}` sent to the model |
| optionally `Harness(prompts=[...])`: the sources, when the code knows them | used as they are, in their order, instead of the environment's |

## What

A prompt source answers one question, `resolve(name, version) -> ResolvedPrompt`
(`trellis.harness.prompts.PromptSource`): the prompt's text — or its chat messages — with
`{{var}}` placeholders, its name, the version it is, the source it came from and its config.
`version=None` is the version the source serves now. Shipped sources:

| Source | How it is named | `name` | `name@v` | Notes |
|---|---|---|---|---|
| code: `Prompt(name, text, version="1")` | `Harness(prompts=[Prompt(...)])`, or given where a prompt is named (`ReAct(prompt=Prompt(...))`, `h.prompt(Prompt(...))`) | its version | its version only | `text` may be chat messages (a list of `{"role", "content"}`) |
| files: `prompts_dir(path)` | `PROMPTS_DIR`, or `Harness(prompts=[prompts_dir("prompts")])` | `<name>.md` now | the file, when its version is `v` | optional front matter: `version` (else the file's content digest), `description`, anything else as its config; `support/triage.md` is `support/triage` |
| Langfuse: `langfuse_prompts(...)` | `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` (at `LANGFUSE_HOST`, else Langfuse Cloud) — the same three names send the traces and scores to Langfuse ([observability.md](observability.md#export)) | the version labelled `production` | version `v` (a number), else the version labelled `v` | Langfuse's public API (`GET /api/public/v2/prompts/{name}`); text and chat prompts, chat placeholders; its `config` |
| Bifrost: the gateway's Prompt Repository | `BIFROST_URL` | the latest committed version | committed version `v` | selected by header where the harness calls the model ([gateway.md](gateway.md#prompts)); its messages are its text elsewhere |

**The order.** Like every block of a `Harness`, the prompt sources passed
(`Harness(prompts=[...])`, or a `PromptSources`) are used as they are, in their order; not
passed, they are the ones the environment names, in this order. The first source that has the
name — and the version named — answers; a name in two sources is the first one's. A `Prompt`
given where a prompt is named is used as it is.

| Order (from the environment) | Source | On when |
|---|---|---|
| 1 | `prompts_dir(PROMPTS_DIR)` | `PROMPTS_DIR` is set |
| 2 | Langfuse | `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set |
| 3 | the gateway's Prompt Repository | `BIFROST_URL` is set (or `Harness(gateway=)`) |

One rule: sources passed replace the environment's, whatever they are; `Harness(prompts=[])`
is no prompt source. To keep one of the environment's beside your own, name it too:
`Harness(prompts=[Prompt(...), prompts_dir("prompts"), langfuse_prompts(...)])`. A `Prompt`
given where a prompt is named (`ReAct(prompt=Prompt(...))`, `h.prompt(Prompt(...))`) needs no
source at all.

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
* **Ours:** a central registry the deployment shares (Bifrost's Prompt Repository, Langfuse),
  a folder reviewed with the code, a version pinned per run and the same text on every replay,
  or a `ReAct` or plain-function agent — and any framework that takes text
  (`await h.prompt(...)`).

## When

* The instructions change more often than the code, or someone else owns them (a support
  lead's tone, a legal rubric for the judge): keep them in Langfuse or the gateway and version
  them there.
* The instructions are reviewed like code: a folder of `.md` files in the repository.
* A test or a one-off agent: a `Prompt` in code.

## Where

Way 1, every adapter: `ReAct(prompt=)` renders it into the instructions (or, for the gateway's,
selects it); LangGraph, Deep Agents, OpenAI Agents, the Claude Agent SDK and a function read it
with `await h.prompt(ref, **vars)` (text) or `await h.prompt_messages(ref, **vars)` (chat
messages) — once when the framework's agent is built, or inside a run (pinned for the run). A
stored prompt of the gateway is selected by the framework's own model client with
`await h.model_headers(prompt="triage")`: every run pins it at its start and the headers,
read per request, select that version ([gateway.md](gateway.md#prompts)). The LLM judge:
`llm_judge(criteria, prompt=)`.

## How

```python
from trellis.harness.prompts import Prompt, prompts_dir

h = Harness(prompts=[Prompt("greet", "Greet {{name}} in one line.")])

# a ReAct: a prompt that is not the gateway's becomes the instructions, before `system` and the
# pushed context (a chat prompt's other messages, e.g. examples, follow the system message)
agent = h.wrap(
    ReAct(system="Be brief.", model=MODEL, prompt="greet", prompt_vars={"name": "Ada"}),
    id="greeter",
)

# any other framework: its own instructions
graph = create_agent(model, tools=tools, system_prompt=await h.prompt("triage", team="EU"))
openai_agent = Agent(name="triage", instructions=await h.prompt("triage", team="EU"))

# the judge
judge = llm_judge("Follows the rubric.", prompt="rubric@4")
```

A `prompts/triage.md`:

```markdown
---
version: 3
description: Triage incoming tickets.
temperature: 0
---
You triage tickets for the {{team}} team. Answer with one priority: P1, P2 or P3.
```

**Variables.** `{{name}}` (spaces inside the braces allowed) is filled with the value given;
a variable the prompt has and the call does not give is a `ConfigurationError` naming it. A
Langfuse chat prompt's placeholder (`{"type": "placeholder", "name": "history"}`) is filled with
the list of messages given under its name. A gateway prompt is prepended by the gateway as it is
stored, so `ReAct(prompt_vars=)` with one is refused.

## Automatic

* **Pinned for the run.** Inside a run (a `ReAct`'s `prompt=`, `h.prompt` in a function or a
  graph node, the stored prompts `h.model_headers(prompt=)` handed to model clients, at the
  run's start) what was resolved — the text, the version, the source — is journaled: a resumed
  run, after a pause or a crash, on this worker or another, reads the same text even after the
  file, the label or the commit changed. A new run reads the source again.
* **Said.** A `prompt` event (`{"prompt": "triage", "version": "3", "source": "prompts_dir(prompts)"}`)
  and the `trellis.prompt` attribute (`triage@3`) of the span current then; a prompt pinned for
  every model call (a `ReAct`'s, `h.model_headers(prompt=)`'s) also puts
  `trellis.prompt.name`, `trellis.prompt.version` and `trellis.prompt.source` on the run's
  agent span; each `ReAct` `chat` span carries `trellis.prompt.name`, `trellis.prompt.version` and `trellis.prompt.source`
  (and `trellis.prompt.id` for the gateway's).
* **Kept.** Langfuse is read once per version and every 300 s per label; while it cannot be
  reached the last copy read stands (asked again after 30 s). The gateway's prompts are kept the
  same way ([gateway.md](gateway.md#prompts)). A folder is read at each lookup: what is on disk
  now.

## On failure

| What | Then |
|---|---|
| a name (or `name@version`) no source has | `ConfigurationError` naming every source tried and what each said: `no prompt 'triage@9' in any source (prompts_dir(prompts): triage.md is version 3, not 9; Langfuse (https://…): no prompt 'triage' version 9; Bifrost: …)` — a `ReAct` run ends `ERROR` with it |
| no prompt source at all | `ConfigurationError`: pass `Harness(prompts=[...])`, or set `PROMPTS_DIR`, the Langfuse keys or `BIFROST_URL` |
| a source that fails (Langfuse unreachable, never read before) | a retryable error saying so; the next source is **not** asked in its place, so what a name resolves to never depends on an outage |
| Langfuse unreachable, read before | the last copy read (a warning in the log) |
| a variable not given | `ConfigurationError` naming it |
| a name that leaves `PROMPTS_DIR` (`../x`, a link elsewhere) | refused: not found in that folder |
| a malformed reference (`"triage@"`) | `ConfigurationError` when `ReAct` or `llm_judge` is built |
| a gateway prompt with a model object, or with `prompt_vars=` | the run ends `ERROR`: the gateway prepends it, to a Bifrost model name's calls only |

## Example

[`examples/05_features/skills_and_prompts.py`](../examples/05_features/skills_and_prompts.py): a folder of prompts and
of skills, a prompt in code, a `ReAct` and a function reading them, with no services.
