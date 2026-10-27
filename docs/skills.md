# Skills: from your stores, in each framework's own slot; and what the agent learned

Two kinds of skill reach an agent. The ones you write — Agent Skills (`SKILL.md` and its files)
kept in a folder, the gateway's repository, Langfuse or a store of your own, edited at any
time, per tenant, workspace, user and agent — come from **skill sources** (`trellis.skills`),
which feed each framework's *own* skills mechanism. The ones the agent learns on its own come
with its memory context, with nothing to configure.

## Skill sources

```python
from trellis import Harness, ReAct, skills

src = skills.combine(skills.directory("skills"), skills.bifrost(), my_store)
h = Harness()
agent = h.wrap(ReAct("You review SQL.", model, skills=src), id="reviewer")
```

| Source | What it is |
|---|---|
| `skills.directory(path, *, level="platform")` | `<name>/SKILL.md` folders on this machine (front matter `name`, `description`; optionally `version`, else the folder's content digest; an allow-list `tenants:` / `users:`), read at each run |
| `skills.bifrost(admin=None)` | the gateway's skills repository (`bifrost_sdk.admin.skills`): platform-wide; any published version loads, its files only of the served one (the gateway serves no other); the allow-list in a skill's extra front matter |
| `skills.langfuse(client=None, *, tag="skill", label="production")` | Langfuse prompts tagged `tag`, the version labelled `label` (checked again every 5 minutes, the last read standing while Langfuse is down): the text the body, `config.description` (else its first line) the description; no files |
| your store | the `SkillSource` protocol: `label`, `async list(scope) -> [SkillInfo]`, `async load(name, version, scope) -> SkillDoc`, `async read_file(name, version, path, scope) -> str`, and optionally `async changed_since(scope, cursor) -> bool` (a cheap check that spares a listing) |
| `skills.combine(*sources)` | several as one |

`SkillInfo(name, description, version, updated_at=None, level="platform", enabled=True)` is one
skill of a scope: `level` is where it is kept (`platform`, `tenant`, `workspace`, `user`,
`agent`), and a `version` names one text the store can serve again (`load` and `read_file`
raise a `LookupError` for a version, or a file, it no longer has). `Scope(tenant, workspace,
user, agent)` is whose. `SkillDoc(body, files)` is one version: its body (no front matter) and
its files' relative paths.

## Where it goes, per engine

Each framework reads the skills through its own skills slot; only an engine with none gets
the harness's two tools.

| Engine | Where | Its own mechanism (in the tested versions) |
|---|---|---|
| `ReAct` | `ReAct(skills=src)` | Deep Agents' `SkillsMiddleware`: the skills a `/skills/` route of the graph's backend (`CompositeBackend`), listed in the system prompt, each `SKILL.md` read with `read_file` |
| Deep Agents (your graph) | `create_deep_agent(model, **src.deep_agent())`; with a sandbox, `**deep_agent(sandbox(), src)` | the same, in its one `backend=`: the sandbox the composite's default, the skills its `/skills/` route |
| Claude Agent SDK | `ClaudeAgentOptions(**src.claude())` | a local plugin (`plugins=`, `skills="all"`): the harness writes the run's skills into a folder of the run's own before each query and removes it after; the CLI finds them as `trellis:<name>` |
| Kimi Code | `kimi(**src.kimi())` | Kimi's own skills folder (`$KIMI_CODE_HOME/skills`): the harness writes the run's skills there, in a home of the run's own, before each attempt; Kimi lists them and loads one with its `Skill` tool ([frameworks/kimi.md](frameworks/kimi.md#native-or-ours-sandbox-skills-hooks)) |
| OpenAI Agents SDK, `SandboxAgent` | `capabilities=[..., src.openai()]` | the SDK's `Skills` capability, lazily loaded: listed in its instructions, copied into the sandbox by its `load_skill` |
| OpenAI `Agent`, LangGraph, a function | `src.tools()` (`agent.tools.use(...)`) | none: `list_skills` and `load_skill(name, path=None)`, harness tools (governed, journaled, recorded) |

The OpenAI Agents SDK has a skills slot only on a `SandboxAgent` (which needs
`framework_options={"run_config": RunConfig(sandbox=...)}`); a plain `Agent` has none. The
Claude Agent SDK's `skills=` only filters what its CLI found on disk: the plugin is how skills
reach it.

## Per run

* **Scope.** A run's scope is its own — `tenant`, the workspace, `user`, the agent's id — passed
  to every source. Code that is not wrapped gives one (`src.deep_agent(scope=)`,
  `src.tools(scope=)`, `src.openai(scope=)`, `await src.claude_dir(path, scope=)` for a Claude
  SDK you run yourself), else the default tenant's.
* **Which one a name means.** The most specific level wins — agent over user over workspace
  over tenant over the platform-wide sources — then the source given first; a winner that is
  disabled hides the name. The listing is sorted by name (a cache-stable prompt).
* **Pinned.** At the run's first use the versions are pinned and kept in its journal (the
  stored prompts' pin, `repository.journaled`): a resumed run — after a pause, a crash, on
  another worker — uses the versions it saw; an edit applies from the next run, never mid-run.
  A run says them: a `skills` event with the versions, the `trellis.skills` span attribute.
* **Fails soft.** A skill deleted (or a version its store no longer serves) is an error the
  model reads, said as a `skills_unavailable` warning; the run goes on. A source that is down
  stands at its last listing of the scope (said once, the same warning); one never read lists
  nothing.
* **Text is data.** A description is one line of at most 1,024 characters, a body at most
  32,000, a file at most 256 KiB (cut, saying so); a name is the Agent Skills spec's (lowercase
  letters, digits, single hyphens, at most 64) — any other is skipped; a file path that climbs
  out of its skill is never listed. Scripts run only where the engine runs commands (its
  sandbox).

## Native or ours

A framework's own skills configuration stays as it is — `create_deep_agent(skills=["/skills/"])`
on your own backend, `ClaudeAgentOptions(skills=[...])` over the CLI's folders, a
`SandboxAgent`'s `Skills(from_=LocalDir(...))`. `ReAct(skills=)` takes Deep Agents' own source
paths too. A skill source is for skills kept in a store, edited while agents run, and different
per tenant and user. Every call a skill leads to is a call of the run — governed, journaled,
recorded — whichever framework made it.

**Example.** [`examples/05_features/skill_sources.py`](../examples/05_features/skill_sources.py): a
folder and a store of the team's own, combined, read by `ReAct` and by a function. Tests:
`tests/unit/test_skills.py` (the sources, merging, caps, a source down, many skills) and
`tests/integration/test_skills.py` (every engine's slot, a resume after an edit, a deleted
skill, two tenants and two users).

## Learned skills

An agent also learns skills of its own, with nothing to configure: the memory service mines
the tool calls of its successful runs, across all of its users, and every later run of that
agent finds what it learned in its memory context, in full and matched to the task:

```text
## Learned skills for this task
- refund-an-order: find_order -> refund (worked 95% of 40 runs)
  - if refund on AlreadyRefunded: stop and tell the user
- adds to your skill refund-policy: find_order -> refund (worked 100% of 12 runs)
```

Nothing is published or loaded: the harness pushes the context before every run (any
adapter, Way 1 or ReAct; Way 2 and no-harness code get it from the same `context()` call). A
learned skill never replaces one of yours: when its runs opened your skill (`load_skill`, which
the harness records like any call), it is shown as what they added to it, and your `SKILL.md` is
never edited. Another user of the agent is offered a learned skill once two users produced it,
another agent never. An administrator sees and dismisses them with the memory service's
`GET /v1/skills` and `POST /v1/skills/{id}/dismiss`
([learned skills](https://github.com/amitmohapatra/agent-memory-service/blob/main/docs/api/skills.md)).
