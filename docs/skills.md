# Skills: from code, `SKILL.md` folders or the gateway

[Agent Skills](https://agentskills.io) — instructions for a kind of task, and the files they
refer to — that the model reads only when a task needs them. A skill is named once —
`"sql-review"`, `"refunds@1.2.0"` — or given (`Skill(...)`), and comes from code, a folder in
the Agent Skills layout or the Bifrost gateway's Skills Repository; skills of every source mix
in one run.

```python
from trellis import Harness, skills
from trellis.harness.skills import Skill

h = Harness()  # SKILLS_DIR=skills, BIFROST_URL: whichever are set
tone = Skill("tone", "How we write to customers.", "Short sentences. No jargon.")
agent = h.wrap(target, id="analyst", skills=["sql-review", "refunds@1.2.0", tone])
graph = create_agent(model, tools=await h.tools(skills("sql-review", tone), framework="langgraph"))
```

| You write | Automatic |
|---|---|
| the skills' names (`"sql-review"`, `"refunds@1.2.0"`) or `Skill(...)` objects | where each is looked up; its version pinned per run and journaled with its body; one `## Skills` section in the context; `load_skill` and `read_skill_file`, governed and journaled |
| optionally `Harness(skills=[...])`: the sources, when the code knows them | used as they are, in their order, instead of the environment's |

## What

A skill source answers `resolve(name, version) -> ResolvedSkill`
(`trellis.harness.skills.SkillSource`): the skill's name, version, description, `SKILL.md` body
and the paths of its other files, which `read(path)` reads. `version=None` is the version the
source serves now. Shipped sources:

| Source | How it is named | `name` | `name@v` | Files |
|---|---|---|---|---|
| code: `Skill(name, description, body, files={path: text}, version="1")` | given where skills are named (`skills=[Skill(...)]`, `skills(Skill(...))`), or `Harness(skills=[Skill(...)])` | its version | its version only | `files` |
| files: `skills_dir(path)` | `SKILLS_DIR`, or `Harness(skills=[skills_dir("skills")])` | `<name>/SKILL.md` now | the folder, when its version is `v` | every other file of `<name>/`, by its relative path; a path out of the folder is refused |
| Bifrost: the gateway's Skills Repository | `BIFROST_URL` | the version served when the run starts | that published version | the served version's only ([gateway.md](gateway.md#skills)) |

A skill folder (the Agent Skills layout):

```text
skills/
  sql-review/
    SKILL.md          # front matter, then the instructions
    rules.md          # read with read_skill_file("sql-review", "rules.md")
    examples/good.sql
```

```markdown
---
name: sql-review
description: Reviews SQL queries against the team's rules.
version: 1.2.0
---
Read rules.md first, then check the query against each rule.
```

The front matter's `name` is the folder's; `description` is required; `version` (or
`metadata.version`) names the version, else it is the folder's content digest (any change is a
new version).

**The order.** A `Skill` given where skills are named is its own. A name is looked up in the
skill sources: like every block of a `Harness`, the ones passed (`Harness(skills=[...])`, or a
`SkillSources`) are used as they are, in their order; not passed, they are the ones the
environment names, in this order. The first that has the name — and the version named —
answers.

| Order (from the environment) | Source | On when |
|---|---|---|
| 1 | `skills_dir(SKILLS_DIR)` | `SKILLS_DIR` is set |
| 2 | the gateway's Skills Repository | `BIFROST_URL` is set (or `Harness(gateway=)`) |

`without={"skills"}` (on `h.wrap` or a run) turns skills off whatever their source: no section,
no tools, nothing pinned.

## When

Instructions for kinds of task that most runs do not need: loading them all into every prompt
costs tokens and attention; listing them and loading one when it fits does not. Keep them in the
repository (`SKILLS_DIR`), publish and roll them out in the gateway, or write one in code.

## Where

Way 1, every adapter: `h.wrap(..., skills=[...])` for a function, `ReAct`, OpenAI Agents and the
Claude Agent SDK; `h.tools(skills(...), framework=...)` for a graph that binds its tools when it
is built (LangGraph, Deep Agents). The `skills` feature covers skills of every source. Way 2,
code that is not wrapped, pins the same way:

```python
from trellis.harness.skills import SkillSources

sources = SkillSources.from_env()  # SKILLS_DIR, BIFROST_URL
kit = await sources.pin(["sql-review", tone])
system = f"{instructions}\n\n{kit.section}"  # into your framework's prompt
checkpoint["skills"] = kit.record()  # keep it with your framework's state


# your framework's two tools
async def load_skill(name: str) -> str:
    return await kit.load(name)


async def read_skill_file(name: str, path: str) -> str:
    return await kit.read(name, path)


# resuming: the same skills, whatever the sources hold now
kit = await sources.pin(["sql-review", tone], recorded=checkpoint["skills"])
```

## How

```python
h = Harness(skills=[skills_dir("skills")])
agent = h.wrap(
    ReAct(system="You review SQL.", model=MODEL),
    id="reviewer",
    skills=["sql-review", Skill("tone", "How we write.", "Short sentences.")],
)
result = await agent.run("Review: SELECT * FROM orders", user="ada")
```

## Automatic

Progressive disclosure, for every framework:

* **Pinned.** At the start of a run each skill is resolved — the version named, else the one its
  source serves — and journaled with its description, body and file list: a resumed run keeps
  them, on this worker or another, whatever its source holds by then.
* **Disclosed.** The context pushed into the framework's input gets one `## Skills` section,
  after the memory context: each skill's name and description, and how to read one.
* **Read.** Two read-only tools, always offered: `load_skill(name)` — the pinned body and its
  file list — and `read_skill_file(name, path)`, one file. Both are harness tools: every call
  goes through the bridge (governed, recorded) and is journaled, so a resumed run reads what it
  read.
* **Said.** A `skills` event (`{"versions": {"sql-review": "1.2.0", "tone": "1"}}`) and the
  `trellis.skills` attribute of the run's span.

## On failure

| What | Then |
|---|---|
| a skill no source has, or whose source is down and never read it | a `skills_unavailable` warning event naming every source tried, and the run goes on without it |
| skills named, and no skill source at all | the run ends `ERROR` with a `ConfigurationError`: pass `Harness(skills=[...])` or `Skill` objects, or set `SKILLS_DIR` or `BIFROST_URL` |
| a file of a version the source no longer holds (the folder changed, the gateway rolled out another version) | an error the model reads, naming both versions; the body still loads (it is the run's) |
| a path the skill does not list, or out of its folder (`../`, a link elsewhere) | an error the model reads: the skill has no such file |
| a `SKILL.md` whose front matter is not the folder's name, has no description, or cannot be read | the skill is unavailable (a warning naming the file and the line) |
| the gateway down, read before | the last copy read stands |

## Example

[`examples/prompts_and_skills.py`](../examples/prompts_and_skills.py).
