# Prompts and skills, as blocks (Way 2)

Your framework runs the agent; your code reads its instructions and its skills from wherever
your team keeps them — code, `.md` files, Langfuse, the Bifrost gateway — with no `Harness` at
all. These are the same objects a `Harness` uses ([prompts.md](../prompts.md),
[skills.md](../skills.md)): one implementation, two ways.

**Install:** `pip install trellis-harness` (no framework extra needed for these blocks).

## The API

| Name (`trellis.harness.prompts`) | What it is |
|---|---|
| `await resolve_prompt(ref, *, sources=None, **vars) -> str` | the prompt `ref` names (`"name"`, `"name@3"`, `"name@staging"`, or a `Prompt`) as text, `{{vars}}` filled; `sources` a `PromptSources` or a list of sources, else the environment's (`PROMPTS_DIR`, the Langfuse keys, `BIFROST_URL`), closed after |
| `await resolve_prompt_messages(ref, *, sources=None, **vars) -> list[dict]` | the same, as chat messages |
| `Prompt(name, text, version="1")`, `prompts_dir(path)`, `langfuse_prompts(public_key=, secret_key=, host=None)` | sources: `await source.resolve(name, version) -> ResolvedPrompt` (`.text` or `.chat`, `.version`, `.source`, `.render(**vars)`, `.messages(**vars)`) |
| `PromptSources([...])`, `PromptSources.from_env()` | sources in order: `await get(ref)`, `await render(ref, **vars)`, `await messages(ref, **vars)`, `await aclose()` |

| Name (`trellis.harness.skills`) | What it is |
|---|---|
| `await resolve_skill(ref, *, sources) -> ResolvedSkill` | the skill `ref` names (`"name"`, `"name@1.2.0"`, a `Skill`): `.description`, `.body`, `.files`, `await .read(path)` |
| `Skill(name, description, body, files={...})`, `skills_dir(path)` | sources: `await source.resolve(name, version)` |
| `SkillSources([...])`, `SkillSources.from_env()` | sources in order: `await pin(refs, recorded=None) -> PinnedSkills` (`.section` for your prompt, `await .load(name)`, `await .read(name, path)` for your tools, `.record()` for your checkpoint) |

## LangGraph

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

from trellis.harness.prompts import PromptSources
from trellis.harness.skills import SkillSources

prompts, skills = PromptSources.from_env(), SkillSources.from_env()
kit = await skills.pin(["sql-review", "refunds@1.2.0"])


@tool
async def load_skill(name: str) -> str:
    """Read a skill's instructions and the files it has."""
    return await kit.load(name)


@tool
async def read_skill_file(name: str, path: str) -> str:
    """Read one file of a skill."""
    return await kit.read(name, path)


system = await prompts.render("analyst@3", team="data")
graph = create_agent(
    model, tools=[load_skill, read_skill_file], system_prompt=f"{system}\n\n{kit.section}"
)
result = await graph.ainvoke({"messages": [{"role": "user", "content": "Review: SELECT * FROM t"}]})
# keep kit.record() with your checkpoint; after a restart: await skills.pin([...], recorded=saved)
```

## OpenAI Agents SDK

```python
from agents import Agent, Runner, function_tool

from trellis.harness.prompts import resolve_prompt
from trellis.harness.skills import SkillSources, skills_dir

skills = SkillSources([skills_dir("skills")])
kit = await skills.pin(["sql-review"])


@function_tool
async def load_skill(name: str) -> str:
    """Read a skill's instructions and the files it has."""
    return await kit.load(name)


@function_tool
async def read_skill_file(name: str, path: str) -> str:
    """Read one file of a skill."""
    return await kit.read(name, path)


instructions = await resolve_prompt("triage", team="EU")  # PROMPTS_DIR, Langfuse, the gateway
agent = Agent(
    name="reviewer",
    instructions=f"{instructions}\n\n{kit.section}",
    tools=[load_skill, read_skill_file],
)
result = await Runner.run(agent, "Review: SELECT * FROM t")
```

## Behaviour

* **Order.** The first source that has the name — and the version named — answers; a name
  none has is a `ConfigurationError` naming every source tried. A source that fails (Langfuse
  unreachable, never read before) is an error: the next source is not asked in its place.
* **Kept.** Langfuse and the gateway keep what they read (a version once, a label or the
  latest every 300 s); while they cannot be reached the last copy stands. A folder is read at
  each lookup.
* **Pinned.** There is no harness journal here: keep what you resolved with your framework's
  own state — `ResolvedPrompt.record()` (`ResolvedPrompt.of_record(saved)`) and
  `PinnedSkills.record()` (`pin(..., recorded=saved)`) — so a resumed run reads the same text.
* **Files.** A skill's file is read from its source for the version pinned; a source that holds
  another version now refuses it, saying both; a path out of a skill folder is refused.

Way 1 does all of this for you: [prompts.md](../prompts.md), [skills.md](../skills.md).
