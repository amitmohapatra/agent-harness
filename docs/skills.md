# Skills: your framework's own, and what the agent learned

Two kinds of skill reach an agent. The ones you write — Agent Skills folders
(`<name>/SKILL.md`) — are your framework's, read by it as it reads them. The ones the agent
learns on its own come with its memory context, with nothing to configure.

## Native or ours

Three frameworks read Agent Skills folders (`<name>/SKILL.md`) themselves:

| Framework | Its own skills |
|---|---|
| Deep Agents | `create_deep_agent(skills=["/skills/project/"], backend=...)` adds `SkillsMiddleware(backend=, sources=)` (`deepagents/middleware/skills.py`): the skills' names and descriptions in the system prompt, each `SKILL.md` read through the backend (`FilesystemBackend(root_dir=)` for a folder on disk; the graph state by default), a later source overriding an earlier one, loaded once per thread. It is a LangChain `AgentMiddleware`, so `create_agent(..., middleware=[SkillsMiddleware(...)])` takes it too |
| OpenAI Agents SDK | the `Skills` capability of a `SandboxAgent` (`agents.sandbox.capabilities`): `Skills(from_=LocalDir(src=...))`, `Skills(lazy_from=LocalDirLazySkillSource(...))` (with a `load_skill` tool) or `Skills(skills=[Skill(...)])`, copied into the sandbox's workspace (`skills_path`, `.agents` by default). A `SandboxAgent` runs only with `RunConfig(sandbox=...)`, which `h.wrap` does not pass: call `Runner.run` yourself ([blocks/openai-agents.md](blocks/openai-agents.md)) |
| Claude Agent SDK | `ClaudeAgentOptions(skills=["sql-review"] \| "all")`: the skills the Claude Code CLI finds, enabled by name; the SDK adds `Skill(name)` to `allowed_tools` and, when `setting_sources` is unset, `["user", "project"]`. A filter on what the model is offered, not a sandbox |

Skills you write are your framework's: the harness keeps none of its own. Every call a skill
leads to is a call of the run — governed, journaled, recorded — whichever framework made it.

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
