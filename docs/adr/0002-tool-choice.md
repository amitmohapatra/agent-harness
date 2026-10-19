# 0002. Tool choice is guarded by opt-in middleware, a warning-only check of the tool set, and per-item coverage

**Status:** Accepted (2026-10-19)

## Context

A deployed `ReAct` agent with memory chose its tools badly in four typical ways: a broad
dashboard tool called instead of the specific ones; a part of a two- or three-part request
never served (a navigation-help tool, a history tool); a request to create something answered
in text, so the write tool was never called and its approval never asked; a conclusion the
results did not support. Its tools came from its own application, so the harness cannot know
them: whatever it does must hold for any tool set, from any source — Bifrost's MCP tools (with
or without Code Mode), other MCP servers behind the gateway, or Python functions.

What the harness had: every source's names, descriptions and schemas passed through to each
framework's format (a function's description cut to its docstring's first paragraph, its
parameters undescribed); duplicate names refused and MCP tools namespaced `<server>-<tool>`;
the memory service's tool hints narrowing a call to at most 8 tools from 5 on; governance asking
for approval when a write is *called*; `called` and `tool_sequence` evaluators per dataset.
Nothing checked a tool set's descriptions, nothing made sure each part of a request was
served or that a requested change was made by its tool, and an evaluation could not say which
tools each item needs — nor score a run that paused for approval, which is the correct outcome
of a write request.

## Decision

1. **Descriptions are the author's; the harness keeps and checks them.** A function tool's
   description is its whole docstring but the sections, its parameters described from `Args:`
   or `:param:`. `tools/lint.py` finds what makes a set hard to choose among; the toolbox logs
   each finding as a warning when its names change. It never refuses a tool: a team that wants
   a clean set asserts `lint(...) == []` in its own CI.
2. **The guards are LangChain middleware, off by default.** `ToolGuidance` (instructions on
   how to choose: every part, specific over general, live figures from tools, changes by their
   tool) and `AnswerCheck` (the draft answer checked once against the request, the draft not
   streamed; the tools still needed called) are switched on by `ReAct(tool_guidance=True,
   answer_check=True)` and work in any `create_agent` or Deep Agents graph. They name no tool
   and read only what the run knows. They are code, not settings: a design choice per agent,
   not a fact of the deployment.
3. **Before a change, the rules; after nothing found, "no record".** `AnswerCheck` checks the
   first change of an invocation against the rules in the instructions and the memory
   context before it runs or governance asks anyone (the rules stay text the model reads: no
   rule language is parsed); when every tool result of the request is empty or not found, or
   memory is `INSUFFICIENT`, it asks for a plain "no record". What the assistant can do beyond
   its tools is declared in code (`capabilities=`), since the harness cannot know it.
4. **The check widens a narrowed run.** When tool hints narrowed what the run offers, the
   check offers every tool (only the writes past 128), so a part the hints did not foresee can
   be served; for that `AnswerCheck` sits outside `HarnessTools`.
5. **Native first.** LangChain's `LLMToolSelectorMiddleware` (many tools), `TodoListMiddleware`
   (a plan) and `tool_choice` are used and documented rather than rebuilt; a first call forced
   to `"required"` is not offered, since a request needing no tool would be made to call one.
6. **Coverage per item.** `tool_coverage()` reads each item's `needed_tools`, and scores a run
   that paused for approval (the call it paused on counts as made).

## Consequences

* Off by default, an agent's behaviour and cost are unchanged until its team turns the guards
  on; `AnswerCheck` costs one more model call per run, `ToolGuidance` a fixed block of
  instructions.
* What they cannot fix is two tools described alike: the check of the tool set points at them,
  and their author must say when to use which.
* The OpenAI Agents and Claude Agent SDK targets get the descriptions, the check of the set and
  the hints, not the middleware.
* `examples/06_scenarios/tool_choice.py` and `tests/live/test_live_tool_choice.py` reproduce
  the four failures with generic tools and measure coverage with and without the guards.
