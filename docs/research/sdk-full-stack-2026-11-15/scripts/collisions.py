# ruff: noqa: T201 - a research script: it reports with print, imports an SDK where a case needs it
"""Part C without a model call: which of a library's own middleware ``trellis.middleware()``
collides with when both are given (each graph is only built). Writes data/collisions.json."""

from __future__ import annotations

import json
from pathlib import Path

from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolRetryMiddleware,
)
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from lc_agent import middleware_on, model

from trellis import ReAct

OWN = {
    "TodoListMiddleware": TodoListMiddleware,
    "ContextEditingMiddleware": ContextEditingMiddleware,
    "ModelCallLimitMiddleware": lambda: ModelCallLimitMiddleware(run_limit=5),
    "AnthropicPromptCachingMiddleware": AnthropicPromptCachingMiddleware,
    "SummarizationMiddleware": lambda: SummarizationMiddleware(model=model()),
    "ToolRetryMiddleware": ToolRetryMiddleware,
}


def built(make) -> str:
    try:
        make()
        return "built"
    except Exception as exc:  # what the user meets when the graph is built
        return f"{type(exc).__name__}: {str(exc)[:160]}"


out: dict[str, dict[str, str]] = {}


def tried(own) -> dict[str, str]:
    return {
        "create_agent + middleware()": built(
            lambda: create_agent(model(), tools=[], middleware=[own(), *middleware_on()])
        ),
        "create_deep_agent + middleware()": built(
            lambda: create_deep_agent(model(), tools=[], middleware=[own(), *middleware_on()])
        ),
        "ReAct(middleware=[own])": built(
            lambda: ReAct("You help.", "claude-haiku-5-5", middleware=[own()])
        ),
    }


for name, own in OWN.items():
    out[name] = tried(own)
path = Path(__file__).resolve().parent.parent / "data" / "collisions.json"
path.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
