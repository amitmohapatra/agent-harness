"""What the examples need to run with no services: ``make examples`` runs every one this way.

Each piece is real when its variable is set and scripted when it is not:

* the model — with ``BIFROST_URL``, a real one through Bifrost's OpenAI-compatible endpoint
  (:data:`MODEL`); else a scripted one that answers from a fixed script. The harness, the
  frameworks and the tools are always real; only the model's choices are written down;
* the memory service — with ``MEMORY_URL`` (and ``TRELLIS_API_KEY``), the real one; else
  :class:`~examples._support.memory.ScriptedMemory`, in process (:func:`offline_blocks`,
  :func:`memory_client`);
* the gateway's MCP tools — with ``BIFROST_URL``, the tools its virtual key allows; else
  :class:`~examples._support.gateway.ScriptedGateway`, serving the example's own functions;
* runs — with ``RUNS_URL``, agent-runs; else the harness's in-process store, which takes the
  same calls (:func:`runs_store`);
* the judge — with ``BIFROST_URL``, ``TRELLIS_JUDGE_MODEL``; else :class:`Judging`.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from examples._support.gateway import McpTool, ScriptedGateway
from examples._support.memory import ScriptedMemory

from trellis import Harness
from trellis.harness.evals import EvalServices
from trellis.harness.runs import LocalRuns, RunStore
from trellis.memory import MemoryClient
from trellis.runs import RunsClient

#: The model the examples use through Bifrost.
MODEL = "openrouter/openai/gpt-4.1-nano"
MAX_TOKENS = 2048
#: A stand-in for the Claude Code CLI that speaks its stream-json protocol.
FAKE_CLAUDE_CLI = str(
    Path(__file__).resolve().parents[2] / "tests" / "support" / "fake_claude_cli.py"
)

Turn = str | tuple[str, dict[str, Any]]
#: a turn may also make several calls in one message
Turns = Turn | list[tuple[str, dict[str, Any]]]


def online() -> bool:
    return bool(os.environ.get("BIFROST_URL"))


def gateway() -> tuple[str, str]:
    """Bifrost's base URL and virtual key, for a framework's own OpenAI client."""
    return os.environ["BIFROST_URL"], os.environ.get("BIFROST_VIRTUAL_KEY") or "unused"


# --------------------------------------------------------------------------- LangChain
def langchain_model(turns: Sequence[Turns]) -> Any:
    """A chat model for LangGraph, Deep Agents and ``ReAct``: Bifrost when online, else the
    script (a turn may make several calls at once)."""
    if online():
        from langchain_openai import ChatOpenAI

        url, key = gateway()
        return ChatOpenAI(model=MODEL, base_url=url, api_key=key, max_completion_tokens=MAX_TOKENS)  # type: ignore[arg-type]
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    script = list(turns)

    class Scripted(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "scripted"

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            return self

        def _generate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> ChatResult:
            turn = script.pop(0)
            if isinstance(turn, str):
                message = AIMessage(content=turn)
            else:
                calls = turn if isinstance(turn, list) else [turn]
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {"name": name, "args": args, "id": f"call_{len(script)}_{n}"}
                        for n, (name, args) in enumerate(calls)
                    ],
                )
            return ChatResult(generations=[ChatGeneration(message=message)])

    return Scripted()


# --------------------------------------------------------------------------- OpenAI Agents
def openai_agents_model(turns: Sequence[Turn]) -> Any:
    """A model for the OpenAI Agents SDK: Bifrost when online, else the script."""
    from agents import Model, ModelResponse, OpenAIChatCompletionsModel, Usage
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    if online():
        from openai import AsyncOpenAI

        url, key = gateway()
        return OpenAIChatCompletionsModel(
            model=MODEL, openai_client=AsyncOpenAI(base_url=url, api_key=key)
        )
    script = list(turns)

    class Scripted(Model):
        async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
            turn = script.pop(0)
            if isinstance(turn, str):
                item: Any = ResponseOutputMessage(
                    id="msg",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[ResponseOutputText(type="output_text", text=turn, annotations=[])],
                )
            else:
                item = ResponseFunctionToolCall(
                    id="fc",
                    call_id=f"call_{len(script)}",
                    type="function_call",
                    name=turn[0],
                    arguments=json.dumps(turn[1]),
                    status="completed",
                )
            return ModelResponse(output=[item], usage=Usage(), response_id=None)

        def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            raise NotImplementedError("the offline examples do not stream this model")

    return Scripted()


# --------------------------------------------------------------------------- ReAct
def react_model(turns: Sequence[Turns]) -> Any:
    """A model name for Bifrost when online (``ReAct`` makes it a gateway model), else the
    script."""
    return MODEL if online() else langchain_model(turns)


# --------------------------------------------------------------------------- evaluation
def answering_model(answers: dict[str, str]) -> Any:
    """A model name for Bifrost when online, else a chat model answering each question from
    the table."""
    if online():
        return MODEL
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class Answering(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "answering"

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            return self

        def _generate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> ChatResult:
            question = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            answer = AIMessage(content=answers.get(str(question), "I don't know."))
            return ChatResult(generations=[ChatGeneration(message=answer)])

    return Answering()


class Judging:
    """A chat-completions endpoint for the evaluation judge (``llm_judge``): it grades an
    answer 1 when it contains the expected one (with nothing expected: when it is an answer at
    all), else 0, as the strict JSON the judge asks for."""

    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
        content = self._grade(str(messages[-1]["content"]))
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}

    @staticmethod
    def _grade(prompt: str) -> str:
        sections = dict(part.split("\n", 1) for part in prompt.split("## ")[1:])
        expected = sections.get("Expected answer", "").strip().casefold()
        graded = sections.get("Answer to grade", "").strip().casefold()
        good = expected in graded if expected else graded not in ("", "i don't know.")
        reasoning = "names the expected answer" if good else "does not name the expected answer"
        return json.dumps({"score": 1.0 if good else 0.0, "reasoning": reasoning})


def judge_offline(h: Harness) -> None:
    """Offline, the harness's judge is the scripted one (online: ``TRELLIS_JUDGE_MODEL``)."""
    if not online():
        h.evals.judge_model = Judging()


def judge_services() -> EvalServices:
    """Langfuse and the judge the environment names; offline, the scripted judge."""
    return EvalServices.from_env() if online() else EvalServices(judge_model=Judging())


# --------------------------------------------------------------------------- Way 2 blocks
#: What the Way 2 examples call ``runs``: agent-runs' ``RunsClient``, or offline the harness's
#: in-process store, which takes the same calls with the same signatures. Your code types it
#: ``RunsClient``.
Runs = RunStore


def runs_store() -> Runs:
    """agent-runs' client with ``RUNS_URL``; else the harness's in-process store, which takes
    the same calls (``start``, ``pause``, ``iterate``, ``resume``, ``finish``, ``claim``...)."""
    return RunsClient() if os.environ.get("RUNS_URL") else LocalRuns()


def memory_client(memory: ScriptedMemory | None = None) -> MemoryClient:
    """The memory service's client with ``MEMORY_URL`` (and ``TRELLIS_API_KEY``), else one on
    the scripted service (``memory``, or a new one)."""
    if os.environ.get("MEMORY_URL"):
        return MemoryClient()
    return (memory or ScriptedMemory()).client()


def offline_blocks(
    *,
    memory: ScriptedMemory | None = None,
    mcp: dict[str, McpTool] | None = None,
    code_mode: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """The blocks the environment does not name, scripted, for ``Harness(**offline_blocks())``:
    the memory service unless ``MEMORY_URL`` is set, the gateway (serving ``mcp``) unless
    ``BIFROST_URL`` is. What is set is the deployment's, as ``Harness()`` reads it."""
    blocks: dict[str, Any] = {}
    if not os.environ.get("MEMORY_URL"):
        blocks["memory"] = (memory or ScriptedMemory()).client()
    if not online() and mcp:
        blocks["gateway"] = ScriptedGateway(mcp, code_mode=code_mode).gateway()
    return blocks


def claude_cli(script: Sequence[dict[str, Any]], server: str) -> dict[str, Any]:
    """``ClaudeAgentOptions`` arguments for the CLI: the real ``claude`` through Bifrost's
    Anthropic route when online, else the scripted stand-in calling tools of ``server``."""
    if online():
        url, key = gateway()
        origin = url.removesuffix("/v1")
        return {"env": {"ANTHROPIC_BASE_URL": f"{origin}/anthropic", "ANTHROPIC_API_KEY": key}}
    environment = {"FAKE_CLAUDE_SCRIPT": json.dumps(list(script)), "FAKE_CLAUDE_SERVER": server}
    return {"cli_path": FAKE_CLAUDE_CLI, "env": environment}
