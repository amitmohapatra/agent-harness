"""ReAct: reason, act, observe, repeat (Yao et al., 2023) — over the harness's own runtime.

The loop is four lines of logic and a great deal of care about when to stop::

    answer = await react(runtime, "who leads the freight operator we use?")

What makes this worth having in the harness rather than in every application:

* **Tool errors are observations, not failures.** A tool that raises hands the model the
  error text and lets it try something else. That is the whole idea of ReAct — a loop that
  aborts on the first bad argument is just a chain with extra steps.
* **It is bounded three ways.** ``max_steps`` caps reasoning, the execution's own deadline
  caps wall-clock (``runtime.remaining_seconds``), and a model that stops asking for tools
  ends the loop. An unbounded agent loop is how a chat node holds a graph open forever.
* **Every step is a span.** The steps nest under ``agent.run``, so Langfuse shows the
  reasoning trace and OTel shows the latency of each hop without the application doing
  anything.
* **The model must implement the port.** ``AgentHarness`` passes an object through only if
  it has both ``invoke`` and ``structured``; anything else is wrapped in ``DirectModelClient``,
  which calls it with the bare message list and drops ``tools`` entirely. A loop given such a
  model still works, but the model is never *told* about the tools, so it cannot ask for one.
* **Native tool calls, with a text fallback.** Models that support ``tools`` answer with a
  structured ``tool_calls`` list; older ones answer with prose. Parsing prose is a fallback,
  not the protocol, because the structured path cannot be confused by a model that merely
  *talks* about calling a tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from universal_agent_contracts.errors import HarnessError, ModelError
from universal_agent_contracts.messages import AgentResponse
from universal_agent_contracts.model import ModelRequest
from universal_agent_contracts.tool import ToolSpec

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_harness.runtime.agent_runtime import AgentRuntime

#: What the model is told about the shape of the job. Deliberately short: a long preamble
#: competes with the user's own instructions for the model's attention, and every token here
#: is paid on every step of the loop, not once.
SYSTEM = (
    "You answer questions by calling tools when they help and answering directly when they "
    "do not. Call one tool at a time and use its result before deciding the next step. "
    "When you have enough to answer, answer in plain language without calling a tool."
)

#: A tool result longer than this is truncated before it goes back to the model. Observations
#: are the fastest-growing part of a ReAct prompt: a loop that pastes whole documents back in
#: spends its context on step 3 and has none left to reason with.
MAX_OBSERVATION_CHARS = 4000


@dataclass(slots=True)
class ReActStep:
    """One reason/act/observe cycle."""

    number: int
    thought: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    observation: str | None = None
    failed: bool = False


@dataclass(slots=True)
class ReActTrace:
    """The whole loop, for assertions and for the caller's own logging."""

    steps: list[ReActStep] = field(default_factory=list)
    answer: str | None = None
    stopped_because: str = "answered"

    @property
    def tool_names(self) -> list[str]:
        return [s.tool for s in self.steps if s.tool]


async def react(
    runtime: AgentRuntime,
    question: str,
    *,
    max_steps: int = 6,
    system: str = SYSTEM,
    tools: list[ToolSpec] | None = None,
    model: str | None = None,
) -> AgentResponse:
    """Run a bounded ReAct loop and return the model's final answer.

    The :class:`ReActTrace` is always left on ``runtime.state["react"]`` — the scratch space
    the harness already gives interceptors. Callers that only want the answer ignore it;
    callers that need to tell a real tool-using answer from a confident guess read it, and
    that distinction is invisible in the answer text alone.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    available = list(tools) if tools is not None else await runtime.tools.list_tools()
    schemas = _schemas(available)
    turns: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    trace = ReActTrace()

    for number in range(1, max_steps + 1):
        runtime.check_cancelled()
        if (left := runtime.remaining_seconds) is not None and left <= 0:
            trace.stopped_because = "deadline"
            break

        with runtime.tracer.span("agent.react.step", attributes={"react.step": number}) as span:
            response = await runtime.model.invoke(_request(turns, model, schemas))
            call = _first_call(response)
            step = ReActStep(number=number, thought=(response.text or "").strip() or None)

            if call is None:
                # No tool asked for: the model is answering, and the loop is done.
                span.set(**{"react.terminal": True})
                trace.steps.append(step)
                trace.answer = (response.text or "").strip()
                break

            step.tool, step.arguments = call
            span.set(**{"react.tool": step.tool})
            turns.append(_assistant_turn(response, step))

            outcome = await _observe(runtime, step)
            turns.append({"role": "user", "content": f"Observation: {step.observation}"})
            span.set(**{"react.failed": step.failed})
            trace.steps.append(step)
            runtime.log(
                "react.step",
                step=number,
                tool=step.tool,
                failed=step.failed,
                status=getattr(outcome, "status", None),
            )
    else:
        trace.stopped_because = "max_steps"

    if trace.answer is None and trace.stopped_because == "answered":
        trace.stopped_because = "max_steps"
    if trace.answer is None:
        # Out of steps or time with tool results in hand: ask once for the answer rather than
        # returning nothing. The budget bought observations; throwing them away wastes it.
        trace.answer = await _final_answer(runtime, turns, model)

    runtime.state["react"] = trace
    return AgentResponse.ok(trace.answer)


# ---------------------------------------------------------------------------- internals
def _request(
    turns: list[dict[str, Any]], model: str | None, schemas: list[dict[str, Any]] | None
) -> ModelRequest:
    """A request, omitting ``tools`` rather than sending an empty one.

    ``ModelRequest.tools`` is a list, not an optional: passing ``None`` fails validation, and
    passing ``[]`` tells the gateway the agent has no tools when the truth may be that this
    particular call is the final answer and is withholding them on purpose.
    """
    request: dict[str, Any] = {"messages": list(turns), "model": model}
    if schemas:
        request["tools"] = schemas
    return ModelRequest(**request)


def _schemas(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Tool specs in the OpenAI ``tools`` shape every gateway speaks."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or t.name,
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def _first_call(response: Any) -> tuple[str, dict[str, Any]] | None:
    """The tool the model asked for, if it asked for one.

    One at a time on purpose: parallel calls read well in a demo and make the observation
    order non-deterministic, which is the hardest kind of agent bug to reproduce.
    """
    for raw in response.tool_calls or []:
        function = raw.get("function") or raw
        name = function.get("name")
        if not name:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except ValueError:
                arguments = {}
        return str(name), dict(arguments or {})
    return None


def _assistant_turn(response: Any, step: ReActStep) -> dict[str, Any]:
    """The model's own turn, replayed back to it so the next step has the context."""
    return {
        "role": "assistant",
        "content": response.text or f"Calling {step.tool}({json.dumps(step.arguments)})",
    }


async def _observe(runtime: AgentRuntime, step: ReActStep) -> Any:
    """Call the tool and turn whatever happens into text the model can act on."""
    try:
        outcome = await runtime.tools.call(step.tool or "", **step.arguments)
    except Exception as exc:  # an error is an observation, not a failure (module docstring)
        step.failed = True
        step.observation = f"{type(exc).__name__}: {exc}"[:MAX_OBSERVATION_CHARS]
        return None
    step.failed = outcome.status != "ok"
    step.observation = _render(outcome)[:MAX_OBSERVATION_CHARS]
    return outcome


def _render(outcome: Any) -> str:
    if outcome.output_summary:
        return str(outcome.output_summary)
    output = outcome.output
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, default=str)
    except (TypeError, ValueError):
        return str(output)


async def _final_answer(
    runtime: AgentRuntime, turns: list[dict[str, Any]], model: str | None
) -> str:
    """One last call with tools withheld, so the model must answer instead of acting."""
    try:
        response = await runtime.model.invoke(
            _request(
                [
                    *turns,
                    {
                        "role": "user",
                        "content": "Answer now in plain language using what you have. "
                        "Do not call any more tools.",
                    },
                ],
                model,
                None,
            )
        )
    except HarnessError:
        raise  # already classified; re-wrapping would lose the category
    except Exception as exc:
        raise ModelError(f"react loop could not produce an answer: {exc}") from exc
    return (response.text or "").strip()
