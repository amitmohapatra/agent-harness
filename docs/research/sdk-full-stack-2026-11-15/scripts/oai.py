# ruff: noqa: PLC0415 - a research script: it reports with print, imports an SDK where a case needs it
"""OpenAI Agents SDK 0.24: its own features as its docs show them, on its own (``bare``:
``Runner.run``) and wrapped (``off``, ``stack``). Haiku through Anthropic's OpenAI-compatible
chat completions API (a model object of the case's own: ``OpenAIChatCompletionsModel``), the
meter in front.

Without a tool catalog an SDK ``FunctionTool`` declares nothing, so under the harness each of
its calls asks a person (governance.md); the cases approve those questions and say so.

Usage: python oai.py [feature[:arm] ...]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from agents import (
    Agent,
    GuardrailFunctionOutput,
    ModelSettings,
    RunConfig,
    RunContextWrapper,
    RunHooks,
    Runner,
    SQLiteSession,
    function_tool,
    input_guardrail,
    output_guardrail,
)
from agents.mcp import MCPServerStdio
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from common import METER, MODEL, case, chosen, key, uid
from openai import AsyncOpenAI
from pydantic import BaseModel
from turns import wrapped

SDK = "openai_agents"
HERE = Path(__file__).resolve().parent


def model() -> OpenAIChatCompletionsModel:
    client = AsyncOpenAI(base_url=f"{METER}/v1/", api_key=key())
    return OpenAIChatCompletionsModel(model=MODEL, openai_client=client)


async def go(agent: Agent, arm: str, questions: list[str], name: str, **kw: Any) -> dict[str, Any]:
    """Each question a turn: ``Runner.run`` (bare, the run's options as given) or wrapped."""
    if arm != "bare":
        return await wrapped(agent, arm, name, questions, **kw)
    opts = dict((kw.get("run") or {}).get("framework_options") or {})
    opts.update((kw.get("wrap") or {}).get("framework_options") or {})
    answers, errors, extra = [], [], {}
    for q in questions:
        try:
            r = await Runner.run(agent, q, **opts)
            while r.interruptions:  # needs_approval: approve, continue the state
                extra["asked"] = [i.raw_item.name for i in r.interruptions]
                state = r.to_state()
                for i in r.interruptions:
                    state.approve(i)
                r = await Runner.run(agent, state, **opts)
            answers.append(r.final_output)
            errors.append(None)
        except Exception as exc:  # the SDK's own outcome (a tripwire, max turns)
            answers.append(None)
            errors.append(f"{type(exc).__name__}: {exc}"[:300])
    return {"answers": answers, "errors": errors, **extra}


def stock_tool(calls: list[str]) -> Any:
    @function_tool
    def stock(sku: str) -> str:
        """How many units of a SKU are in stock."""
        calls.append(sku)
        return f"{sku}: 7 units in stock"

    return stock


async def tools(arm: str) -> dict[str, Any]:
    calls: list[str] = []
    agent = Agent(
        name="clerk",
        instructions="Answer stock questions with the tool.",
        model=model(),
        tools=[stock_tool(calls)],
    )
    got = await go(agent, arm, ["How many units of SKU-1 are in stock?"], "oa-tools")
    return {"ok": "7" in str(got["answers"][-1]) and calls == ["SKU-1"], "calls": calls, **got}


class Weather(BaseModel):
    city: str
    celsius: float


async def structured(arm: str) -> dict[str, Any]:
    @function_tool
    def temperature(city: str) -> float:
        """Today's temperature in a city, in Celsius."""
        return 21.5

    agent = Agent(
        name="wx",
        instructions="Report the weather.",
        model=model(),
        tools=[temperature],
        output_type=Weather,
    )
    got = await go(agent, arm, ["What is the temperature in Oslo?"], "oa-structured")
    a = got["answers"][-1]
    w = a if isinstance(a, Weather) else None
    if w is None and isinstance(a, dict):  # the harness keeps a pydantic output as JSON
        w = Weather.model_validate(a)
    return {"ok": w is not None and w.celsius == 21.5, "type": type(a).__name__, **got}


async def stream(arm: str) -> dict[str, Any]:
    agent = Agent(name="writer", instructions="Write as asked.", model=model())
    q = "Write three short sentences about rivers."
    deltas, final = 0, ""
    if arm == "bare":
        r = Runner.run_streamed(agent, q)
        async for ev in r.stream_events():
            if ev.type == "raw_response_event" and getattr(ev.data, "type", "") == (
                "response.output_text.delta"
            ):
                deltas += 1
                final += ev.data.delta
    else:
        from common import harness

        async with harness(arm) as h:
            async for ev in h.wrap(agent, id="oa-stream").stream(q, user=f"oa-{uid()}"):
                if str(ev.type).endswith("TEXT_MESSAGE_CONTENT"):
                    deltas += 1
                    final += (ev.data or {}).get("delta", "")
    return {"ok": deltas > 1 and len(final) > 40, "deltas": deltas, "text": final[:300]}


async def handoff(arm: str) -> dict[str, Any]:
    made: list[str] = []

    @function_tool
    def refund(order: str) -> str:
        """Refund an order."""
        made.append(order)
        return f"refunded {order}"

    refunds = Agent(
        name="refunds",
        instructions="You refund orders with the refund tool.",
        model=model(),
        tools=[refund],
    )
    triage = Agent(
        name="triage",
        instructions="Hand refund requests to the refunds agent.",
        model=model(),
        handoffs=[refunds],
    )
    got = await go(triage, arm, ["Please refund order O-9."], "oa-handoff")
    return {"ok": made == ["O-9"], "made": made, **got}


async def agent_tool(arm: str) -> dict[str, Any]:
    calls: list[str] = []
    checker = Agent(
        name="stock_checker",
        instructions="Look the SKU up with the tool.",
        model=model(),
        tools=[stock_tool(calls)],
    )
    lead = Agent(
        name="lead",
        instructions="Use the stock_checker tool for stock questions.",
        model=model(),
        tools=[
            checker.as_tool(tool_name="stock_checker", tool_description="Checks a SKU's stock.")
        ],
    )
    got = await go(lead, arm, ["How many units of SKU-4 are in stock?"], "oa-agenttool")
    return {"ok": calls == ["SKU-4"] and "7" in str(got["answers"][-1]), "calls": calls, **got}


async def input_guard(arm: str) -> dict[str, Any]:
    @input_guardrail
    async def no_homework(ctx: Any, agent: Agent, given: Any) -> GuardrailFunctionOutput:
        trip = "homework" in str(given).lower()
        return GuardrailFunctionOutput(output_info={"homework": trip}, tripwire_triggered=trip)

    agent = Agent(name="tutor", instructions="Help.", model=model(), input_guardrails=[no_homework])
    got = await go(
        agent,
        arm,
        ["Do my math homework: 2+2?", "What is the capital of France?"],
        "oa-inguard",
        thread=None,
    )
    tripped = "Tripwire" in str(got["errors"][0]) or "guardrail" in str(got["errors"][0]).lower()
    return {"ok": tripped and "paris" in str(got["answers"][1]).lower(), **got}


async def output_guard(arm: str) -> dict[str, Any]:
    @output_guardrail
    async def no_numbers(ctx: Any, agent: Agent, out: Any) -> GuardrailFunctionOutput:
        trip = any(c.isdigit() for c in str(out))
        return GuardrailFunctionOutput(output_info={"digits": trip}, tripwire_triggered=trip)

    agent = Agent(
        name="a", instructions="Answer briefly.", model=model(), output_guardrails=[no_numbers]
    )
    got = await go(agent, arm, ["What is 6 times 7? Answer with digits."], "oa-outguard")
    tripped = "guardrail" in str(got["errors"][0]).lower()
    return {"ok": tripped, **got}


async def session(arm: str) -> dict[str, Any]:
    own = SQLiteSession(f"s-{uid()}")
    agent = Agent(name="a", instructions="Be brief.", model=model())
    qs = ["My favourite colour is teal. Just say noted.", "What is my favourite colour?"]
    got = await go(agent, arm, qs, "oa-session", wrap={"framework_options": {"session": own}})
    return {"ok": "teal" in str(got["answers"][-1]).lower(), **got}


async def needs_approval(arm: str) -> dict[str, Any]:
    made: list[str] = []

    @function_tool(needs_approval=True)
    def delete_user(user_id: str) -> str:
        """Delete a user account."""
        made.append(user_id)
        return f"deleted {user_id}"

    agent = Agent(
        name="admin",
        instructions="Do as asked with your tools.",
        model=model(),
        tools=[delete_user],
    )
    got = await go(agent, arm, ["Delete the user account U-12."], "oa-approval")
    asked = bool(got.get("asked"))
    return {"ok": asked and made == ["U-12"], "made": made, **got}


async def hooks(arm: str, given: str = "wrap") -> dict[str, Any]:
    """``RunHooks`` given on ``h.wrap`` (every run's: a resume after an approval keeps them);
    ``hooks_per_run`` gives them on ``run`` (an object is that call's only, runs.md)."""
    seen: list[str] = []

    class Mine(RunHooks):
        async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
            seen.append("llm_end")

        async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
            seen.append(f"tool_start:{tool.name}")

    calls: list[str] = []
    agent = Agent(
        name="clerk", instructions="Answer with the tool.", model=model(), tools=[stock_tool(calls)]
    )
    got = await go(
        agent,
        arm,
        ["How many units of SKU-2?"],
        "oa-hooks",
        **{given: {"framework_options": {"hooks": Mine()}}},
    )
    ok = "tool_start:stock" in seen and seen.count("llm_end") >= 2
    return {"ok": ok, "hooks_seen": seen, **got}


async def max_turns(arm: str) -> dict[str, Any]:
    calls: list[str] = []
    agent = Agent(
        name="clerk",
        instructions="Check every SKU the user names, one call each.",
        model=model(),
        tools=[stock_tool(calls)],
    )
    got = await go(
        agent,
        arm,
        ["Check SKU-1, then SKU-2, then SKU-3, one tool call at a time."],
        "oa-maxturns",
        run={"framework_options": {"max_turns": 1}},
    )
    stopped = "MaxTurns" in str(got["errors"]) or "PARTIAL" in str(got.get("statuses"))
    return {"ok": stopped, "calls": calls, **got}


async def mcp(arm: str) -> dict[str, Any]:
    params = {"command": sys.executable, "args": [str(HERE / "mcp_server.py")]}
    async with MCPServerStdio(params=params, name="warehouse") as server:  # type: ignore[arg-type]
        agent = Agent(
            name="clerk",
            instructions="Use the warehouse tools.",
            model=model(),
            mcp_servers=[server],
        )
        got = await go(
            agent, arm, ["How many units of SKU-7 are in the central warehouse?"], "oa-mcp"
        )
    return {"ok": "12" in str(got["answers"][-1]), **got}


class Account:
    def __init__(self, number: str) -> None:
        self.number = number


async def context(arm: str, given: str = "wrap") -> dict[str, Any]:
    @function_tool
    def balance(ctx: RunContextWrapper[Account]) -> str:
        """The caller's account balance."""
        return f"account {ctx.context.number}: 420 EUR"

    agent = Agent[Account](
        name="bank", instructions="Answer with the tool.", model=model(), tools=[balance]
    )
    got = await go(
        agent,
        arm,
        ["What is my balance?"],
        "oa-context",
        **{given: {"framework_options": {"context": Account("AC-9")}}},
    )
    return {"ok": "420" in str(got["answers"][-1]), **got}


async def hooks_per_run(arm: str) -> dict[str, Any]:
    return await hooks(arm, "run")


async def context_per_run(arm: str) -> dict[str, Any]:
    return await context(arm, "run")


async def tool_use_behavior(arm: str) -> dict[str, Any]:
    calls: list[str] = []
    agent = Agent(
        name="clerk",
        instructions="Use the tool.",
        model=model(),
        tools=[stock_tool(calls)],
        tool_use_behavior="stop_on_first_tool",
    )
    got = await go(agent, arm, ["Stock of SKU-5?"], "oa-stopfirst")
    return {"ok": got["answers"][-1] == "SKU-5: 7 units in stock", **got}


async def model_settings(arm: str) -> dict[str, Any]:
    agent = Agent(
        name="a", instructions="Answer.", model=model(), model_settings=ModelSettings(max_tokens=16)
    )
    got = await go(agent, arm, ["Write a 500-word essay about the sea."], "oa-settings")
    words = len(str(got["answers"][-1]).split())
    return {"ok": 0 < words <= 20, "words": words, **got}


def sandbox_agent(client: Any, options: Any = None) -> tuple[Agent, dict[str, Any]]:
    from agents.sandbox import SandboxAgent, SandboxRunConfig
    from agents.sandbox.capabilities import Shell

    agent = SandboxAgent(
        name="box", instructions="Use the shell to answer.", model=model(), capabilities=[Shell()]
    )
    cfg = RunConfig(sandbox=SandboxRunConfig(client=client, options=options))
    return agent, {"framework_options": {"run_config": cfg}}


async def sandbox_local(arm: str) -> dict[str, Any]:
    from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

    agent, opts = sandbox_agent(UnixLocalSandboxClient())
    got = await go(
        agent,
        arm,
        ["Run `echo $((21*2))` in the shell and tell me the output."],
        "oa-sbx-local",
        wrap=opts,
    )
    return {"ok": "42" in str(got["answers"][-1]), **got}


async def sandbox_docker(arm: str) -> dict[str, Any]:
    import docker
    from agents.sandbox.sandboxes.docker import DockerSandboxClient, DockerSandboxClientOptions

    client = DockerSandboxClient(docker.from_env())
    agent, opts = sandbox_agent(client, DockerSandboxClientOptions(image="python:3.12-slim"))
    got = await go(
        agent,
        arm,
        [
            "Run `python3 -c 'import sys; print(sys.version_info[:2])'` "
            "in the shell and tell me the output."
        ],
        "oa-sbx-docker",
        wrap=opts,
    )
    return {"ok": "3, 12" in str(got["answers"][-1]) or "3.12" in str(got["answers"][-1]), **got}


FEATURES: dict[str, Any] = {
    "tools": {"bare", "off", "stack"},
    "structured": {"bare", "off", "stack"},
    "stream": {"bare", "off"},
    "handoff": {"bare", "off", "stack"},
    "agent_tool": {"bare", "off", "stack"},
    "input_guard": {"bare", "off", "stack"},
    "output_guard": {"bare", "off"},
    "session": {"bare", "off", "stack"},
    "needs_approval": {"bare", "off", "stack"},
    "hooks": {"bare", "off"},
    "max_turns": {"bare", "off"},
    "mcp": {"bare", "off", "stack"},
    "context": {"bare", "off"},
    "hooks_per_run": {"bare", "off"},
    "context_per_run": {"bare", "off"},
    "tool_use_behavior": {"bare", "off"},
    "model_settings": {"bare", "off"},
    "sandbox_local": {"bare", "off"},
    "sandbox_docker": {"bare", "off", "stack"},
}


async def main() -> None:
    for feature, arm in chosen(sys.argv[1:], FEATURES):
        await case(SDK, feature, arm, lambda f=feature, a=arm: globals()[f](a))


if __name__ == "__main__":
    asyncio.run(main())
