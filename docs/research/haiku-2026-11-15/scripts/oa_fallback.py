"""OpenAI Agents through the harness's Bifrost route, pointed at a gateway stand-in that has no
Responses route (Anthropic's OpenAI-compatible API behind meter_proxy.py): the first call
learns /v1/responses is not served (404) and falls back to chat completions; later calls go
straight there; a hosted tool on the fallback is refused before any call.

Usage: ANTHROPIC_KEY_FILE=<file> METER=<base url of meter_proxy> python oa_fallback.py <out>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
KEY = Path(os.environ.pop("ANTHROPIC_KEY_FILE")).read_text().strip()
for name in list(os.environ):
    if name.startswith(("LLM_", "OPENAI_", "ANTHROPIC_", "MEMORY_URL", "RUNS_URL")):
        del os.environ[name]

from agents import Agent, WebSearchTool, function_tool  # noqa: E402

from trellis import Harness, Settings  # noqa: E402
from trellis.harness import budget  # noqa: E402

ACCOUNTED: list[dict[str, Any]] = []
_account = budget.account


def _accounting(runtime: Any, **kw: Any) -> None:
    _account(runtime, **kw)
    ACCOUNTED.append({"run": runtime.run_id, **{k: v for k, v in kw.items() if k != "model"}})


budget.account = _accounting  # type: ignore[assignment]


@function_tool
def stock_level(sku: str) -> str:
    """Units on hand of a SKU.

    Args:
        sku: the SKU id, e.g. SKU-00938
    """
    return json.dumps({"sku": sku, "on_hand": 314})


async def main() -> None:
    config = Settings(
        llm_gateway="bifrost",
        llm_base_url=f"{os.environ['METER']}/v1/",
        llm_api_key=KEY,
        runs_url=None,
        memory_url=None,
    )
    found: dict[str, Any] = {}
    async with Harness(config=config, memory=False) as h:
        agent = Agent(
            name="stock",
            instructions="Answer with the stock_level tool. Be brief.",
            model="claude-haiku-5-5",
            tools=[stock_level],
        )
        wrapped = h.wrap(agent, id=f"oa-fallback-{uuid.uuid4().hex[:6]}")
        for n in (1, 2):
            res = await wrapped.run("How many units of SKU-00938 are on hand?", user="probe")
            asked = (
                res.interrupt.tool_call.tool if res.interrupt and res.interrupt.tool_call else None
            )
            if res.interrupt is not None:  # an undeclared function tool asks: approved
                res = await wrapped.resume(res.interrupt.interrupt_id, "approve", reviewer="lead")
            found[f"run_{n}"] = {
                "run": res.run_id,
                "status": res.status.value,
                "asked_about": asked,
                "answer": str(res.answer)[:300],
                "error": res.error.message[:300] if res.error else None,
            }
        hosted = Agent(
            name="searcher",
            instructions="Search the web.",
            model="claude-haiku-5-5",
            tools=[WebSearchTool()],
        )
        try:
            res = await h.wrap(hosted, id=f"oa-hosted-{uuid.uuid4().hex[:6]}").run(
                "What's the weather in Leeds?", user="probe"
            )
            found["hosted_tool_on_fallback"] = {
                "status": res.status.value,
                "error": res.error.message[:400] if res.error else None,
            }
        except Exception as exc:
            found["hosted_tool_on_fallback"] = {"raised": f"{type(exc).__name__}: {exc}"[:400]}
    found["accounted"] = ACCOUNTED
    (OUT / "oa_fallback.json").write_text(json.dumps(found, indent=2, default=str))
    sys.stdout.write(json.dumps(found, default=str)[:2000] + "\n")


asyncio.run(main())
