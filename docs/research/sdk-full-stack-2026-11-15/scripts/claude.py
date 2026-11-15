# ruff: noqa: PLC0415 - a research script: it reports with print, imports an SDK where a case needs it
"""Claude Agent SDK 0.2.182 (bundled CLI 2.1.325): its own features as its docs show them, on
its own (``bare``: ``query`` / ``ClaudeSDKClient``) and wrapped (``off``, ``stack``). The CLI
runs in a home of its own, no settings files (``setting_sources=[]``), a working folder per
case, its model Haiku through the meter.

Two ways: ``query`` (one prompt, a process per run) and the live session (``ClaudeSDKClient``;
under the harness a conversation's turns, the CLI process kept between them).

Usage: python claude.py [feature[:arm] ...]
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultError,
    ResultMessage,
    StreamEvent,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
    tool,
)
from common import METER, MODEL, case, chosen, harness, key, uid
from mcp.types import ToolAnnotations
from turns import wrapped

SDK = "claude_agent_sdk"
HERE = Path(__file__).resolve().parent


def folder(tag: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"cl-{tag}-"))


def options(cwd: Path, **kw: Any) -> ClaudeAgentOptions:
    env = {
        "ANTHROPIC_API_KEY": key(),
        "ANTHROPIC_BASE_URL": METER,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
    return ClaudeAgentOptions(model=MODEL, setting_sources=[], cwd=str(cwd), env=env, **kw)


async def bare(
    opts: ClaudeAgentOptions, prompts: list[str], *, live: bool = False, partial: bool = False
) -> dict[str, Any]:
    """``query`` per prompt (the session resumed for the next: ``resume=``) or, ``live``, one
    ``ClaudeSDKClient`` for every prompt."""
    answers: list[Any] = []
    tools_used: list[str] = []
    subtypes: list[str] = []
    deltas = 0

    def seen(m: Any) -> None:
        nonlocal deltas
        if isinstance(m, AssistantMessage):
            tools_used.extend(b.name for b in m.content if isinstance(b, ToolUseBlock))
        if isinstance(m, StreamEvent) and m.event.get("type") == "content_block_delta":
            deltas += 1
        if isinstance(m, ResultMessage):
            answers.append(m.structured_output or m.result)
            subtypes.append(m.subtype)

    if live:
        async with ClaudeSDKClient(options=opts) as client:
            for p in prompts:
                await client.query(p)
                async for m in client.receive_response():
                    seen(m)
    else:
        session = None
        for p in prompts:
            o = opts if session is None else _with(opts, resume=session)
            try:
                async for m in query(prompt=_stream(p) if opts.can_use_tool else p, options=o):
                    seen(m)
                    if isinstance(m, ResultMessage):
                        session = m.session_id
            except ResultError as exc:  # the SDK's own outcome (its turn limit, say)
                subtypes.append(f"ResultError: {exc}"[:200])
    return {
        "answers": answers,
        "tools_used": tools_used,
        "subtypes": subtypes,
        "deltas": deltas if partial else None,
    }


async def _stream(prompt: str) -> Any:
    yield {"type": "user", "message": {"role": "user", "content": prompt}}


def _with(opts: ClaudeAgentOptions, **kw: Any) -> ClaudeAgentOptions:
    import dataclasses

    return dataclasses.replace(opts, **kw)


async def go(
    opts: ClaudeAgentOptions, arm: str, prompts: list[str], name: str, **kw: Any
) -> dict[str, Any]:
    if arm == "bare":
        return await bare(
            opts, prompts, live=kw.pop("live", False), partial=kw.pop("partial", False)
        )
    kw.pop("live", None)
    kw.pop("partial", None)
    return await wrapped(opts, arm, name, prompts, **kw)


# --------------------------------------------------------------------------- features
def stock_server(calls: list[str]) -> Any:
    @tool(
        "stock",
        "How many units of a SKU are in stock.",
        {"sku": str},
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def stock(args: dict[str, Any]) -> dict[str, Any]:
        calls.append(args["sku"])
        return {"content": [{"type": "text", "text": f"{args['sku']}: 7 units in stock"}]}

    return create_sdk_mcp_server("shop", tools=[stock])


async def sdk_tool(arm: str) -> dict[str, Any]:
    """An in-process SDK MCP server's tool."""
    calls: list[str] = []
    opts = options(
        folder("tool"),
        mcp_servers={"shop": stock_server(calls)},
        allowed_tools=["mcp__shop__stock"],
    )
    got = await go(opts, arm, ["How many units of SKU-1 are in stock?"], "cl-tool")
    return {"ok": calls == ["SKU-1"] and "7" in str(got["answers"][-1]), "calls": calls, **got}


async def builtin_read(arm: str) -> dict[str, Any]:
    cwd = folder("read")
    (cwd / "notes.txt").write_text("The vault combination is 31-14-59.\n")
    opts = options(cwd, allowed_tools=["Read"])
    got = await go(opts, arm, ["Read notes.txt and tell me the vault combination."], "cl-read")
    return {"ok": "31-14-59" in str(got["answers"][-1]), **got}


async def builtin_write(arm: str) -> dict[str, Any]:
    """``permission_mode="acceptEdits"``: Claude writes the file without asking."""
    cwd = folder("write")
    opts = options(cwd, permission_mode="acceptEdits")
    got = await go(opts, arm, ["Create a file hello.txt containing exactly: hi there"], "cl-write")
    made = (cwd / "hello.txt").read_text().strip() if (cwd / "hello.txt").exists() else None
    return {"ok": made == "hi there", "file": made, **got}


async def structured(arm: str) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "population_millions": {"type": "number"}},
        "required": ["city", "population_millions"],
    }
    opts = options(folder("struct"), output_format={"type": "json_schema", "schema": schema})
    got = await go(
        opts,
        arm,
        ["What is the capital of Norway and its population in millions?"],
        "cl-structured",
    )
    a = got["answers"][-1]
    return {"ok": isinstance(a, dict) and "oslo" in str(a.get("city", "")).lower(), **got}


async def own_hooks(arm: str) -> dict[str, Any]:
    """The options' own ``PreToolUse`` and ``PostToolUse`` hooks."""
    seen: list[str] = []

    async def pre(given: Any, tool_use_id: Any, context: Any) -> Any:
        seen.append(f"pre:{given.get('tool_name')}")
        return {}

    async def post(given: Any, tool_use_id: Any, context: Any) -> Any:
        seen.append(f"post:{given.get('tool_name')}")
        return {}

    cwd = folder("hooks")
    (cwd / "a.txt").write_text("alpha\n")
    opts = options(
        cwd,
        allowed_tools=["Read"],
        hooks={
            "PreToolUse": [HookMatcher(matcher="Read", hooks=[pre])],
            "PostToolUse": [HookMatcher(matcher="Read", hooks=[post])],
        },
    )
    got = await go(opts, arm, ["Read a.txt and tell me the word in it."], "cl-hooks")
    ok = "pre:Read" in seen and "post:Read" in seen and "alpha" in str(got["answers"][-1])
    return {"ok": ok, "hooks_seen": seen, **got}


async def can_use_tool(arm: str) -> dict[str, Any]:
    """The options' own ``can_use_tool``: it denies ``Bash``; Claude says it was refused."""
    asked: list[str] = []

    async def gate(name: str, args: dict[str, Any], context: Any) -> Any:
        asked.append(name)
        if name == "Bash":
            return PermissionResultDeny(message="Shell commands are not allowed for you.")
        return PermissionResultAllow()

    opts = options(folder("cut"), can_use_tool=gate)
    got = await go(
        opts,
        arm,
        ["Run the shell command `touch made.txt`."],
        "cl-canuse",
        decide=lambda _i: ("approve", None),
    )
    cwd = Path(opts.cwd or ".")
    return {"ok": "Bash" in asked and not (cwd / "made.txt").exists(), "gate_asked": asked, **got}


async def subagent(arm: str) -> dict[str, Any]:
    calls: list[str] = []
    opts = options(
        folder("sub"),
        mcp_servers={"shop": stock_server(calls)},
        agents={
            "stock-checker": AgentDefinition(
                description="Checks stock of a SKU.",
                prompt="Check the SKU with the stock tool and report.",
                tools=["mcp__shop__stock"],
                model="haiku",
            )
        },
        allowed_tools=["mcp__shop__stock", "Agent", "Task"],
    )
    got = await go(opts, arm, ["Use the stock-checker subagent to check SKU-8."], "cl-sub")
    used = got.get("tools_used") or []
    return {
        "ok": calls == ["SKU-8"] and "7" in str(got["answers"][-1]),
        "calls": calls,
        "delegated": any(t in ("Agent", "Task") for t in used) or None,
        **got,
    }


async def max_turns(arm: str) -> dict[str, Any]:
    calls: list[str] = []
    opts = options(
        folder("turns"),
        mcp_servers={"shop": stock_server(calls)},
        allowed_tools=["mcp__shop__stock"],
        max_turns=1,
    )
    got = await go(
        opts, arm, ["Check SKU-1, then SKU-2, then SKU-3 with the tool, one at a time."], "cl-turns"
    )
    said = str(got.get("subtypes"))
    stopped = (
        "max_turns" in said
        or "maximum number of turns" in said
        or ("PARTIAL" in str(got.get("statuses")))
    )
    return {"ok": stopped, "calls": calls, **got}


async def partial_stream(arm: str) -> dict[str, Any]:
    opts = options(folder("stream"), include_partial_messages=True)
    q = "Write three short sentences about rivers."
    if arm == "bare":
        got = await bare(opts, [q], partial=True)
        deltas = got["deltas"]
    else:
        deltas, got = 0, {}
        async with harness(arm) as h:
            async for ev in h.wrap(opts, id="cl-stream").stream(q, user=f"cl-{uid()}"):
                if str(ev.type).endswith("TEXT_MESSAGE_CONTENT"):
                    deltas += 1
    return {"ok": deltas > 1, "deltas": deltas, **got}


async def plan_mode(arm: str) -> dict[str, Any]:
    """``permission_mode="plan"``: Claude plans and changes nothing."""
    cwd = folder("plan")
    opts = options(cwd, permission_mode="plan")
    got = await go(
        opts,
        arm,
        ["Create a file plan.txt with the word go in it."],
        "cl-plan",
        decide=lambda _i: ("reject", "stay in plan mode"),
    )
    return {"ok": not (cwd / "plan.txt").exists(), **got}


async def stdio_mcp(arm: str) -> dict[str, Any]:
    server = {
        "type": "stdio",
        "command": sys.executable,
        "args": [str(HERE / "mcp_server.py")],
        "env": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    }
    opts = options(
        folder("mcp"),
        mcp_servers={"warehouse": server},
        allowed_tools=["mcp__warehouse__warehouse_stock"],
    )
    got = await go(opts, arm, ["How many units of SKU-7 are in the central warehouse?"], "cl-mcp")
    return {"ok": "12" in str(got["answers"][-1]), **got}


async def session_resume(arm: str) -> dict[str, Any]:
    """query way: the second prompt resumes the first's session (``resume=``); wrapped, the
    conversation's next run continues it."""
    opts = options(folder("resume"))
    qs = ["My favourite colour is teal. Just say noted.", "What is my favourite colour?"]
    got = await go(opts, arm, qs, "cl-resume")
    return {"ok": "teal" in str(got["answers"][-1]).lower(), **got}


async def live_session(arm: str) -> dict[str, Any]:
    """live way: one ``ClaudeSDKClient`` for both prompts; wrapped, a conversation's turns
    served by the kept process."""
    opts = options(folder("live"))
    qs = ["Remember the code word: PELICAN. Just say ok.", "What was the code word?"]
    got = await go(opts, arm, qs, "cl-live", live=True)
    return {"ok": "pelican" in str(got["answers"][-1]).lower(), **got}


async def background_bash(arm: str) -> dict[str, Any]:
    """live way: a ``Bash`` command run in the background in turn 1, its output read in a
    later turn while the session goes on."""
    cwd = folder("bg")
    opts = options(cwd, allowed_tools=["Bash", "Read", "BashOutput"])
    qs = [
        "Run this shell command in the background (run_in_background true) and do not wait "
        "for it: `sleep 4 && echo BG-DONE-77 > out.txt`. Then reply 'started'.",
        "What is 2+2? Reply with just the number.",
        "Wait for the background command to finish, then read out.txt and tell me what it says.",
    ]
    got = await go(opts, arm, qs, "cl-bg", live=True)
    return {"ok": "BG-DONE-77" in str(got["answers"][-1]), **got}


FEATURES: dict[str, Any] = {
    "sdk_tool": {"bare", "off", "stack"},
    "builtin_read": {"bare", "off", "stack"},
    "builtin_write": {"bare", "off", "stack"},
    "structured": {"bare", "off"},
    "own_hooks": {"bare", "off"},
    "can_use_tool": {"bare", "off"},
    "subagent": {"bare", "off", "stack"},
    "max_turns": {"bare", "off"},
    "partial_stream": {"bare", "off"},
    "plan_mode": {"bare", "off"},
    "stdio_mcp": {"bare", "off"},
    "session_resume": {"bare", "off", "stack"},
    "live_session": {"bare", "off", "stack"},
    "background_bash": {"bare", "off", "stack"},
}


async def main() -> None:
    for feature, arm in chosen(sys.argv[1:], FEATURES):
        await case(SDK, feature, arm, lambda f=feature, a=arm: globals()[f](a))


if __name__ == "__main__":
    asyncio.run(main())
