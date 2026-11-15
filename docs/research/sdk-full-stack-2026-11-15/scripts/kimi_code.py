# ruff: noqa: PLC0415 - a research script: it reports with print, imports an SDK where a case needs it
"""Kimi Code 2.6.2: its own features as its docs show them, on its own (``bare``: ``kimi -p``
with ``--output-format stream-json``, a session continued with ``-S``) and wrapped (``off``,
``stack``: ``kimi(...)`` over ACP). Its model is Haiku through Kimi's ``anthropic`` provider,
the meter in front; a Kimi home (``KIMI_CODE_HOME``) and a working folder per case.

Usage: python kimi_code.py [feature[:arm] ...]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from common import METER, MODEL, case, chosen, key
from turns import wrapped

from trellis import kimi

SDK = "kimi"
HERE = Path(__file__).resolve().parent


def place(tag: str) -> tuple[Path, Path]:
    root = Path(tempfile.mkdtemp(prefix=f"ki-{tag}-"))
    (root / "cwd").mkdir()
    (root / "home").mkdir()
    return root / "cwd", root / "home"


def env(home: Path) -> dict[str, str]:
    return {
        "KIMI_MODEL_PROVIDER_TYPE": "anthropic",
        "KIMI_MODEL_BASE_URL": METER,
        "KIMI_MODEL_API_KEY": key(),
        "KIMI_MODEL_NAME": MODEL,
        "KIMI_MODEL_MAX_CONTEXT_SIZE": "200000",
        "KIMI_MODEL_MAX_OUTPUT_SIZE": "16000",
        "KIMI_CODE_HOME": str(home),
        "KIMI_CODE_NO_AUTO_UPDATE": "1",
        "KIMI_DISABLE_TELEMETRY": "1",
    }


async def bare(cwd: Path, home: Path, prompts: list[str], *flags: str) -> dict[str, Any]:
    answers: list[str] = []
    tools_used: list[str] = []
    session: str | None = None
    for p in prompts:
        args = ["kimi", "-p", p, "--output-format", "stream-json", *flags]
        if session:
            args += ["-S", session]
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            env={**os.environ, **env(home)},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), 300)
        said = ""
        for line in out.decode(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("role") == "assistant":
                tools_used += [c["function"]["name"] for c in row.get("tool_calls") or []]
                said = row.get("content") or said
            if row.get("type") == "session.resume_hint":
                session = row.get("session_id")
        answers.append(said)
    return {"answers": answers, "tools_used": tools_used}


async def bare_acp(
    cwd: Path,
    home: Path,
    prompts: list[str],
    *,
    mode: str | None = None,
    servers: list[dict[str, Any]] = (),
    permit: bool = False,
) -> dict[str, Any]:  # type: ignore[assignment]
    """Kimi over ACP with a client of the case's own (no harness): the bare way for what
    prompt mode cannot do (``-p`` refuses ``--plan``; MCP servers given in ``session/new``).
    Each permission request is answered ``permit`` (allow once) or denied."""
    from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
    from acp.schema import (
        AllowedOutcome,
        ClientCapabilities,
        DeniedOutcome,
        McpServerStdio,
        RequestPermissionResponse,
    )

    said: list[str] = []
    asked: list[str] = []

    class Client:
        async def session_update(self, session_id: str, update: Any, **kw: Any) -> None:
            if getattr(update, "session_update", "") == "agent_message_chunk":
                said.append(getattr(update.content, "text", ""))

        async def request_permission(
            self, options: Any, session_id: str, tool_call: Any, **kw: Any
        ) -> Any:
            asked.append(str(getattr(tool_call, "title", "")))
            allow = next((o for o in options if "allow" in str(o.kind)), None)
            if permit and allow is not None:
                return RequestPermissionResponse(
                    outcome=AllowedOutcome(outcome="selected", option_id=allow.option_id)
                )
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    answers = []
    async with spawn_agent_process(
        Client(), "kimi", "acp", env={**os.environ, **env(home)}, cwd=str(cwd)
    ) as (conn, _proc):
        await conn.initialize(
            protocol_version=PROTOCOL_VERSION, client_capabilities=ClientCapabilities()
        )
        mcp = [McpServerStdio.model_validate({"args": [], "env": [], **s}) for s in servers]
        opened = await conn.new_session(cwd=str(cwd), mcp_servers=mcp)
        if mode is not None:
            await conn.set_session_mode(session_id=opened.session_id, mode_id=mode)
        for p in prompts:
            said.clear()
            await asyncio.wait_for(
                conn.prompt(session_id=opened.session_id, prompt=[text_block(p)]), 300
            )
            answers.append("".join(said))
    return {"answers": answers, "permission_asked": asked}


async def go(
    arm: str,
    tag: str,
    prompts: list[str],
    *,
    flags: tuple[str, ...] = (),
    setup: Any = None,
    **target: Any,
) -> tuple[dict[str, Any], Path]:
    cwd, home = place(tag)
    if setup is not None:
        setup(cwd, home)
    if arm == "bare":
        return await bare(cwd, home, prompts, *flags), cwd
    decide = target.pop("decide", None)
    kw = {"decide": decide} if decide else {}
    agent = kimi(model=MODEL, cwd=str(cwd), env=env(home), **target)
    return await wrapped(agent, arm, f"ki-{tag}", prompts, **kw), cwd


# --------------------------------------------------------------------------- features
async def read(arm: str) -> dict[str, Any]:
    got, _ = await go(
        arm,
        "read",
        ["Read tide.txt and tell me when high water is."],
        setup=lambda c, h: (c / "tide.txt").write_text("High water at 06:42.\n"),
    )
    return {"ok": "06:42" in str(got["answers"][-1]), **got}


async def bash_write(arm: str) -> dict[str, Any]:
    """A ``Bash`` command that writes: Kimi asks (its default mode); approved, it runs."""
    # bare: prompt mode (``-p``) runs the command itself, nobody to ask
    got, cwd = await go(arm, "bash", ["Run the shell command `echo built > status.txt`."])
    made = (cwd / "status.txt").read_text().strip() if (cwd / "status.txt").exists() else None
    return {"ok": made == "built", "file": made, **got}


async def agents_md(arm: str) -> dict[str, Any]:
    def setup(c: Path, h: Path) -> None:
        (c / "AGENTS.md").write_text("Always end every answer with the line: -- KIMI-OK\n")

    got, _ = await go(arm, "agentsmd", ["What is 3+4?"], setup=setup)
    return {"ok": "KIMI-OK" in str(got["answers"][-1]), **got}


async def mcp(arm: str) -> dict[str, Any]:
    server = {
        "name": "warehouse",
        "command": sys.executable,
        "args": [str(HERE / "mcp_server.py")],
        "env": [{"name": "PYTHONPATH", "value": os.environ.get("PYTHONPATH", "")}],
    }

    def setup(c: Path, h: Path) -> None:  # Kimi's own MCP config, for the bare run
        conf = {
            "mcpServers": {
                "warehouse": {
                    "command": sys.executable,
                    "args": [str(HERE / "mcp_server.py")],
                    "env": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")},
                }
            }
        }
        (h / "mcp.json").write_text(json.dumps(conf))

    q = ["How many units of SKU-7 are in the central warehouse? Use the warehouse tool."]
    if arm == "bare":  # the same server, given in ACP's session/new as the harness gives it
        cwd, home = place("mcp")
        got = await bare_acp(cwd, home, q, servers=[server], permit=True)
    else:
        got, _ = await go(arm, "mcp", q, mcp_servers=[server])
    return {"ok": "12" in str(got["answers"][-1]), **got}


async def subagent(arm: str) -> dict[str, Any]:
    def setup(c: Path, h: Path) -> None:
        (c / "facts.txt").write_text("The project codename is HERON.\n")

    q = [
        "Use the Agent tool to start an explore sub-agent that reads facts.txt and reports "
        "the project codename. Then tell me the codename."
    ]
    got, _ = await go(arm, "sub", q, setup=setup)
    used = got.get("tools_used") or []
    return {
        "ok": "HERON" in str(got["answers"][-1]),
        "delegated": ("Agent" in used) if used else None,
        **got,
    }


async def session(arm: str) -> dict[str, Any]:
    qs = ["Remember the code word PELICAN. Just say ok.", "What was the code word?"]
    got, _ = await go(arm, "session", qs)
    return {"ok": "pelican" in str(got["answers"][-1]).lower(), **got}


async def skills(arm: str) -> dict[str, Any]:
    def setup(c: Path, h: Path) -> None:
        skill = h / "skills" / "haiku-poet"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: haiku-poet\ndescription: Write a haiku about any topic the user "
            "names.\n---\nWrite exactly three lines, then a fourth line: -- SKILL:HAIKU\n"
        )

    got, _ = await go(
        arm, "skills", ["Use your haiku-poet skill: a haiku about autumn rain."], setup=setup
    )
    return {"ok": "SKILL:HAIKU" in str(got["answers"][-1]), **got}


async def todo(arm: str) -> dict[str, Any]:
    q = [
        "Use your TodoList tool to plan three steps (name a fruit, a vegetable, a grain), "
        "mark them done as you go, then give the three names."
    ]
    got, _ = await go(arm, "todo", q)
    used = got.get("tools_used") or []
    planned = ("TodoList" in used) if arm == "bare" else None
    return {"ok": bool(got["answers"][-1]) and planned is not False, "planned": planned, **got}


async def plan_mode(arm: str) -> dict[str, Any]:
    """Plan mode (``--plan``, ``mode="plan"``): Kimi plans and writes nothing until approved;
    the plan is rejected here."""
    q = ["Create a file plan.txt containing the word go."]
    if arm == "bare":  # prompt mode refuses --plan: Kimi over ACP, plan mode, plan denied
        cwd, home = place("plan")
        got = await bare_acp(cwd, home, q, mode="plan")
    else:
        got, cwd = await go(arm, "plan", q, mode="plan", decide=lambda _i: ("reject", "not now"))
    return {"ok": not (cwd / "plan.txt").exists(), **got}


async def background(arm: str) -> dict[str, Any]:
    qs = [
        "Run this shell command in the background (run_in_background true), do not wait: "
        "`sleep 4 && echo BG-DONE-77 > out.txt`. Then reply 'started'.",
        "What is 2+2? Reply with just the number.",
        "Wait until the background command has finished, then read out.txt and say what "
        "it contains.",
    ]
    got, _ = await go(arm, "bg", qs)
    return {"ok": "BG-DONE-77" in str(got["answers"][-1]), **got}


FEATURES: dict[str, Any] = {
    "read": {"bare", "off", "stack"},
    "bash_write": {"bare", "off", "stack"},
    "agents_md": {"bare", "off"},
    "mcp": {"bare", "off", "stack"},
    "subagent": {"bare", "off", "stack"},
    "session": {"bare", "off", "stack"},
    "skills": {"bare", "off"},
    "todo": {"bare", "off"},
    "plan_mode": {"bare", "off"},
    "background": {"bare", "off", "stack"},
}


async def main() -> None:
    for feature, arm in chosen(sys.argv[1:], FEATURES):
        await case(SDK, feature, arm, lambda f=feature, a=arm: globals()[f](a))


if __name__ == "__main__":
    asyncio.run(main())
