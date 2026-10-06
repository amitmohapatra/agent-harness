"""Scenario: SQL that changes data waits for a DBA, on the DBA team's own screen.

* the analyst agent (``ReAct``) has one tool, ``run_sql``; a ``before_tool`` hook lets a
  ``SELECT`` run and asks ``role:dba`` about anything else, on the ``sql-review`` component
  with the statement and the tables it touches as its props;
* your screen is two calls: ``h.inbox("role:dba")`` lists what waits (with the component and
  props to render), and ``agent.resume`` answers — here an ``edit``: the DBA adds a ``WHERE``
  the statement lacked, the tool runs with the edited SQL, the model reads the result;
* the next change is approved for the rest of the run (``remember="run"``): a third one runs
  without asking.

    python -m examples.06_scenarios.sql_approval_own_screen
"""

from __future__ import annotations

import asyncio
import re

from examples._support.offline import react_model

from trellis import Ask, Harness, Hooks, ReAct, tool
from trellis.contracts import ToolCall

TABLES = re.compile(r"\b(?:from|update|into|join)\s+(\w+)", re.IGNORECASE)
executed: list[str] = []


@tool(side_effects="write")
def run_sql(sql: str) -> str:
    """Run one SQL statement against the warehouse."""
    executed.append(sql)
    return "3 rows" if sql.lower().startswith("select") else "1 row changed"


class SqlReview(Hooks):
    async def before_tool(self, call: ToolCall) -> Ask | None:
        sql = str(call.args.get("sql", ""))
        if call.tool != "run_sql" or sql.lower().startswith("select"):
            return None
        return Ask(
            "Run this statement?",
            assignee="role:dba",
            component="sql-review",
            props={"sql": sql, "tables": TABLES.findall(sql)},
        )


async def dba_screen(h: Harness) -> None:
    """Your own review screen: what waits on the DBAs, rendered with your component."""
    for waiting in await h.inbox("role:dba"):
        asked = waiting.awaiting
        assert asked is not None
        print(f"[{asked.component}] {asked.props}")


async def main() -> None:
    model = react_model(
        [
            ("run_sql", {"sql": "SELECT id FROM orders WHERE status = 'stuck'"}),
            ("run_sql", {"sql": "UPDATE orders SET status = 'retry'"}),
            ("run_sql", {"sql": "UPDATE jobs SET state = 'queued' WHERE order_id = 7"}),
            ("run_sql", {"sql": "UPDATE jobs SET tries = 0 WHERE order_id = 7"}),
            "Order 7 is set to retry and its job re-queued.",
        ]
    )
    async with Harness() as h:
        agent = h.wrap(
            ReAct(system="You fix stuck orders with SQL.", model=model),
            id="analyst",
            tools=[run_sql],
            hooks=[SqlReview()],
        )
        result = await agent.run("Retry the stuck order 7.", user="ada")
        await dba_screen(h)
        assert result.interrupt is not None
        fixed = {"sql": "UPDATE orders SET status = 'retry' WHERE id = 7"}
        result = await agent.resume(
            result.interrupt.interrupt_id,
            "edit",
            answer=fixed,
            reviewer="dba-lee",
            comment="scoped to order 7",
        )
        await dba_screen(h)
        assert result.interrupt is not None
        result = await agent.resume(
            result.interrupt.interrupt_id,
            "approve",
            reviewer="dba-lee",
            remember="run",  # the DBA's own later statements in this run are not asked again
        )
        print(result.status.value, result.answer)
        print("executed:", executed)


if __name__ == "__main__":
    asyncio.run(main())
