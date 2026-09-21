"""Visibility prerequisites (the Memory Service's ``visibility_keys`` rules).

Each visibility level is backed by an audience key built from the execution context, so a
level whose id is missing cannot be expressed:

    PRIVATE / TENANT / GLOBAL  always available
    USER                       requires user_id
    GROUP                      requires group_ids
    AGENT_GROUP                requires agent_group_id
    RUN                        requires agent_run_id
    THREAD                     requires thread_id
    WORK                       requires work_id
    WORKSPACE                  requires workspace_id

This matters more than it looks. The API *accepts* an observation whose visibility it cannot
satisfy and fails later, in the background job that would have created the memory — so the
caller sees a 202 and the memory never appears. Checking here turns silent data loss into an
immediate, actionable error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_contracts.context import AgentExecutionContext

#: visibility -> the context field that must be present for it.
REQUIRED_FIELD: dict[str, str] = {
    "USER": "user_id",
    "GROUP": "group_ids",
    "AGENT_GROUP": "agent_group_id",
    "RUN": "agent_run_id",
    "THREAD": "thread_id",
    "WORK": "work_id",
    "WORKSPACE": "workspace_id",
}

#: Levels that need nothing from the context.
UNCONDITIONAL = frozenset({"PRIVATE", "TENANT", "GLOBAL"})

VISIBILITIES = frozenset(REQUIRED_FIELD) | UNCONDITIONAL


def missing_requirement(visibility: str, context: AgentExecutionContext) -> str | None:
    """The context field this visibility needs and does not have, if any."""
    field = REQUIRED_FIELD.get(visibility)
    if field is None:
        return None
    return field if not getattr(context, field, None) else None


def check(visibility: str | None, context: AgentExecutionContext) -> None:
    """Raise if ``visibility`` cannot be satisfied by this execution context."""
    if visibility is None:
        return
    from universal_agent_contracts.errors import ConfigurationError  # noqa: PLC0415

    if visibility not in VISIBILITIES:
        raise ConfigurationError(
            f"unknown visibility {visibility!r}; the Memory Service accepts "
            f"{', '.join(sorted(VISIBILITIES))}",
            source="memory.visibility",
        )
    field = missing_requirement(visibility, context)
    if field:
        raise ConfigurationError(
            f"visibility {visibility!r} requires {field} on the execution context, which is "
            f"not set. The service would accept the write and then fail the job that creates "
            f"the memory, so it is refused here instead.",
            details={"visibility": visibility, "missing": field},
            source="memory.visibility",
        )
