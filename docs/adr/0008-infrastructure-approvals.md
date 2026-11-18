# 0008. Infrastructure is the platform operator's: a call of it that needs approval is denied

**Status:** Accepted (2026-11-18)

## Context

The owner's principle: *like Claude for the end user, enterprise governance for everything
else.* The run's user is asked only about actions in their own name on their business data —
an order, a refund, an email, a price, their schedules, deleting their alerts, approving a
plan. Infrastructure is not theirs, and not their company's either: many client companies
(tenants) share the platform, and its shells and its network out belong to the **platform
operator**. The infrastructure tools are the shells (Claude Code's and Kimi Code's `Bash`, an
OpenAI `SandboxAgent`'s `exec_command` and `write_stdin`, Deep Agents' `execute`, the harness's
`sandbox_exec` and `sandbox_write`) and the network tools (Claude's `WebFetch` and `WebSearch`,
Kimi's `FetchURL` and `WebSearch`).

Today (`5e2429a`):

* a shell runs unasked and announced (`Decision.unasked`, `Tool.confined`); a network tool is
  refused until the tool catalog gives it a risk or an `approve_when` (`bridge.outbound`);
* when the catalog makes one of them ask — a declared risk `write` or `irreversible`, or an
  `approve_when` that holds — the question goes to **the run's user** (`bridge._decided` →
  `Runtime.approve` with no assignee: anyone's, answered on the chat surface); and a
  `before_tool` hook's `Ask(assignee="role:…")` routes it to a tenant's role. Either way
  someone who does not own the platform approves its infrastructure.

## Decision

1. **One definition of an infrastructure tool** — `trellis.harness.governance.infrastructure`:
   a tool is infrastructure when **the harness or the framework made it** and **its name is a
   shell's or a network tool's**:

   * `SHELLS`: Claude Code's `Bash`, `PowerShell`, `REPL`, `Monitor` (`host.CLAUDE_SHELLS`),
     Kimi Code's `Bash` (`governance.shell.SHELLS`), an OpenAI `SandboxAgent`'s `exec_command`,
     `write_stdin`, Deep Agents' `execute`, LangChain's `ShellToolMiddleware` `shell`, the
     harness's `sandbox_exec`, `sandbox_write`;
   * `NETWORK`: Claude's `WebFetch`, `WebSearch`, Kimi's `FetchURL`, `WebSearch`.

   "Made by the harness or the framework" is what already exists: `Tool.confined` for a call
   the harness runs (the sandbox's tools; a LangChain / Deep Agents tool whose code is the
   framework's, `middleware.confined`; the SDK's `Shell` capability, `shell_of_sandbox`), and a
   framework's built-in decided through its own permission flow (`bridge.permitted`: no harness
   `Tool`). A team's own function that happens to be called `execute` or `shell` is not
   infrastructure: it says what it does, as ever. An OpenAI `LocalShellTool` / local
   `ShellTool` runs *your* executor (not confined): governed as your tool — unchanged.

2. **The rule, at the one decision point.** `bridge._decided` (every call the harness runs,
   and every framework built-in through `bridge.permitted`) and `bridge.asks_first` (a call
   held at OpenAI Agents' own gate). A call of an **infrastructure** tool that would **ask** a
   person — for whatever reason: the catalog's risk or `approve_when`, a rewind past a write, a
   `before_tool` hook's `Ask` (with an assignee or without) — is **denied**: not run, its
   outcome `REJECTED` (`error_class="InfrastructureDenied"`), the model reads `"<tool> was not
   run: it is the platform's infrastructure, and a call of it that needs approval is denied; no
   user can approve it."`, and the run records it (the call's result on the run's stream, its
   span and memory tool record, the `decision` event `REFUSE`).

3. **No new setting, no new default.** Shells still run unasked and announced; network tools
   are still refused without the catalog's word; `TRELLIS_HOST_EXECUTION` and the sandbox are
   as they were. Only the question is gone.

4. **The `before_tool` hook is unchanged** — a general feature for business approvals
   (`Ask(assignee="role:finance")` on a refund). It is no way to approve infrastructure: on an
   infrastructure tool its `Ask` is denied like any other question.

5. **Schedules.** A schedule's confirmation ("allowing X in its runs") never lists an
   infrastructure tool. An unattended scheduled run follows the same rule: denied.

6. **Plan mode** is unchanged: approving a plan is the user's (business). Until it is
   approved a call that does more than read is refused as every such call is; after, the rule.

7. **AG-UI and A2A.** Nothing new reaches a chat surface: no infrastructure question is ever
   shown, to the user or a role.

8. **Back-compat.** A deployment whose catalog made an infrastructure tool ask, relying on the
   run's user (or a hook's role) answering, now sees those calls denied; to let them run, the
   operator declares them `read` in the catalog (or drops the `approve_when`). A question
   asked before the upgrade and answered after keeps its answer (the journal's). Unaffected:
   business tools; infrastructure tools nothing makes ask; provider-hosted tools (held, never
   asked); Way 2's `governed()` (the team's `on_ask`).

## Consequences

* One place decides, by one definition of infrastructure; adapters gain no per-adapter code.
* The adapter matrix gains a test per adapter path (`tests/matrix/test_infra_approval.py`).

## Engineering review

`/plan-eng-review` (gstack) on this record, run unattended (a subagent: each choice took the
review's recommendation). It first reviewed a version that routed the question to an
administrator's role behind a new setting; the owner then simplified the design twice (no
setting; then denial, infrastructure being the platform operator's), and the findings below
carried over. Scope accepted as-is (4 source files, 1 new module, no new class).

1. **Deny before anything allowed the tool beyond one call** (architecture, 9/10,
   `tools/bridge.py` `_decided`: `_allowed(...)` runs before the approval is asked). A user's
   "for this conversation" or "always" for `Bash` kept before this change, a schedule's
   allowance, or a framework gate's approval (`journal.approved`) must not let an
   infrastructure call through. The rule runs right after the hooks, governance and the rewind
   check, before `_allowed` and the gate.
2. **OpenAI Agents' own gate** (architecture, 9/10, `bridge.asks_first`: `await
   runtime.approve(...)`). It would pause and ask. It now reports "does not ask" for an
   infrastructure call that would, so the SDK runs the call into the bridge, which denies it
   there: one place denies.
3. **A rewind past an infrastructure write** (edge case, 8/10, `_rewound_past` asks a person
   whatever the risk): denied too, by the same rule.
4. **Recording a denied framework built-in** (code quality, 7/10). A call the harness runs is
   recorded by the bridge's refused path (`_events`); a framework's built-in denied in
   `bridge.permitted` gets the same `TOOL_CALL_*` events, span and memory record through
   `bridge.observed`, with the `decision` event `REFUSE`.
5. **One list, derived where it can be** (red flag "hand-synced list"): the module derives from
   `host.CLAUDE_SHELLS` and `governance.shell.SHELLS` (light modules); the names held in heavy
   modules (`CAPABILITY_SHELLS`, `middleware.EXECUTE`, the sandbox's `EXEC`/`WRITE`,
   `CLAUDE_NETWORK`, `KIMI_NETWORK`) are literal here and a unit test fails when any of them is
   not inside the definition.
6. **Schedules' confirmation** excludes infrastructure tools (`schedules._asking`).

Second shape considered: a `Tool.infrastructure` flag set wherever `confined` is set, plus a
flag each adapter passes to `bridge.permitted`. Rejected: it spreads the decision over every
place a tool is made and every adapter's permission flow, where the chosen shape keeps one list
and one check behind `infrastructure(name, own=...)`. Performance: one frozenset lookup per
call that would ask. Not in scope: OpenAI `LocalShellTool`/local `ShellTool` (your executor),
provider-hosted tools, Way 2's `governed()`.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Outside Review | — | Independent 2nd opinion | 0 | skipped (unattended subagent) | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | ISSUES OPEN → resolved in design | 6 issues, 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **OUTSIDE COVERAGE:** Outside Voice skipped: unattended run, no external reviewer.
- **VERDICT:** ENG CLEARED — ready to implement (all six findings folded into the decision).

NO UNRESOLVED DECISIONS
