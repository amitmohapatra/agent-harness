# Architecture decision records

The decisions behind the harness's design, one file each: the context, the decision, and what
follows from it. A record is never rewritten once accepted; a later one supersedes or amends it
and says so in both.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-version-policy.md) | The harness pins the framework versions it tested and moves only by a release | Accepted |
| [0002](0002-tool-choice.md) | Tool choice is guarded by opt-in middleware, a warning-only check of the tool set, and per-item coverage | Accepted; amended 2026-10-28 (the middleware removed) |
| [0003](0003-automatic-features.md) | ReAct turns native features on from signals, never from new settings | Accepted |
| [0004](0004-provider-neutral-models.md) | Model access is provider-neutral: three strategies (route, auth, profile); configure the harness only, memory follows | Accepted |
| [0005](0005-conversations-across-runs.md) | A conversation across runs: history where nothing keeps it, busy-conversation modes, one id per message, the framework's session continued | Accepted; amends 0003 point 6 |
| [0006](0006-what-the-model-gets.md) | The intelligent harness: what the model gets (a context map, data cards, built-in tools by run state), so a small model gives a big model's answers | Accepted as the direction; parts proposed |

The sibling repositories keep their own: the memory service's
[docs/adr](https://github.com/amitmohapatra/agent-memory-service/blob/main/docs/adr/README.md)
(among them 0020, framework adapters live in the consumer — here) and the contracts'
[docs/adr](https://github.com/amitmohapatra/agent-contracts/tree/main/docs/adr).
