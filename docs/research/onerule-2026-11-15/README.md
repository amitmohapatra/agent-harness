# One approval rule — live check with Haiku, 2026-11-15

What was checked: an MCP tool whose server sends **no annotations** (`tidy` of
`tests/support/mcp_risks_server.py`) asks; the person picks **Always allow**; the user's next
conversation runs it with nobody asked. Model `claude-haiku-5-5`, memory the suite's in-process
memory service (`FakeMemoryService`), what the harness published kept in its catalog between the
two conversations as the service keeps it. Script: [`live_check.py`](live_check.py)
(`LLM_API_KEY=… python live_check.py react out.json`, `ANTHROPIC_API_KEY=… python live_check.py
claude out.json`).

| adapter | way the tool came | 1st conversation | choices offered | after "always" | published to the catalog | 2nd conversation | ran | spend |
|---|---|---|---|---|---|---|---|---|
| ReAct (Anthropic's OpenAI-compatible endpoint) | `trellis.mcp(stdio(...))`, `ledger-tidy` | paused: "ledger-tidy has not declared its side effects." | once, run, always, reject | SUCCESS, kept for the user | no annotations (risk null) | SUCCESS, not asked | `tidy 7`, `tidy 8` | 16,395 tokens (≤ $0.0082, every token priced as output) |
| Claude Agent SDK (the bundled CLI) | Claude's own stdio MCP server, `mcp__ledger__tidy` | paused: "mcp__ledger__tidy has not declared its side effects." | once, run, always, reject | SUCCESS, kept for the user | no annotations (risk null) | SUCCESS, not asked | `tidy 7`, `tidy 8` | $0.0019 (the CLI's own cost) |

Total spend: about $0.010 (cap $1). Before the change, the harness published
`destructiveHint: true` for such a tool, so the catalog derived `irreversible` and every call
asked with allow once or deny only; on ReAct's `mcp(...)` it was `irreversible` from the first
call.
