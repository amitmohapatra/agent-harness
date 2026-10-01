# Overhaul progress (source of truth for resuming — only run unchecked items)

## Done (committed, tested)
- [x] agent-contracts (63fc723) — 174 tests
- [x] bifrost-sdk (3bd7c31) — 104 tests
- [x] agent-runs (500ec10) — 194 tests incl. fake GCS
- [x] memory passes 1–3 (d5c41d1)
- [x] harness final pass commits (fe7b1b2) — full test run NOT yet done (see Harness)
- [x] trellis-acceptance scaffold (92a5d07)

## Memory final pass (agent-memory-service, branch overhaul; WIP uncommitted in working tree)
Already done in working tree (verify compiles, then commit): keys/self, context format=prompt,
[mN] handles, window flag, observations route removed, outcome route removed, groups removed,
5 agent tools, approve_when + suggestion accept, workspace model-key removed, OpenFGA timeout,
GCS fake test written, qdrant partially two-phase, openapi regenerated.
- [x] M1 Remove briefs module (+ routes, SDK, tests, docs, migration) → profile block with source_query — code already gone in WIP; docs/descriptions cleaned (3e6e16d WIP, 565bdb7)
- [x] M2 Remove memory webhooks module (+ routes, SDK, tables via migration, jobs, tests, docs); harness signing helper must not break (check agent-harness imports of trellis.memory.webhooks) — code/tables already gone in WIP (0021); docs cleaned; harness uses own signer (e74998c)
- [x] M3 Fix stale "/v1/observations" references (e.g. api/routers/v1/memory.py:100) and any other leftovers of removed routes — docs/examples/tests moved to final surface; removed-capability tests deleted (cba917e)
- [x] M4 Finish + measure two-phase Qdrant read; p95 /v1/context (format=prompt) and /v1/recall recorded in docs/MEASUREMENTS.md; decide memory_entity_search with numbers — two-phase measured slower (paired) and removed; context p95 525 / recall 421 ms (format=prompt, 4-core box); entity search stays on (d956edf)
- [x] M5 Verify every /v1/context section fills correctly on a seeded scenario (profile, summary+window, memories, knowledge, graph facts, procedures, tools next/prefill/missing); fix root causes — all sections fill (tests/agent/test_api_context_push.py 2/2); fixed: summary dropped with window=false (4dbb273)
- [x] M6 Run the fake-GCS integration test (chat archive, documents, tool outputs >4KB, checksum, purge) — 4/4 passed against fake-gcs-server 1.52; teardown fix (ca3adf8)
- [x] M7 Update docs/DATA_PLACEMENT_REVIEW.md to reality — updated with 'now' markers + status table; open: tuple revocation for threads/docs/memories (13440c1)
- [x] M8 Full suites green ONCE (unit, integration, e2e+agent, contract, lint, typecheck); regenerate openapi; commit in logical steps — green: unit 1620, contract 107+34s, integration 250+5s, e2e 40, agent 79; ruff/format/pyright clean; openapi regenerated (25fed78..1ff3362). Open: bundle records (verify) survive a MEMBERSHIP revision bump for 30 min

## Harness (after memory M8)
- [x] H1 Per-agent tool scoping for h.tools (no global name clash) — each h.tools call tagged, wrap() reads graph ToolNodes (cd68fce)
- [x] H2 List MCP tools via gateway /mcp tools/list with the virtual key — bifrost-sdk tools() /mcp-only, Code Mode tools from listToolFiles/readToolFile (bifrost-sdk c005360, harness fd4bb0c). Note: mcp_logs (/api/mcp-logs, Code Mode nested-call import) is still admin-only
- [x] H3 Full suite + live suite green once against final memory — check 188 passed, examples 8 ok, live 26 passed/1 skipped (needs env VK; passes with one) (ced9d85)

## Follow-ups (after merge; pre-existing security gaps)
- [ ] F1 Memory: revoke OpenFGA tuples on thread/document/memory delete + orphan-tuple cleanup job
- [ ] F2 Memory: /v1/verify rechecks membership (no 30-min window after revocation)
- [ ] F6 bifrost-sdk/harness: mcp_logs() uses admin-only /api/mcp-logs → Code Mode nested-call import 401s with admin auth on; find a virtual-key path or document admin token
- [ ] F7 Memory: query by knowledge time ("what did we know at T") using observed_at alongside valid_from/to
- [ ] F8 Memory: beliefs with evidence-driven confidence (Hindsight-style) — only if F5 shows a gap
- [ ] F3 Memory: index thread summaries as searchable episodic records (cross-thread recall)
- [ ] F4 Memory: verify/complete relative-date normalisation on write (temporal.py)
- [ ] F5 Memory: measure long-term recall on LongMemEval; fix enum docstring (BELIEF/ENTITY_SUMMARY no longer derived)

## Merge (owner decision: merge BEFORE acceptance, right after H3)
- [ ] A0 Fast-forward every repo's main to overhaul together (contracts, bifrost-sdk, runs, schedules, memory, harness) + tag platform-2026.10; don't push without asking

## Acceptance (after A0)
- [ ] A1 make acceptance once; fix root causes on main; report build/acceptance-report.md
