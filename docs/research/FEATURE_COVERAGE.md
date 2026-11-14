# Feature coverage: every documented feature, and the test that proves its purpose

What a feature is *for* — the outcome a user relies on — and the tests that prove it, at each
level: **unit** (in process, one module), **integration** (in process, the harness with fakes:
`LocalRuns`, a fake memory service, scripted models), **matrix** (`tests/matrix`, every adapter
× way × mode, in process; its rows are the `F..` ids of [docs/matrix.md](../matrix.md)),
**live** (`tests/live`, the real memory service and agent-runs, models scripted; the ones that
passed on this commit in the live57 run of d2bf754), **service suite** (agent-runs' and the
memory service's own tests: a real PostgreSQL, the app in process), and **e2e** — this
branch's `tests/features`: the real memory service, two real agent-runs API replicas and the
ticker, each in a process of its own, every model scripted, the services' databases read back.

IDs are the master test plan's (`/home/user/TEST_PLAN.md`: MEM, RUN, SCH, FS, SURF, CON, plus
HAR-* for the harness inventory and the matrix's `F..`). Every e2e test carries its IDs in a
`@pytest.mark.plan(...)` marker; `pytest -m features --collect-only -q tests/features` lists
them. Status: **pass**, **xfail(strict)** (a defect found, its test kept failing until fixed:
see [Findings](#findings)), **skip** (with the reason), **prior** (proven before this branch at
the level named, not repeated here).

Stack under test: agent-harness `claude/cool-maxwell-w5avo8-final` d2bf754 (+ this branch's
tests), agent-memory-service 0f20438, agent-runs 422ef68, agent-contracts 8ca570c.

## Summary

| Area | Features | Purpose already proven end to end (before) | Added end to end (this branch) | Defects found |
|---|---|---|---|---|
| Memory (MEM) | 24 | 9 (live) | 23 (about 90 test cases with the per-adapter ones) | 5 in the memory service (BUG-M1..M5; M2 and M5 fixed), F-504 (contract) |
| Runs (RUN) | 13 | 6 (live) + every route in the service suite (in process) | 13 (17 tests) | — |
| Schedules (SCH) | 9 | 3 (live) | 8 + DST skipped (9 tests) | — |
| Files, query, sandbox (FS) | 6 | 1 (live) | 3 (+2 proven in process, the sandbox skipped: no Docker) | H-CSV (fixed) |
| Surfaces (SURF) | 6 | 4 (live) | 6 (6 tests) | H-DOC, H-A2A (fixed), H-AGUI |
| Contracts (CON) | 3 | 2 (contract suite, in process) | 2 (+ INV-7 on every test) | — |
| Harness inventory (HAR) | 29 | 15 (live; all 29 in process: matrix, integration) | 22 (on every applicable adapter: about 75 test cases) | F-DATE (agent-runs SDK), H-ID |
| **Total** | **90** | **40** | **77** | **12** (5 fixed) |

(The counts are features, not tests; a feature proven before and again here counts in both.)

## Memory (MEM)

| ID | Feature: the purpose | unit / integration (harness) | matrix | live | memory service suite | e2e (this branch) | Status |
|---|---|---|---|---|---|---|---|
| MEM-01 | A turn is recorded once: user message, tool calls, answer, even across a pause and a resumed attempt | `test_memory_pipeline.py`, `test_conversation.py` | F43 (all) | `test_memory_is_pushed_pulled_and_written_back[*]` | `e2e/test_idempotent_writes.py`, `integration/test_idempotency.py` | `test_mem_records.py::test_a_paused_and_approved_run_is_recorded_once[10 adapters]`, `::test_a_turn_retried_by_a_new_process_is_stored_once` | pass |
| MEM-01b | A large tool output is recorded bounded | `test_large_data.py` | F26 | — | `integration/test_tool_outputs.py` | `test_mem_records.py::test_a_large_tool_output_is_recorded_bounded` | pass |
| MEM-02 | A fact told in chat 1 is known in chat 2 of the same user (pushed, and found by `memory_search`) | `test_memory_use.py` | F41, F42 (all) | `test_a_run_finds_what_its_user_said_in_an_earlier_conversation` (react) | `integration/test_earlier_chats.py`, `test_statements_follow_the_user.py` | `test_mem_recall.py::test_a_fact_told_in_one_chat_is_known_in_the_next_and_to_no_one_else[10 adapters]` | pass |
| MEM-03 | Never recalled for another user or tenant, checked in the store | `test_conversation_scope.py` | X01 | `test_a_restricted_key_answers_only_its_persons_runs` | `security/test_isolation.py`, `test_retrieval_isolation.py` | `test_mem_recall.py` (both tests: recall, inventory, context, other tenant; visibility keys and thread owners read from the DB) | pass |
| MEM-03b | A user's inventory (GET /v1/memories) lists what they said | — | — | — | `e2e/test_list_and_cache_surface.py` (user-scoped writes only) | `test_mem_lifecycle.py::test_the_users_inventory_lists_what_they_said_in_a_chat` | **xfail: BUG-M3** |
| MEM-04 | Extraction: a sentence becomes a fact (the verbatim turn beside it); the same fact said again is one fact; a correction replaces it | — | — | — | `integration/test_dedup_memories.py`, `test_replacement_across_conversations.py` | `test_mem_lifecycle.py::test_a_fact_said_in_three_chats_is_one_fact_and_a_correction_replaces_it` | pass |
| MEM-04b | Consolidation into beliefs (a model, scripted): one cited line per facet, a repeat kept not duplicated, the person's own, in their next context | — | — | — | `integration/test_beliefs*.py` (stand-in model in process) | `test_mem_beliefs.py::test_facts_are_consolidated_into_cited_beliefs_once` (the service's real job, a scripted OpenAI-compatible endpoint registered as the tenant's model key, policy `reflection` only) | pass |
| MEM-05 | Forgetting what the user said asks a person, then it is gone from recall and the store | `test_memory_forget.py` (fake service) | F42d (fake service) | — | `integration/test_forgetting.py`, `test_concurrent_forget.py`, `test_agent_relayed_user_facts.py` (+2 on the memory branch) | `test_mem_lifecycle.py::test_forgetting_the_users_words_asks_then_they_are_gone[9]`, `::test_the_user_may_forget_what_an_agent_relayed_for_them` (pass with the BUG-M2 fix, agent-memory-service `claude/cool-maxwell-w5avo8-features`); `::test_forgetting_a_fact_read_from_the_users_words_asks_first` | pass (with the M2 fix); **xfail: BUG-M1** |
| MEM-05b | Erasing a user (admin): nothing any read returns | — | — | — | — | `test_mem_lifecycle.py::test_erasing_a_user_leaves_nothing_any_read_returns` | **xfail: BUG-M4** |
| MEM-06 | Retention: memories past the tenant's `retention_days` are forgotten, today's kept, another tenant's untouched | — | — | — | `integration/test_forgetting.py` (retention sweep, in process) | `test_mem_jobs.py::test_retention_forgets_what_is_older_than_the_tenants_policy` (the daily sweep run on demand, a row's age moved back) | pass |
| MEM-07 | Profile blocks an agent edits are in every next chat of the user, no one else's | — | — | `test_a_profile_block_an_agent_edits_is_in_the_next_context` (react) | `integration/test_profile_and_summary.py`, `test_about_user.py` | `test_mem_lifecycle.py::test_a_profile_the_agent_edits_is_in_every_next_chat[9]` | pass |
| MEM-07b | The memory budget given to the model: 8,000 tokens, a quarter of a known window under 32,000, never under 2,000 | `test_memory_window.py` | — | — | `e2e/test_window_flow.py` | `test_mem_lifecycle.py::test_the_memory_budget_follows_the_models_window[3]` | pass |
| MEM-07c | The memories a run cites are capped (50) on its outcome | unit | — | — | — | `test_mem_lifecycle.py::test_the_memories_a_run_cites_are_capped` | pass |
| MEM-08 | Feedback: one TOOL_CALL decision per call on the right run (nested args structured), the system outcome once | `test_hitl_feedback.py` | F40 | `test_feedback_waits_for_review_in_memory_and_is_scored` | `integration/test_feedback_projection.py`, `test_feedback_votes.py` | `test_mem_records.py` (all adapters), `test_mem_resilience.py::test_a_write_committed_but_answered_504_is_stored_once[fb]` | pass |
| MEM-09 | "Always allow" kept in memory for the user in every conversation, no one else, never offered for irreversible | `test_approvals.py` | F33 | `test_a_learned_approval_rule_is_applied_automatically` | `agent/test_api_tool_permissions.py` | `test_mem_governance.py::test_always_allow_is_kept_for_the_user_in_every_conversation[10]`, `::test_always_is_never_offered_for_an_irreversible_tool` | pass |
| MEM-09b | Learned rules start from asking; accepted, the learned shape runs unasked, others still ask | — | — | `test_procedures_tool_stats_approvals_and_profile_are_learned` | `integration/test_tool_rule_guard.py` | `test_mem_governance.py::test_a_learned_rule_starts_from_asking_and_is_applied_once_accepted` | pass |
| MEM-09c / HAR-GOV | Catalog rules: an admin's `risk` overrides the declaration, `approve_when` asks exactly when it holds, a change reaches a running agent within 30 s | `test_governance.py` | F32, F33 | `test_an_approval_rule_in_the_catalog_decides_which_calls_wait`, `test_a_catalog_rule_set_after_the_graph_was_built_decides_its_calls` | `agent/test_api_tools.py` | `test_mem_governance.py::test_the_catalogs_risk_and_rule_decide_and_reach_a_running_agent` | pass |
| MEM-10 | Memory down mid-chat: the turn answers (said), the earlier turn from the run store, writes spooled and delivered once when it is back | `test_memory_outage.py` | — | — | `failure/` | `test_mem_resilience.py::test_memory_down_mid_chat_the_turn_answers_and_its_writes_arrive_once_it_is_back` | pass |
| MEM-10b | A `504` (the service's deadline): a context retried, one that stays 504 a warning; a write committed then answered 504 stored once | — | — | — | — | `test_mem_resilience.py` (3 + 3 tests) | pass |
| MEM-10c | `/v1/context` latency against its 5 s deadline at 1, 8, 16 concurrent | — | — | — | `benchmark/` | `test_mem_deadline.py` | see Findings F-504 |
| MEM-11 | Every route refuses a missing or unknown key; platform routes refuse tenant keys; admin routes refuse the service key | — | — | `test_the_key_names_the_tenant_and_its_role` | `security/test_tenant_binding.py`, `agent/test_api_tenancy.py` | `test_mem_resilience.py::test_every_route_but_the_probes_refuses_a_missing_or_unknown_key`, `::test_the_platform_and_admin_routes_refuse_lesser_keys` | pass |
| MEM-13 | Two chats of one user writing at once lose nothing, duplicate nothing | — | X03 | — | `integration/test_concurrent_forget.py` | `test_mem_lifecycle.py::test_two_chats_of_one_user_writing_at_once_lose_nothing` | pass |
| MEM-14 | Documents: added for a user, cited by their next context | — | — | `test_a_document_added_for_a_user_is_in_their_next_context`, `test_a_runs_context_draws_on_the_documents_it_names` | `e2e/test_files_flow.py` | `test_con.py` (added and named by a queued run) | prior + pass |
| MEM-15 | Archive read path: messages archived into a verified segment; once purged in the hot store, the thread reads back whole from the segment | — | — | — | `integration/test_archive.py` | `test_mem_jobs.py::test_a_purged_message_is_read_back_whole_from_its_archive_segment` (the hourly purge run on demand past its 24 h grace) | pass |

## Runs (RUN)

| ID | Feature: the purpose | harness unit / integration | live | agent-runs suite (real PG, in process) | e2e (this branch) | Status |
|---|---|---|---|---|---|---|
| RUN-01 | Every status reached along its transitions only; invalid ones refused; a repeat is not an error | `test_runs.py` (LocalRuns) | `test_the_runs_client_speaks_agent_runs` | `test_state_machine.py`, `test_runs.py` | `test_run_api.py::test_every_status_is_reached_along_its_transitions_only` | pass |
| RUN-02 | Priority first, then age; one run per `concurrency_key` at a time | `test_conversation.py` | `test_a_conversations_second_run_is_claimed_after_its_first` | `test_claim_order.py`, `test_queue.py` | `test_run_api.py::test_the_queue_hands_out_by_priority_and_one_run_per_concurrency_key` | pass |
| RUN-03 | Claims, leases, heartbeats (progress saved), a lapse requeued by the ticker fencing the old worker, release, holds, `holder_lost`, the 60 s resume grace | `test_worker.py` | `test_a_lapsed_lease_puts_the_run_back_and_fences_the_old_worker` | `test_holding.py`, `test_release.py`, `test_ticker.py` | `test_run_api.py` (3 tests: lease, hold, resume grace measured ≥ 55 s) | pass |
| RUN-04 | A retryable error retried after its backoff, a later one stands; schedule breaker | `test_reliability.py` | `test_writes_of_unknown_effect_are_not_run_again_after_a_worker_dies` | `test_retries.py` | `test_run_api.py::test_a_retryable_error_is_retried_with_backoff_and_a_later_one_stands`, `test_sch.py::test_three_failed_runs_in_a_row_pause_the_schedule_and_say_so` | pass |
| RUN-05 | Webhooks `run.paused`, `run.finished`, signed (and not by a wrong secret), a rotated secret signs, redeliver, a non-http URL refused | — | `test_pauses_and_endings_are_signed_webhook_deliveries`, `test_a_pause_reaches_a_webhook_with_what_a_notice_says` | `test_webhooks.py`, `test_webhook_delivery.py` | `test_run_api.py::test_pauses_and_endings_are_delivered_signed_and_a_rotated_secret_signs` | pass (private targets are allowed in `dev`: SSRF refusal is the service suite's) |
| RUN-06 | Events appended on one replica stream live from the other; reconnect after `Last-Event-ID`; nothing after the end | — | `test_a_workers_run_is_streamed_by_another_replica_from_agent_runs` | `test_events.py` | `test_run_api.py::test_events_appended_on_one_replica_stream_live_from_the_other` | pass |
| RUN-07 | The inbox pages every waiting run once (`Link`), `h.inbox` the same | `test_inbox` (integration) | `test_approvals_choices_comments_and_results_from_outside_through_agent_runs` | `test_pagination.py` | `test_run_api.py::test_the_inbox_pages_every_waiting_run_once`, `test_har_adapters.py::test_ask_user_*` | pass |
| RUN-08 | A start repeated with its idempotency key is one run; the same key with another start is 409 naming the fields | — | — | `test_runs.py` | `test_run_api.py::test_a_start_repeated_with_its_idempotency_key_is_one_run` | pass |
| RUN-09 | Rate limits: 429 with `Retry-After` when the tenant's budget is spent, the SDK waits it | `test_runs.py` (Throttled) | — | `test_limits.py` | `test_run_ops.py::test_a_tenants_spent_budget_is_429_with_retry_after_and_the_sdk_waits_it` | pass |
| RUN-10 | Cancel at each status (queued, paused, in process, held by a worker, ended); a child names its parent; cancelling a parent cancels its waiting child | `test_subagents.py` | `test_a_running_queued_run_is_cancelled_from_another_process` | `test_cancel.py` | `test_run_api.py::test_cancel_works_at_every_status_and_a_child_names_its_parent`, `::test_a_key_restricted_to_one_person_answers_and_cancels_only_their_runs`, `test_har_adapters.py::test_a_sub_agent_is_a_child_run_and_cancelling_the_parent_cancels_it[10]` | pass |
| RUN-11 | Retention: a run ended past `RETENTION_DAYS` deleted by the ticker with its events and resolutions | — | — | `test_retention.py` | `test_run_ops.py::test_a_run_ended_past_the_retention_is_deleted_with_its_log` | pass |
| RUN-12 | Every route of `openapi.json` exercised for its purpose | — | — | `test_openapi.py` + every module | [route table](#agent-runs-routes) below | pass |
| RUN-13 | Migrations: one revision down and up keeps a run; down to base and up serves again | — | — | `test_migrations.py` | `test_run_ops.py::test_the_schema_goes_down_and_up_again_with_a_run_in_it` | pass |

### agent-runs routes

| Route | e2e test (this branch) |
|---|---|
| `POST /v1/runs` | `test_run_api.py` (every test), `test_run_ops.py` |
| `POST /v1/runs/claim` | `test_run_api.py::test_every_status…`, `::test_the_queue…`, `::test_a_lapsed_lease…` |
| `GET /v1/runs/{id}` | every test (`kit.record`) |
| `GET /v1/runs` | `::test_the_inbox_pages…`, `::test_cancel_works…`, `test_sch.py` |
| `POST /v1/runs/{id}/heartbeat` | `::test_a_lapsed_lease…`, `::test_a_hold_that_lapses…`, `::test_cancel_works…` |
| `POST /v1/runs/{id}/release` | `::test_a_lapsed_lease…`, `test_surf.py::test_the_worker_cli…` (on SIGTERM) |
| `POST /v1/runs/{id}/pause` | `::test_every_status…`, `::test_the_inbox…`, `::test_a_key_restricted…` |
| `POST /v1/runs/{id}/resume` | `::test_every_status…`, `::test_a_held_run…`, `::test_a_key_restricted…` |
| `GET /v1/runs/{id}/resolutions` | `::test_every_status…`, `test_con.py` |
| `POST /v1/runs/{id}/cancel` | `::test_cancel_works…`, `::test_another_tenants_runs…` |
| `POST /v1/runs/{id}/finish` | every lifecycle test |
| `POST /v1/runs/{id}/events`, `GET …/events`, `GET …/events/stream` | `::test_events_appended_on_one_replica…` |
| `POST /v1/runs/{id}/artifacts`, `GET /v1/artifacts/{id}` | `::test_artifacts_and_payloads_are_stored_once_and_served_back` |
| `POST /v1/runs/{id}/payloads`, `GET /v1/payloads/{sha}`, `POST /v1/payloads/read` | `::test_artifacts_and_payloads…`, `test_con.py` (a worker's journal segments) |
| `POST/GET /v1/schedules`, `GET/PATCH/DELETE /v1/schedules/{id}`, `POST …/fire` | `test_sch.py::test_schedules_are_upserted_listed_patched_and_deleted`, every SCH test |
| `POST/GET /v1/webhooks`, `GET/DELETE /v1/webhooks/{id}`, `POST …/rotate-secret`, `GET /v1/webhooks/deliveries`, `POST …/redeliver` | `::test_pauses_and_endings_are_delivered_signed…` |
| `GET /health/live`, `/health/ready`, `/metrics` | `::test_the_probes_and_metrics_answer_without_a_key` |

## Schedules (SCH)

| Feature: the purpose | before | e2e (this branch) | Status |
|---|---|---|---|
| A schedule set in chat (`schedule_task`) acts for the user in their thread; the fired run's answer lands in that conversation; it is offered no schedule tool | integration `test_schedules.py`; matrix F95, F06; live `test_a_schedule_fires_from_the_ticker_to_a_worker` | `test_sch.py::test_a_schedule_set_in_chat_fires_into_the_users_conversation` | pass |
| Cron in the user's zone; jitter within the minute after the tick | agent-runs `test_cadence.py` | same test (next fire 09:00:ss Europe/Lisbon) | pass |
| Missed fires: fired once after an outage, not per missed tick; overlap: no fire while the last run is queued | agent-runs `test_schedules.py` | `test_sch.py::test_missed_fires_fire_once_and_an_overlapping_fire_is_skipped` | pass |
| `once`: deleted after its run; `condition`: `NO_REPORT` not delivered | integration `test_schedules.py` | `test_sch.py::test_a_once_schedule_is_gone_after_its_run_and_a_condition_that_fails_is_quiet` | pass |
| Breaker: three failures in a row pause it and the error says so; a paused one does not fire | integration | `test_sch.py::test_three_failed_runs_in_a_row_pause_the_schedule_and_say_so` | pass |
| Expiry: 90 days untouched asks "Still need this?" before working; Cancel deletes it | integration | `test_sch.py::test_a_schedule_untouched_for_90_days_asks_still_need_this` | pass |
| A user who left: `on_run_start` ends the fired run before any tool | integration | `test_sch.py::test_a_fired_run_of_a_user_who_left_ends_before_any_tool` | pass |
| `agent.schedule` from code: upsert, run options on every fired run, list/patch/delete | live `test_a_schedule_carries_the_agents_limit_and_version` | `test_sch.py::test_schedules_are_upserted_listed_patched_and_deleted` | pass |
| DST forward and back | agent-runs `test_cadence.py`; harness unit | skipped: the service's clock cannot be moved end to end | skip |

## Files, query, sandbox (FS)

| Feature: the purpose | before | e2e (this branch) | Status |
|---|---|---|---|
| A file a run saves is the user's alone, served by AG-UI from another replica sharing the store | integration `test_files.py`, `test_stored_files.py`; matrix X13 | `test_fs.py::test_a_saved_file_is_served_to_its_user_alone_from_another_replica` | pass |
| `query_result(save_as=)` keeps the selected rows; a later run of the user (another process) is told of it and reads it back; another user is not told | integration `test_query_export.py`, `test_stored_files.py` | `test_fs.py::test_query_result_save_as_keeps_the_rows_and_a_later_run_reads_them_back` | pass (after H-CSV) |
| Large results kept, paged to the last part (`read_file`), counted over every row (`query_result`) | matrix F26, F26r, F26q | `test_har_adapters.py::test_a_large_result_is_kept_paged_to_its_end_and_counted_over_every_row[5]` | pass |
| query_result limits (200 rows, 20,000 chars, SQL 5,000), wide tables | unit `test_query*.py`, integration `test_query_safety.py` | — (in process is the right level: no service is involved) | prior |
| Skills (sources, learned skills in the next context) | integration `test_skills.py`; matrix F41s | live `test_an_agent_learns_a_skill_from_its_runs_and_its_next_runs_are_offered_it` | prior |
| Sandbox (per-run container, outputs, reaper) | integration `test_sandbox*.py`; matrix F73 | skipped: no Docker daemon on this machine (L20) | skip |

## Surfaces (SURF)

| Feature: the purpose | before | e2e (this branch) | Status |
|---|---|---|---|
| AG-UI: a run paused on one replica is replayed (from agent-runs' log) and resumed on another | live `test_agui_streams_pauses_resumes_and_replays` (one app) | `test_surf.py::test_agui_a_run_paused_on_one_replica_is_replayed_and_resumed_on_the_other` (two apps, real HTTP) | pass |
| AG-UI: a run's event ids are the same on every replica, so `Last-Event-ID` resumes where the client is and numbering continues across attempts | — | `test_surf.py::test_agui_event_ids_are_the_same_on_every_replica` | **xfail(strict): H-AGUI** |
| AG-UI problem codes: 401 no identity, 404 another user's run (read, cancel), 409 `BAD_RESUME`, 422 bad mode, 409 cancel of an ended run | integration `test_agui.py` | same test | pass |
| AG-UI files | integration | `test_fs.py` | pass |
| A2A: card public with no credential, saying what `h.wrap(description=)` says (H-A2A, fixed); 401 without identity, past 4096 characters, another tenant; a call answers | integration `test_a2a_server.py`; live `test_a2a_round_trip_with_a_remote_question` | `test_surf.py::test_a2a_the_card_is_public_and_a_call_needs_a_trusted_identity` | pass |
| Worker CLI: works the queue, SIGTERM releases the held run to the queue and exits 0 | unit `test_worker.py`; live `test_workers_claim_pause_resume_and_survive_a_crash` | `test_surf.py::test_the_worker_cli_works_the_queue_and_releases_on_sigterm` | pass |
| Doctor against the stack | unit `test_doctor.py` | `test_surf.py::test_doctor_finds_the_services_and_says_the_keys_tenant_and_role` | pass |

## Contracts (CON)

| Feature: the purpose | before | e2e (this branch) | Status |
|---|---|---|---|
| Every request the harness sends memory and agent-runs, and every answer, fits their committed OpenAPI documents | `tests/contract` (fakes), `test_openapi.py` | `test_con.py::test_every_route_the_harness_calls_speaks_its_openapi_document` (the routes hit cover the documented list) + INV-7 at every test's teardown | pass |
| The documented route lists name real routes | — | `test_con.py::test_the_documented_route_lists_name_real_routes` | pass |
| OpenAPI regenerated equals committed (both services) | agent-runs `test_openapi.py`, memory `contract/` | — (their suites) | prior |

## Harness inventory (HAR): every feature of the harness docs

`matrix` is the adapter count of [docs/matrix.md](../matrix.md) (in process). e2e adapters: all
ten (function, ReAct, ReAct on Deep Agents, create_agent, create_agent + `middleware()`,
hand-built graph, Deep Agents, OpenAI Agents, Claude (scripted CLI), Kimi (real `kimi acp`,
planned endpoint)) unless said.

| ID | Feature (doc) | matrix | live | e2e (this branch) | Status |
|---|---|---|---|---|---|
| HAR-GOV | governance tiers and the catalog ([governance.md](../governance.md)) | F32, F33: all | yes | `test_mem_governance.py` | pass |
| HAR-APR | approvals, one question, a rejection read by the model | F04, F36: all | yes | `test_mem_records.py` (10) | pass on 8; **error on Claude and Kimi: H-ID** (INV-10) |
| HAR-ALW | always allow | F33 | yes | `test_mem_governance.py` (10) | pass |
| HAR-ASK | `ask_user` (date, choices, form, asked once) | F36u/m/f/o: all | — | `test_har_adapters.py::test_ask_user_…[10]` (paused in agent-runs, answered in another process, a misfit by type refused); `::test_a_date_question_refuses_an_answer_that_is_not_a_date` | pass; **xfail: F-DATE** |
| HAR-BUDGET | budgets and limits ([runs.md](../runs.md#spending-a-budget-per-run)) | integration `test_budget.py` | — | `test_har_adapters.py::test_a_budget_ends_the_run_partial…[10]` | pass |
| HAR-TIME | run time limit, stalled model, stall guard | F09: all | `test_working_time_counts_across_a_crash…` | `test_har_more.py::test_a_model_that_never_answers…[10]`, `::test_the_stall_guard_ends_a_stalled_step` | pass |
| HAR-STEP | the step limit | integration `test_react_long_tasks.py` | — | `test_har_more.py::test_the_step_limit…` | pass |
| HAR-CTX | context fitting and compaction | F65, F65f (react, middleware) | — | — (in process: the model's window, no service) | prior |
| HAR-DATE | the date line | integration `test_tool_updates.py` | — | `test_har_adapters.py::test_the_model_is_told_todays_date[9]` | pass |
| HAR-LARGE | large results, read_file paging, query_result | F26, F26r, F26q | `test_a_large_table_is_a_run_artifact_served_back` | `test_har_adapters.py` (5), `test_fs.py` | pass |
| HAR-RISK | tool risk (undeclared asks, MCP hints) | F32 | yes | `test_har_more.py::test_an_mcp_servers_tools…` (a read-only hint runs unasked), `test_mem_governance.py` | pass |
| HAR-MCP | MCP servers of the agent's own, resources, elicitation | F18s/r/e: all | gateway (skipped: no gateway) | `test_har_more.py::test_an_mcp_servers_tools_are_governed_and_recorded[10]` | pass |
| HAR-A2A / AGUI | A2A and AG-UI surfaces | F62, F63, F16 | yes | `test_surf.py` | pass |
| HAR-EVAL | evals and judges (grounding verified by memory) | F54, F55: all | `test_a_sampled_answer_is_verified_against_its_context`, `test_online_judges_score_every_sampled_run` | `test_har_more.py::test_evaluate_verifies_answers_against_their_contexts_in_memory` | pass |
| HAR-OBS | observability and redaction | F57, F61: all | — | `test_har_adapters.py::test_a_credential_never_reaches_a_trace[10]`; INV-8 at every teardown | pass |
| HAR-DOC | doctor | — | — | `test_surf.py` | pass |
| HAR-WRK | worker CLI | F03 | yes | `test_surf.py` | pass |
| HAR-SCH | schedules from chat | F95: all | yes | `test_sch.py` | pass |
| HAR-SUB | sub-agents and child runs | F30p, F30c, X08-X11: all | `test_a_planner_with_two_children_one_asking_survives_a_killed_worker` | `test_har_adapters.py::test_a_sub_agent_is_a_child_run…[10]` | pass |
| HAR-HLP | helpers (framework's own sub-agents) | integration `test_native_helpers.py` | `test_a_deep_agents_sub_agent_call_waits_in_agent_runs` | — | prior |
| HAR-HOOK | hooks | F71*: all | — | `test_sch.py` (`on_run_start`) | prior + pass |
| HAR-REL | reliability: idempotent writes, retries, unknown outcomes | F21, F22, F24: all | `test_a_read_that_meets_a_busy_service_is_retried_and_a_write_sends_its_key` | `test_mem_resilience.py` (504 retries, writes once), `test_mem_records.py::test_a_turn_retried…` | pass |
| HAR-FILE | files and the file store | X13 | — | `test_fs.py` | pass |
| HAR-SKILL | skills | F41s | yes | — | prior |
| HAR-SBX | sandbox | F73 | skipped (Docker) | skipped (Docker) | skip |
| BG | background work | integration `test_background_by_adapter.py` | — | `test_har_more.py::test_background_work_of_an_unmerged_branch[3]` | **xfail(strict): pending on react-background, bg-kimi, bg-claude-da** |
| HAR-STREAM | streamed text, events (F02, F60) | all | yes | INV-10 at every teardown | prior |
| HAR-JRN | journal, segments, progress checkpoints (F23, F12) | all | `test_a_journal_larger_than_a_checkpoint_survives_a_crash_and_a_pause` | `test_con.py` (segments through a worker) | prior + pass |
| HAR-CONV | conversations, mode, busy chats | X01, X02 | yes | scenario suite's (CONV) | owner: scenarios |

## Gaps (not proven end to end, why)

* **MEM-06 retention** and **MEM-15 the archive read path** are the memory service's periodic
  jobs (daily retention; the hourly archive purge after a 24 h grace). No route runs them on
  demand, so their tests queue the job the way the worker's scheduler does (one row in
  Procrastinate's queue: `conftest.run_periodic_now`) and move a row's age back
  (`conftest.move_clock`): the only two writes the suite makes to a service's database.
* **DST**: the service's clock. **Sandbox**: Docker. **Gateway features** (Code Mode, the key's
  MCP tools, stored prompts): no Bifrost gateway here (live suite's 30 skips).

## Findings

| ID | Where | What does not serve its purpose | Root cause | Fix |
|---|---|---|---|---|
| BUG-M1 | agent-memory-service | `memory_forget` of a fact read out of the user's own words runs unasked: the agent retracts what the user said, against docs/memory.md (asks a person first) | the fact is owned by `agent:<user>/<agent>`; `said_by_a_person` (`modules/memory/pipeline.py:358`) only knows a `user:` owner or a `said` turn, so `_require_owner` (`modules/memory/service.py:276`) lets the agent | proposed: persist the reading's origin (`system_metadata["said_by_user"]` from `said_by_user(observation)` at classify, `native.py:1231`) and read it in `said_by_a_person`; backfill from each memory's evidence message. Test: `test_forgetting_a_fact_read_from_the_users_words_asks_first` (xfail strict) |
| BUG-M2 | agent-memory-service | the user is refused forgetting their own statement relayed by the harness, so the documented flow (ask, approve, forget as the user) never forgets | `_require_owner` (`service.py:288`) accepts only the owner principal or `user:<id>`, not "the user an agent acts for" its docstring promises | **fixed** on agent-memory-service `claude/cool-maxwell-w5avo8-features` (`ab3546f`), failing test first (`tests/integration/test_agent_relayed_user_facts.py`) |
| BUG-M3 | agent-memory-service | the user's inventory (`GET /v1/memories`, "what do we hold about this user?") misses every statement anchored in a thread | `list_memories` (`service.py:322-396`) lists by scope anchors; ADR 0036 anchors a user's statements at the thread with USER visibility | proposed: without a `thread_id`, also list the memories whose visibility keys hold `user:<t>/<u>` (any anchor). Test: `test_the_users_inventory_lists_what_they_said_in_a_chat` (xfail strict) |
| BUG-M4 | agent-memory-service | the tenant's admin key cannot forget a user's memory: with BUG-M2, nothing but the relaying agent could erase a user | `is_tenant_admin` (`modules/authz/service.py:170`) checks an authorization tuple of the acting principal only, never the key's `admin` role (`api/deps.py:481` `is_tenant_administrator`) | proposed: pass the key's role into the memory service's owner check (an `admin`/`platform` key is a tenant admin). Test: `test_erasing_a_user_leaves_nothing_any_read_returns` (xfail strict) |
| BUG-M5 | agent-memory-service SDK (and service) | a second user of a tenant adding the same file (same bytes and name) got back the first user's document id — which they may not read (`403` on the next read) — and their own was never stored: `h.add_document` failed for them | the SDK's derived upload key (`sdk/python/src/trellis/memory/advanced.py:96`) covered the bytes and the form but not the bound scope; the service replayed the first answer although the form's `scope` differed | **fixed in the SDK** on agent-memory-service `claude/cool-maxwell-w5avo8-features` (`a745cee`), failing test first (`sdk/python/tests/test_client.py`); proposed for the service: compare the upload's scope when a key is replayed and answer `409` (`differing`) as the other writes do. Test: `test_mem_recall.py::test_two_users_adding_the_same_file_each_get_a_document_of_their_own` |
| H-DOC | agent-harness | `trellis doctor` reported every working agent-runs as failing (`401`) | the runs check sent `Authorization: Bearer`; agent-runs reads `X-API-Key` only (`doctor.py:420`) | **fixed** here (`f5cf2b0`), failing unit test first |
| F-504 | agent-memory-service (contract) | the service answers `504 TIMEOUT` (its deadline middleware, `api/deadline.py:107`) on 69 operations (every one but the probes, `/metrics` and uploads) whose OpenAPI lists no `504` | `error_responses(...)` in each router omits 504 | proposed (needs approval: a contract change): add `504` (the `Problem` schema, `code: TIMEOUT`, `retryable: true`) to every operation `route_class()` gives a deadline — all `/v1` routes but `POST /v1/documents` — e.g. once in the app's OpenAPI hook, then `make openapi`. The harness handles it: `test_mem_resilience.py` (504 retried, a write committed then answered 504 stored once) |
| F-DATE | agent-runs SDK (`trellis.runs.answers`, used by agent-runs and the harness) | a date question (`ask_user(kind="date")`, `expects: {"type": "string", "format": "date"}`) takes any string: "next week" resumes the run and the model reads it as the date | `validator_for(expects)(expects)` (answers.py:57) checks with no format checker, and JSON Schema's `format` is then only an annotation | proposed: `validator_for(expects)(expects, format_checker=validator_for(expects).FORMAT_CHECKER)` (and the same on `check_schema`'s side), with a failing test in agent-runs `tests/test_answering.py`. Test: `test_har_adapters.py::test_a_date_question_refuses_an_answer_that_is_not_a_date` (xfail strict) |
| H-A2A | agent-harness | an A2A card ignored `h.wrap(description=)`: a caller choosing an agent by its card read the target's docstring, not what the operator said it does | `_description` (`a2a/server.py`) read the target's own description and docstring only | **fixed** here (`c549d44`), failing integration test first (`test_a2a_server.py::test_the_card_says_what_h_wrap_was_told_the_agent_does`) |
| H-AGUI | agent-harness | AG-UI event ids of one run differ by replica: after `Last-Event-ID: 1` another replica sends again an event the client has, and an attempt resumed on another replica numbers from the start again, against docs/surfaces.md ("any replica resumes a stream", ids "those of the run's events") | the serving replica numbers the events it streams itself (from 0, the interrupt folded into the finishing event); a reconnect replays agent-runs' log and numbers by log position (from 1, the interrupt counted) | proposed: the serving replica sends each event under its agent-runs log position (the id a reconnect uses), on the first attempt and on a resume. Test: `test_surf.py::test_agui_event_ids_are_the_same_on_every_replica` (xfail strict) |
| H-ID | agent-harness | on Claude and Kimi a resumed run's tool calls reuse one id across attempts: two different calls are both `run_<id>:1` in the run's events (INV-10: "tool calls started more than once"), so a client matching results to calls by id (AG-UI, the run's event log) pairs the wrong ones | `tools/bridge.py:359` `ref = call_id or f"{runtime.run_id}:{step}"`: these bridges give no call id, and `step` counts from 1 again in every attempt | proposed: when the framework gives no id, name the call by its `idempotency_key` (line 355: `"{run_id}:{key}:{occurrence}"`, this run's n-th such call, the same in every attempt and different for different calls) instead of the attempt's step. Tests: `test_mem_records.py::…[claude_agent_sdk]`, `[kimi]` (error at teardown, INV-10) |
| H-CSV | agent-harness | `query_result(save_as="csv")` of one text column, read back in a later run, counted one row too many (208 for 207): the kept file's header was read as a row | the kept text went through `query._shape`, which takes one column of text (`line` over values) for lines | **fixed** here (`4d78d5e`): a kept `.csv` is read under its first line (`query._kept_file`); failing integration test first (`test_stored_files.py::test_a_kept_csv_of_one_text_column_is_read_under_its_header`) |
| L-NEW | platform | a harness process started while the memory service is down runs nothing (`ConfigurationError`): the tenant a key speaks for is the memory service's word | by design (memory is the key registry; agent-runs too, after its 60 s key cache) | not in the plan's §6 list; `test_a_process_started_while_memory_is_down_runs_nothing` documents it |
| INV | tests/support/invariants.py | four false positives of the shared checker | a memory agent tool (logged by the service) counted as unrecorded; the one answer that shows a new key read as a leak; a call with no id matched by its tool alone; a call that paused its run counted as run and as started twice | fixed here (adopting the scenarios job's version, f191ec4, and adding these) |

## How to run

The suite needs a running memory service and agent-runs (two API replicas on one database, and
its ticker), and reads their databases with `psql`. It skips every test when they are not set.

```bash
export MEMORY_URL=http://127.0.0.1:18680 RUNS_URL=http://127.0.0.1:18690 \
       FEAT_RUNS_URL_2=http://127.0.0.1:18691            # the second agent-runs replica
export TRELLIS_API_KEY=...        # a service key of tenant A (acts for anyone)
export TRELLIS_ADMIN_KEY=...      # tenant A's administrator key
export TRELLIS_API_KEY_OTHER=...  # a service key of tenant B (the tenant boundary)
export FEAT_KEY_ALICE=...         # a service key of tenant A restricted to user:alice
export FEAT_PLATFORM_KEY=...      # the memory service's platform (bootstrap) key
export FEAT_MEMORY_DSN=postgresql://.../memory FEAT_RUNS_DSN=postgresql://.../runs
export FEAT_RUNS_LAUNCH=.../start_runs.sh   # starts an agent-runs API: `api <port>`, `migrate`,
                                            # `downgrade -- <rev>` (RUN-09, RUN-13)
export TRELLIS_MEMORY_OPENAPI=.../agent-memory-service/docs/openapi.json \
       TRELLIS_RUNS_OPENAPI=.../agent-runs/docs/openapi.json
make test-features          # or: pytest -m features tests/features -k MEM  (by file or name)
pytest -m features tests/features --collect-only -q   # every test, its plan IDs in its markers
```

The agent-runs ticker runs with `RUNS__LIMITS__RETENTION_DAYS=30` (RUN-11); the memory
service with its job worker in the API process (the live suite's `mem_api.py`), no model.
Every test's teardown checks the plan's invariants INV-1..INV-10
(`tests/support/invariants.py`).
