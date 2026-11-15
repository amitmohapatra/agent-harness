### claude_agent_sdk

| feature | bare | off (same?) | stack (records) | on |
|---|---|---|---|---|
| background_bash | pass | pass = | pass (pass) | - |
| builtin_read | pass | pass = | pass (pass) | - |
| builtin_write | pass | pass = | pass (pass) | - |
| can_use_tool | pass | pass = | - | - |
| lifecycle_cancel | - | - | pass (FAIL) | - |
| lifecycle_pause | - | - | pass (-) | - |
| lifecycle_queued | - | - | pass (pass) | - |
| lifecycle_resume | - | - | pass (pass) | - |
| live_session | pass | pass = | pass (pass) | - |
| max_turns | pass | pass = | - | - |
| own_hooks | pass | pass = | - | - |
| partial_stream | pass | pass = | - | - |
| plan_mode | pass | pass = | - | - |
| sdk_tool | pass | pass = | pass (pass) | - |
| session_resume | pass | pass = | pass (pass) | - |
| stdio_mcp | pass | pass = | - | - |
| structured | pass | pass = | - | - |
| subagent | pass | FAIL ≠ | pass (pass) | - |

### deepagents

| feature | bare | off (same?) | stack (records) | on |
|---|---|---|---|---|
| files | pass | pass = | pass (pass) | pass |
| interrupt_on | pass | pass = | pass (pass) | pass |
| lifecycle_cancel | - | - | pass (FAIL) | - |
| lifecycle_pause | - | - | pass (-) | - |
| lifecycle_queued | - | - | pass (pass) | - |
| lifecycle_resume | - | - | pass (pass) | - |
| memory_files | pass | pass = | pass (pass) | - |
| permissions | pass | pass = | - | - |
| skills | pass | pass = | - | pass |
| store_backend | pass | pass = | pass (pass) | - |
| structured | pass | pass = | - | pass |
| subagent | pass | pass = | pass (pass) | pass |
| todos | pass | pass = | pass (pass) | FAIL |

### kimi

| feature | bare | off (same?) | stack (records) | on |
|---|---|---|---|---|
| agents_md | pass | pass = | - | - |
| background | pass | FAIL ≠ | FAIL (FAIL) | - |
| bash_write | pass | pass = | pass (pass) | - |
| lifecycle_cancel | - | - | pass (pass) | - |
| lifecycle_pause | - | - | pass (-) | - |
| lifecycle_queued | - | - | pass (pass) | - |
| lifecycle_resume | - | - | pass (pass) | - |
| mcp | FAIL | FAIL = | FAIL (pass) | - |
| plan_mode | pass | pass = | - | - |
| read | pass | pass = | pass (pass) | - |
| session | pass | pass = | pass (pass) | - |
| skills | pass | pass = | - | - |
| subagent | pass | pass = | pass (pass) | - |
| todo | pass | pass = | - | - |

### langgraph

| feature | bare | off (same?) | stack (records) | on |
|---|---|---|---|---|
| checkpointer | pass | pass = | pass (pass) | pass |
| context_schema | pass | pass = | - | - |
| custom_stream | pass | pass = | - | - |
| fallback | pass | pass = | - | - |
| hitl | pass | pass = | pass (pass) | pass |
| lifecycle_cancel | - | - | pass (FAIL) | - |
| lifecycle_pause | - | - | pass (-) | - |
| lifecycle_queued | - | - | pass (pass) | - |
| lifecycle_resume | - | - | pass (pass) | - |
| own_interrupt | pass | pass = | pass (pass) | pass |
| pii | pass | pass = | pass (pass) | - |
| store | pass | pass = | pass (pass) | - |
| stream | pass | pass = | - | pass |
| structured | pass | pass = | pass (FAIL) | pass |
| tool_retry | pass | pass = | pass (pass) | - |
| tools | pass | pass = | pass (pass) | pass |

### openai_agents

| feature | bare | off (same?) | stack (records) | on |
|---|---|---|---|---|
| agent_tool | pass | pass = | pass (pass) | - |
| context | pass | pass = | - | - |
| context_per_run | pass | FAIL ≠ | - | - |
| handoff | pass | pass = | pass (pass) | - |
| hooks | pass | pass = | - | - |
| hooks_per_run | pass | FAIL ≠ | - | - |
| input_guard | pass | pass = | pass (pass) | - |
| lifecycle_cancel | - | - | pass (FAIL) | - |
| lifecycle_pause | - | - | pass (-) | - |
| lifecycle_queued | - | - | pass (pass) | - |
| lifecycle_resume | - | - | pass (pass) | - |
| max_turns | pass | pass = | - | - |
| mcp | pass | pass = | pass (pass) | - |
| model_settings | pass | pass = | - | - |
| needs_approval | pass | pass = | pass (pass) | - |
| output_guard | pass | pass = | - | - |
| sandbox_docker | pass | pass = | pass (pass) | - |
| sandbox_local | pass | pass = | - | - |
| session | pass | pass = | pass (pass) | - |
| stream | pass | pass = | - | - |
| structured | pass | pass = | pass (pass) | - |
| tool_use_behavior | pass | pass = | - | - |
| tools | pass | pass = | pass (pass) | - |

### react

| feature | bare | off (same?) | stack (records) | on |
|---|---|---|---|---|
| lifecycle_cancel | - | - | pass (FAIL) | - |
| lifecycle_pause | - | - | pass (-) | - |
| lifecycle_queued | - | - | pass (pass) | - |
| lifecycle_resume | - | - | pass (pass) | - |
| r_approval | - | pass | pass (pass) | - |
| r_files | - | pass | pass (pass) | - |
| r_plan_mode | - | pass | pass (pass) | - |
| r_reject | - | pass | pass (pass) | - |
| r_structured | - | pass | pass (FAIL) | - |
| r_subagent | - | pass | pass (pass) | - |
| r_tools | - | pass | pass (pass) | - |

### swarm

| feature | bare | off (same?) | stack (records) | on |
|---|---|---|---|---|
| s_handoff | pass | pass = | pass (FAIL) | pass |
