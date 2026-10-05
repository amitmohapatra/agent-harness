# Getting started: onboard a tenant

What it takes to go from a running memory service and agent-runs to an application whose
agents remember, pause for people and run on workers. Four steps; the first two are done once
per tenant, by people who hold the keys to do them.

> **What changes for me?** Nothing, if your application already runs with a service key: the
> keys the memory service issues may act for anyone by default, and such a key answers any
> paused run, as it always did. What is new is a choice: a key restricted to one person
> (step 4) now answers only that person's runs.

## 1. The operator creates the tenant

The memory service's operator holds the **platform key** (the service's
`MEMORY__AUTHENTICATION__BOOTSTRAP_ADMIN_KEY`). Creating a tenant returns the tenant and its
first **admin key**, whose secret is shown once: hand it to the tenant's administrator.

```bash
curl -X POST "$MEMORY_URL/v1/admin/tenants" \
  -H "X-API-Key: $PLATFORM_KEY" -H "Content-Type: application/json" \
  -d '{"name": "Acme", "tenant_id": "acme"}'
# 201 {"tenant": {"tenant_id": "acme", ...}, "admin_key": {"key_id": "...", "role": "admin", "token": "mk_..."}}
```

```python
from trellis.memory import MemoryClient

async with MemoryClient(MEMORY_URL, api_key=PLATFORM_KEY) as platform:
    created = await platform.admin.create_tenant(
        "Acme", tenant_id="acme", idempotency_key="onboard-acme"
    )
    admin_key = created.admin_key.token  # shown once (None on an idempotent replay)
```

## 2. The tenant's admin issues the application's key

The application (its agents, its workers, its UI backend) holds a **service key**. Issued
without `may_act_as`, it may act for anyone in the tenant, which is what an application that
serves many users needs.

```bash
curl -X POST "$MEMORY_URL/v1/keys" \
  -H "X-API-Key: $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"role": "service", "name": "my-app"}'
# 201 {"key_id": "...", "role": "service", "may_act_as": ["*"], "token": "mk_..."}
```

```python
async with MemoryClient(MEMORY_URL, api_key=ADMIN_KEY) as admin:
    issued = await admin.tenant.keys.issue("service", "my-app")
    app_key = issued.token  # shown once
```

Keep the admin key for administration (keys, workspaces); give agents the service key, never
the admin key.

## 3. Configure the application and its workers

The same environment for the process that runs the agents and for every worker:

```bash
export TRELLIS_API_KEY=mk_...                # the service key from step 2
export MEMORY_URL=https://memory.example.com
export RUNS_URL=https://runs.example.com      # durable runs, the inbox, workers, schedules
export BIFROST_URL=https://bifrost.example.com/v1   # MCP tools and model names, when used
export BIFROST_VIRTUAL_KEY=sk-bf-...                # the agent's models, tools and budget
```

`Harness()` (and each block's `from_env`) reads exactly these; every variable is in
[configuration.md](configuration.md). One key serves both services: agent-runs keeps no keys
and needs no onboarding of its own. It asks the memory service who a key is
(`GET /v1/keys/self`) and learns the tenant from the answer, so nothing names the tenant
anywhere. Revoking a key or suspending the tenant in the memory service applies to agent-runs
too, within its 60-second cache of each key.

## 4. Optional: a key per person, for an approvals UI

An approvals UI, or a person's own client, can hold a key restricted to that person instead of
the application's key. The admin issues it with `may_act_as`:

```python
async with MemoryClient(MEMORY_URL, api_key=ADMIN_KEY) as admin:
    issued = await admin.tenant.keys.issue("service", "priya-approvals", may_act_as=["user:priya"])
```

What it may do in agent-runs, by the rule of
[who may answer a paused run](https://github.com/amitmohapatra/agent-runs/blob/main/README.md#who-may-answer-a-paused-run):

| | The application's key (`may_act_as` `*`), or an admin key | Priya's key (`may_act_as=["user:priya"]`) |
|---|---|---|
| Read runs, list any inbox | yes | yes: reads are tenant-wide for every key, the assignee is only a filter |
| Answer a run assigned to `user:priya` | yes, as any reviewer | yes, as `priya` (`reviewer="priya"` or `"user:priya"`) |
| Answer an unassigned run | yes | yes, as `priya` |
| Answer a run assigned to `user:raj` | yes | no |
| Answer a run assigned to `role:finance` | yes | no: agent-runs cannot see who is in a group |
| Answer as `raj` | yes | no |

A refusal is a `403` (`trellis.runs.AuthorizationError`) whose message says why, for example
`the run is assigned to role:finance, a group: a key restricted to listed people cannot answer
it; answer with the application's key or an admin key`. So a run assigned to a group, or
escalated to one, is answered through the application (its UI backend holds the application's
key and names the person as `reviewer`), or with an admin key.

An `ask` is the run's user's to answer unless it names someone else
([interrupts.md](interrupts.md#asking)): `ask("…")` in a run for `user="priya"` is assigned to
`user:priya`, which her key answers. A run started in process continues in the process that
answers it, so a harness holding her key also runs the rest of the agent with it; in the memory
service a restricted key acts only for whom it lists, so list the agent too
(`may_act_as=["user:priya", "agent:procurement"]`) for its memory calls to be allowed.
