# Versioning

The harness pins what it tested, and moves only by a release. Each framework extra in
`pyproject.toml` admits the minor range a release was tested with — no wider — so an install
resolves inside it; a newer framework arrives with a newer `trellis-harness`, after the feature
matrix and the live tests have run against it. The decision and its reasons are
[ADR 0001](adr/0001-version-policy.md).

## Compatibility

What each release was tested with. The ranges are the extras' pins; the tested versions are
the lock's (`uv.lock`), against which `make test`, `make matrix` and `make examples` ran.

### trellis-harness 0.4.0

| Distribution | Extra | Pinned range | Tested |
|---|---|---|---|
| `langgraph` | `langgraph` | `>=1.2,<1.3` | 1.2.11 |
| `langchain` | `langgraph` | `>=1.4,<1.5` | 1.4.2 |
| `langchain-core` | `langgraph` | `>=1.6,<1.7` | 1.6.5 |
| `langchain-openai` | `react` | `>=1.6,<1.7` | 1.6.6 |
| `deepagents` | `deepagents`, `react` | `>=0.7.19,<0.8` | 0.7.19 |
| `openai-agents` | `openai-agents` | `>=0.22.3,<0.23` | 0.22.3 |
| `claude-agent-sdk` | `claude-agent-sdk` | `>=0.2.160,<0.3` | 0.2.160 |
| `a2a-sdk` | `a2a` | `>=1.1,<1.2` | 1.1.5 |

| Sibling | Pinned | Tested | Service |
|---|---|---|---|
| `trellis-contracts` | `>=0.6.1,<0.7` | 0.6.1 | — |
| `trellis-memory` | `>=0.4` | 0.4.0 | agent-memory-service 0.3.0 (`docs/openapi.json`) |
| `trellis-runs` | `>=0.4.0` | 0.4.1 | agent-runs 0.4.0 (`docs/openapi.json`) |
| `bifrost-sdk` | `>=0.3` | 0.3.0 | the Bifrost gateway |

`trellis-contracts` 0.6.1 is the floor because schedules carry the run options in their
metadata (`ScheduleSpec.metadata`, priority and concurrency key) from that version. The
platform is not yet published: the siblings are path sources of the same checkout
([README](../README.md#install)).

The ranges live in one place, `src/trellis/harness/compat.py` (`TESTED`); `pyproject.toml`
says the same and `tests/unit/test_compat.py` fails when they disagree.

## When the installed version is outside the range

An environment can still hold another version — an application's own lock, an override, a
`pip install --upgrade`. The harness never refuses it: when the first target of a framework is
wrapped, it logs one warning per distribution (logger `trellis.harness`):

```text
openai-agents 0.30.1 is installed, but this trellis-harness was tested with
openai-agents>=0.22.3,<0.23: use a newer trellis-harness (docs/versioning.md)
```

The run may well work; nobody has proven it does. Either install a version in the range, or
upgrade the harness to a release that tested yours.

## How to upgrade

**An application** upgrades `trellis-harness`, and the framework comes with it:

```bash
pip install --upgrade 'trellis-harness[langgraph]'   # the extra brings the tested range
```

Read the [CHANGELOG](../CHANGELOG.md) for what changed; a release that moves a framework's
range says so there and in the table above.

**The harness** (a maintainer) moves a range only in a release:

1. Read the weekly canary's report (`.github/workflows/canary.yml`, the `canary` artifact): it
   installs the latest release of every framework, unpinned, and runs `make test` and
   `make matrix`, so a new version's breakage is known before anyone depends on it.
2. Raise the range in `src/trellis/harness/compat.py` and the extra in `pyproject.toml`
   together, relock (`uv lock`), and fix what the gates find: `make check`, `make matrix`,
   `make examples`, `make test-live` against the services.
3. Update the table above and the CHANGELOG, and release.

## Releases

The harness follows semantic versioning on its public API (`trellis.__all__`, the blocks and
the documented behaviour). Before 1.0 a minor release may change the API; the CHANGELOG says
how to move. Each release records the ranges above; the canary never changes them.
