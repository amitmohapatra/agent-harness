# 0001. The harness pins the framework versions it tested and moves only by a release

**Status:** Accepted (2026-10-06)

## Context

The harness attaches to six kinds of targets through the frameworks' public APIs: LangGraph and
LangChain's `create_agent`, Deep Agents, the OpenAI Agents SDK, the Claude Agent SDK, and A2A.
Each moves fast and pre-1.0 frameworks change their APIs in minor releases. Until now the extras
admitted every version below the next major (`langgraph>=1.2,<2`, `a2a-sdk>=1.1,<2`), so a
fresh install could resolve a framework nobody had run the harness against. The adapters read
exactly what a framework returns (LangGraph's interrupts and checkpoints, an OpenAI Agents
`RunState`, the Claude CLI's stream protocol, LangChain's middleware hooks); a change there
breaks pauses and resumes silently, not at import.

The harness proves its behaviour per framework: `make test` at full coverage, the generated
feature matrix (`make matrix`: every feature × adapter × way × mode × selection), the examples,
and the live tests against the services. That proof holds for the versions it ran against.

## Decision

1. **Pin what was tested.** Each framework extra in `pyproject.toml` admits the minor range
   the release was tested with (`langgraph>=1.2,<1.3`, `openai-agents>=0.22.3,<0.23`, ...). No
   upgrade is made outside a release.
2. **One place.** The ranges are kept in `src/trellis/harness/compat.py` (`TESTED`), and a test
   fails when `pyproject.toml` says otherwise.
3. **Warn, never refuse.** When a target is wrapped and its framework's installed version is
   outside the range, the harness logs one warning per distribution naming the version, the
   tested range and "use a newer trellis-harness". It does not refuse: an application may have
   reasons for its own lock.
4. **A canary, not a gate.** A weekly workflow installs the latest release of every framework,
   unpinned, runs `make test` and `make matrix`, and uploads a report. It never blocks a pull
   request; it tells the maintainers what the next release must handle.
5. **Moving a range is a release.** A maintainer raises the range in `compat.py` and the
   extra together, relocks, makes every gate pass, updates [versioning.md](../versioning.md)
   and the CHANGELOG, and releases.

## Consequences

* An install resolves to versions the harness was proven against; a framework's breaking
  minor release cannot reach users through `pip install trellis-harness[...]`.
* Users wanting a newer framework wait for a harness release; the canary shortens that wait by
  showing the breakage early. An override still works, with a warning in the log.
* The sibling SDKs (`trellis-contracts`, `trellis-memory`, `trellis-runs`, `bifrost-sdk`) are
  versioned with the platform and checked on the wire against the services' OpenAPI documents;
  `trellis-contracts` is pinned to its tested minor (`>=0.6.1,<0.7`).
