# Manifest: retired handoff notes and earlier material

Their still-relevant parts are in [REMAINING.md](REMAINING.md) (sections 12 and 13) and [DECISIONS.md](DECISIONS.md) (the round's working orders).

The files below moved off `main` on 2026-11-15 to keep its tree small; none was lost. They
are on the orphan branch **`data/research-archive`** of this repository, at the same paths
(one commit per run). To get them back:

```bash
git fetch origin data/research-archive
git restore --source=origin/data/research-archive -- <path>   # one file, or a folder
```

Check a file against this list with `sha256sum <path>`. 10 files, 377,186 bytes.

| path | bytes | sha256 |
|---|---|---|
| `docs/research/handoff/STOP_NOTE.txt` | 975 | `89b95729db2efd8c91ab7aef4f8660da36fc1d4cab0e8c8d29f1e6d34abe5861` |
| `docs/research/handoff/CONTINUE_NOTE.txt` | 1,157 | `7e4dbdaeef3aa57ac831d158faf80c393f1544a52c9f845a4783a6d97ed7700c` |
| `docs/research/handoff/ROUND3_NOTE.txt` | 2,969 | `053b85066ad6588c77a54a5bc0f0329f48eaf3e512e723d0a7548e3f20bb3802` |
| `docs/research/handoff/earlier/README.md` | 2,339 | `16f0937e40bd3d514f682bd7a6595e61d12fb4222710f973d7c768df92474dbc` |
| `docs/research/handoff/earlier/experiments/memory-assistant-words-NOTES.txt` | 32,246 | `39460c487e76e0cc058858f127f697d01224c7da536c95dab6fac05573bd0c70` |
| `docs/research/handoff/earlier/experiments/memory-longmemeval-step-b-report.txt` | 20,266 | `9962746e5c600af8a5cadb27512561fedd8448f6a08e9d5da9d01e4bf3418332` |
| `docs/research/handoff/earlier/experiments/memory-longmemeval-step-d-report.txt` | 4,015 | `b720fcba4378e22b5e737895ee2e5688bf883864149d8540ea0fc6bf16c1a62a` |
| `docs/research/handoff/earlier/reports/duplication-audit.md` | 23,390 | `5e84735c01e2a89dd4cf14605ce0f3d908256c4ec639532fdab54dfd24d22688` |
| `docs/research/handoff/earlier/reports/holistic-haiku-evaluation.md` | 25,957 | `8d7f9f845dca2272341c5dbbe8e5c9f1f0639f68ec417947d6f7add27d1b03dd` |
| `docs/research/handoff/earlier/trellis-confluence.tar.gz` | 263,872 | `ea0beb44e81006f4caa60a36ee7c254e9fee998ae57713d8c0f95528913a45a0` |
