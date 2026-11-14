# Earlier material kept with the handoff

Files that existed only on the build machine and are kept here so nothing is lost.

- `reports/duplication-audit.md` (31 Oct): the duplication audit of the stack.
- `reports/holistic-haiku-evaluation.md` (1 Nov): the holistic evaluation with the Haiku model.
- `trellis-confluence.tar.gz` (3–10 Nov): the documentation set written for Confluence
  (start page, glossary, getting started, user guide, frameworks, architecture, end-to-end
  flows, quality and performance, operations, reference, roadmap; about 50 Markdown files).
  It is an archive because it describes all five repositories at once (other repositories'
  make targets and APIs), which this repository's docs check would reject file by file.
  Unpack with `tar -xzf trellis-confluence.tar.gz`. It predates this round: check it against
  the current code before publishing it anywhere. The Word version of the same set was sent
  to the owner as a file and is not in the repository (12 MB of binaries).
- `../STOP_NOTE.txt`, `../CONTINUE_NOTE.txt`, `../CLEAN_CODE_NOTE.txt`: short instructions
  sent to the coding jobs during the round, kept for the record.
- `experiments/memory-assistant-words-NOTES.txt` and the two `memory-longmemeval-step-*-report.txt`
  files: working notes and score reports of an earlier memory experiment (the
  "assistant words" work and its LongMemEval retrieval comparisons). They name folders and
  databases on the build machine that no longer exist; the numbers are the record.

Code that existed only on the build machine was pushed to backup branches (not for merging):
in agent-harness `claude/cool-maxwell-w5avo8-backup-aa-both`, `-backup-aa-final`,
`-backup-aa-run`, `-backup-aa-score`, `-backup-aa-verify` and
`claude/cool-maxwell-w5avo8-asked-answer` (local test merges and one benchmark commit of an
earlier experiment); in agent-memory-service
`claude/cool-maxwell-w5avo8-backup-router-experiment` (two source files of a retrieval-router
experiment that concluded with nothing restored).

Not kept anywhere: raw experiment outputs (answer files, logs, per-question scores, test
result XML, benchmark databases; several hundred MB to several GB), tool scratch output of
the reviews, drafts of pull request descriptions (their final text is on the pull
requests), and virtual environments.
