## What and why

<!-- What changes, and the gap it closes (link the docs/research/IMPROVEMENT_BACKLOG.md row). -->

## Challenge the change ([design rules](https://github.com/amitmohapatra/agent-harness/blob/main/docs/research/DESIGN_RULES.md#challenge-every-change))

Answer each question with evidence: a file, a test, a number.

1. **The right approach at all?** Which alternatives were considered (the adapter's own mechanism, an existing library), and why was each rejected?
2. **The right data structure and algorithm?** Time and space on the hot path, and the worst case at 50k rows or a long chat. No full scans or copies where streaming or paging fits.
3. **The right design pattern?** One owner per concern, a strategy or capability instead of branches, no duplication.
4. **Accuracy evidence:** the tests, written first and failing on main, and the judge's verdicts under [EVAL_PROTOCOL.md](https://github.com/amitmohapatra/agent-harness/blob/main/docs/research/EVAL_PROTOCOL.md).
5. **Latency, tokens per layer, memory, cost:** before and after, as numbers.
6. **What can be removed now?**
