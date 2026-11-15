# Answer from the record, not from recall

> Research report kept as a living reference for improving the harness ([improvement backlog](IMPROVEMENT_BACKLOG.md)). Written 2026-10-30 against Granite 3.3 2B runs of the explain suite; file links point at this repository (`../../src`) or the memory service's `main` on GitHub, and code that is only on an unmerged branch is named with its branch. Run outputs that lived on the author's machine are not kept.

Trellis already stores almost everything a small model needs to explain its earlier answers, but that data rarely reaches Granite 3.3 2B in a form it can use. The run journal keeps each step's tool, arguments, status, output preview, source citations (name, date, link), timing, approvals and spend (explain.py: `src/trellis/harness/explain.py` (branch `claude/cool-maxwell-w5avo8-explain`, draft #59; not on main)). The model's problem is reading and copying that record, not finding it. In the saved runs, the 2B model answered **5 of 33** explain turns "correctly" on main and **1 of 23** with the `earlier_answers` lookup tool. **None** of the source, step or tool questions passed in either arm. Its main failure was **re-running turn 1's tools (21 and 14 turns)** instead of explaining them (P/out (local run output, not kept)). The outside evidence agrees on the fix. Small models fail at multi-turn tool choice (1–35% on BFCL multi-turn at 0.6–4B) ([TinyLLM](https://arxiv.org/pdf/2511.22138)). Citing works best when the evidence is already in context: **73.6 vs 26.7** citation recall for ChatGPT, inline vs after the fact ([ALCE](https://arxiv.org/html/2305.14627v2)). Models' own "why" stories are often unfaithful ([Turpin et al.](https://arxiv.org/abs/2305.04388)). So the harness should put the previous run's record in front of the model and let the model only phrase it. But the current test is too loose and too confounded to show whether any change helps. Fix the test first, then try four or five single changes against main, one at a time.

## The data exists, but the 2B model rarely sees it in usable form

The table below maps each kind of question to the data it needs, where Trellis stores that data, and whether turn 2 sees it today. Everything in it comes from the code audit, not from captured prompts.

| Question | Data needed | Stored? | Reaches the model on turn 2 today? |
|---|---|---|---|
| What steps / which tools | Tool name, args, order, status | Yes: journal `Step` | Memory off: **no, nothing from turn 1**. Memory on: replayed calls, but the old memory window (M0, 2,000-token / 20-message cap) drops them for "long" and "many" turns. `earlier_answers` gives steps in plain words and **hides tool names by design**. |
| Where from / cite sources | Source name, date, link per step | Yes: `Step.sources`, only for tools that return an artifact | Only through `earlier_answers`, if the model calls it |
| How long, was it approved | Seconds, approvals | Yes | Through `earlier_answers` |
| How sure | Evidence signals: source count, tool success, memory coverage | Partly: the record has evidence and coverage, but no confidence score | No derived score is shown |
| Why that conclusion | Which results the answer rests on | Only indirectly: the evidence, not the reasoning (the model does not think) | No |

The code paths behind this are short. A ReAct run checkpoints into its own run, so every turn starts an empty graph thread. With memory off, `runtime.history` stays empty ([react.py](../../src/trellis/harness/react.py), [agent.py](../../src/trellis/harness/agent.py)). With memory on, the harness writes the question, AI tool calls and tool results back to memory. The newer memory window (#62) returns the newest turn whole ([M62 window.py](https://github.com/amitmohapatra/agent-memory-service/blob/main/src/memory_service/modules/context/window.py)).

Inference: two more things then work against turn 1's results. First, `ClearToolUsesEdit` fires at **8,192 tokens** (half of a 16,384 window), and mean turn-2 input was **8,173 (main) and 10,678 (lookup)** tokens. So clearing likely ran on many turns, and it removes the oldest results first, which are turn 1's ([react.py](../../src/trellis/harness/react.py)). Second, the memory context block's `## Tools` section suggests tools for the *new* question ([context_bundle.py](https://github.com/amitmohapatra/agent-memory-service/blob/main/src/memory_service/domain/context_bundle.py)), which may push the model to re-run them. Neither effect was observed directly, because no prompts were captured.

## What frontier systems and frameworks do

Frontier APIs keep earlier tool calls by default and trim results, not calls. The OpenAI Agents SDK stores "all new items generated during the run" and replays the full history on each run unless you set a limit ([openai-agents-python](https://github.com/openai/openai-agents-python/blob/main/docs/sessions/index.md)). Anthropic's and LangChain's tool-result clearing share defaults: trigger at **100,000 tokens, keep the last 3**, and keep the call and its arguments. Only the result body is replaced with a placeholder, and stored history is left intact ([Anthropic context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing), [LangChain middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)). So a large model can usually answer "which tools did you use" straight from context.

Anthropic's guidance is to keep "lightweight identifiers (file paths, stored queries, web links)" rather than raw output ([Anthropic engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)). Manus says compression should be "restorable", for example dropping page text but keeping the URL ([Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)). No vendor documents how the ChatGPT or Claude apps build their "steps used" panels. The likely source is recorded tool-call events, not the model's own report, but that is an inference.

Anthropic's Citations API returns pointers "guaranteed" to be valid ([Claude docs](https://platform.claude.com/docs/en/build-with-claude/citations)). It works only with Claude, which fails the any-model rule.

What carries over to Trellis: our journal record is the "restorable identifier" layer that frontier systems recommend. The difference is the model. A 2B model with a 16k window is past these trimming triggers on turn 2 and cannot be relied on to choose a lookup tool. Small-model tool calling collapses on multi-turn tasks: Qwen3-1.7B scores about **17%** and xLAM-2-1B **8%** on BFCL multi-turn ([TinyLLM](https://arxiv.org/pdf/2511.22138); that paper's tables disagree by one row). Our 12/23 lookup-call rate with 1 correct answer is that collapse in action.

## Which techniques have numbers behind them

Most of these numbers come from other models and tasks. They show which way a change pushes, not what it will do to Granite 2B on our suite.

| Technique | Measured result (source) | Library or native? | Fit for us |
|---|---|---|---|
| Evidence in context, cite inline (not after the fact) | Citation recall 73.6 inline vs 26.7 after the fact, ChatGPT/ASQA ([ALCE](https://arxiv.org/html/2305.14627v2)) | Prompt pattern | Strong: supports putting the record in context |
| One or two examples in the prompt | Recall 69.3 with no example vs 73.6 with two ([ALCE](https://arxiv.org/html/2305.14627v2)) | Prompt pattern | Cheap |
| Build "why"/"steps" from the trace, not from the model's own account | Bias cut accuracy by up to 36%, and the stated reasoning never mentioned the bias ([Turpin](https://arxiv.org/abs/2305.04388)). Hint use was verbalized only 25–39% of the time ([Chen 2025](https://arxiv.org/pdf/2505.05410)) | Design rule | Strong: the record already is the trace |
| Confidence from evidence, not self-reported | Self-reported confidence has ECE > 0.377 ([Xiong](https://arxiv.org/pdf/2306.13063)). RLHF makes overconfidence worse ([ICLR 2025](https://arxiv.org/html/2410.09724v1)) | Design rule | Use the record's sources and tool status |
| Offer fewer tools | Selection accuracy 13.6% → 43.1% with retrieval, qwen-max ([RAG-MCP](https://arxiv.org/html/2505.03275v1)). Gains for 1.5–8B models ([Less-is-More](https://arxiv.org/html/2411.15399)) | `LLMToolSelectorMiddleware` (LangChain) | Medium: a 2B selector makes the same mistakes |
| Describe a repeated call in the harness's words instead of echoing it back | Repeat rate goes from 0.06 to 0.54 after a failure; a harness description removes 76% of that; "do not repeat" has no effect ([arXiv 2608.23651](https://arxiv.org/abs/2608.23651), abstract only) | Partly: `ToolCallLimitMiddleware` bounds calls; the message wording is ours | Strong match for our StallGuard errors |
| Grammar-constrained tool calls | Llama-3.2-3B 33.1% → 77.8% correct calls ([XGrammar-2](https://arxiv.org/pdf/2601.04426), the engine's own authors) | XGrammar, llama.cpp grammars | Fixes malformed calls and leaked tags, not wrong-tool choice |
| Quote first, then answer (trained) | Citation F1 73.6 for a 7B model vs 71.2 for ChatGPT ([FRONT](https://arxiv.org/html/2408.04568)) | Needs fine-tuning | Later, not now |

These do not work, or do not fit, so leave them out. Citing after the fact is the weakest option measured. Self-reported confidence is poorly calibrated. "Do not repeat" instructions do nothing. Summaries written by a small model are not documented to keep a tool log. The Citations API is Claude-only.

Two risks remain unmeasured. **Quantization** cost Llama-3.1-8B from 63.0% to 39.6% on BFCL at q4_K_M ([Less-is-More](https://arxiv.org/html/2411.15399)), and our GGUF's quantization level is not recorded. **Granite template misalignment** in llama.cpp was real enough to need a fix ([PR #16537](https://github.com/ggml-org/llama.cpp/pull/16537)).

## Fix four flaws in the test before re-measuring

The current numbers cannot rank changes. The table lists each flaw and the fix to make before the next run.

| Flaw | Why it matters | Fix |
|---|---|---|
| **Loose scoring** | `approved` passes on any "no record of approval" and `how_long` on any "N seconds". Main's 5 passes come from these checks. The source/step/tool kinds were 0/24 and 0/16 (explain.py checks: `benchmark/cost/explain.py` (branch `claude/cool-maxwell-w5avo8-explain`, draft #59; not on main)) | Score against the run's own record: exact tool names, source name and date, step count, seconds within a tolerance. Add a **made-up rate**: tools or sources claimed but absent from the journal. |
| **4.0 chat template on a 3.3 model** | The 4.0 template replays tool results as user-role `<tool_response>` turns, which matches answers echoing that tag. The 3.3 template never renders `tool_calls`, so replayed calls disappear (rbstack.sh (local run output, not kept)) | Run both templates as a setup arm and check that the llama.cpp log names the Granite format, not Hermes or chatml. Fix the template choice before testing anything else. |
| **No memory-off arm** | Both jobs passed `--memory`, so nothing shows what memory adds | Add main with memory off as a floor. |
| **No prompt capture** | Whether turn 1 was in context, cleared or cut is only inferred | Log each request body at the gateway (Bifrost) or llama.cpp, with the token count and whether clearing fired. |
| Confounded arms, small n | "After" changed memory (M0 → #62) **and** added the lookup tool. It was partial (25/36) and run once, with only 1 "many" conversation | Change one thing per arm, run every conversation, at least 3 repeats, and compare arms on the same conversations. Record the quantization level. |

## Shortlist to test against main, one change at a time

Each candidate below is one mechanism for one concern, uses library features where they exist, and works with any model size. Gains are **inferred** from the evidence above. Measure every candidate against main, with the same memory version and the fixed template, and **keep it only if** strict correctness rises across repeats, the made-up rate does not rise, and turn-2 tokens stay within budget. All metrics come from the fixed scorer: strict correctness by question kind, made-up rate, domain-tool re-call rate, stall/error rate, turn-2 input tokens and latency.

| # | Change | Evidence | Expected gain (inferred) | Cost | Watch in particular |
|---|---|---|---|---|---|
| 0 | Setup: correct template, memory-off floor, prompt capture, strict scorer | Flaws above | Trustworthy baseline | Test-harness work only | — |
| 1 | **Previous run's record in context on turn ≥2**: one numbered line per step with tool, source name and date, status and seconds. It replaces the `earlier_answers` tool, so there is still one mechanism. | Inline 73.6 vs after-the-fact 26.7. Small models choose tools poorly. 12/23 lookup calls but 1/23 right | Largest: removes the tool-choice step that fails today | About one line per step, only the last run (inference), versus about 2,400 tokens for a replayed filing result | Correct rate on steps, sources and tools; re-call rate |
| 2 | Show tool names in the record view (they are hidden today) | "Which tools" cannot be answered from a view that hides them | Unlocks the which_tools kind | Near zero | which_tools correct |
| 3 | One worked example (question about an earlier answer → answer copied from the record) | ALCE, +4.3 recall from examples | Small to medium; fewer leaked tags and placeholders | About 100 tokens | Leaked-tag and placeholder rate |
| 4 | Repeated-call message in the harness's own words ("already fetched: <result>"), with `ToolCallLimitMiddleware` as the backstop | 76% of repeats removed; "do not repeat" does nothing | Fewer StallGuard errors (13 and 10 turns errored) | Wording change plus library config | Stall/error rate |
| 5 | Confidence and "why" from the record (sources agreeing, tool status), phrased by the model | Self-reported confidence ECE about 0.3–0.38; stated reasoning unfaithful | Answers can be checked; fewer invented reasons | Small | Made-up rate on why/how-sure kinds |
| 6 | Narrow tools on turn ≥2 with `LLMToolSelectorMiddleware` | 13.6 → 43.1 selection accuracy (large model) | Uncertain with a 2B selector | One extra model call per turn | Re-call rate, latency |

Run 1 first: if it works, it makes 6 unnecessary. Also try turning off the memory `## Tools` hint on explain turns. That is a one-line ablation, not a new mechanism.

Candidate 1 bends the "fetch only when needed" rule, because it adds context on every later turn. Two things justify it. The record is small, and it replaces larger replayed results. And the alternative, letting a 2B model decide when to fetch, has already failed in our data. A deterministic rule that detects explain questions would also need validating first, so the simpler always-include version goes in the test.

## Conclusion

The real gap is between storing the record and getting the model to use it, not missing data. Frontier systems can rely on the model to look back through a long, lightly trimmed history. A 2B model at 16k tokens gets neither the long history nor the judgment to look back. That shifts the work from the model to the harness. The harness should decide what earlier evidence is shown and state facts (steps, sources, timing, confidence signals) from the journal. The model's job shrinks to phrasing what it is given, the one part small models do reliably.

This also makes "no made-up answers" checkable. Every claim can be compared with the journal. So the made-up rate deserves as much weight in the scorer as correctness. Without it, a change could raise the pass rate while still letting the model invent sources.
