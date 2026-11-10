# Arm C: SearchIQ v2 (locked before running)

**What it is.** Arm C is what a v2 user sees under answer B:
1. The **arm-B answer text, reused as-is.** It isn't regenerated, so the only difference from B is the check.
2. Then **`agent.render_note()`**: the confidence label, the per-claim labels (✓ Backed [n] / ⚠ Double-check / ✗ Contradicted [n]) and any false-premise warning.

**How the check runs.** The check is `agent.check_answers()` exactly as deployed (prompt `agent.CHECK_SYSTEM`), using `openai/gpt-oss-20b` at temperature 0 (the live app uses 0.4), on the same cached Tavily results as B. It checks B alone. Eval arm A has no retrieval, unlike the live app's A, so the **disagreement view is not evaluated**. If the check fails twice, C is B's answer unchanged.

**Scoring.** The locked judge prompt and rubric are unchanged. C is graded like B: with sources and the same reference snippets. That gives the same correctness / groundedness / declined numbers per group as A and B.

**Exploratory checker diagnostics.** These are not in the locked rubric, so they're reported separately. "Flagged" = confidence not High, or premise flagged.
- Of B answers the judge scored below 2: how many C flagged. This is the catch rate.
- Of B answers the judge scored 2: how many C flagged. This is the false-alarm rate.
- Premise flags on trick questions vs on all other questions.

**Expectation, written before the run.** The check doesn't change B's text, so correctness can only move when a warning changes the judge's reading: mostly on trick questions, where a premise warning may count as correcting the premise. Groundedness may move either way.
