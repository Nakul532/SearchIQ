# Arm D: the check acts (locked before running)

**What it is.** Arm D continues arm C's pipeline from C's own check of B's answer.
- **When the check flags something** (a false premise, or any claim labelled *contradicted*; `agent.needs_revision`), `agent.revise_flagged()` makes two more calls:
  1. **Rewrite** (`agent.REVISE_SYSTEM`): the rewrite corrects the premise up front, removes or corrects contradicted claims from the sources, keeps backed claims, marks any kept unsupported claim "(not confirmed by the sources)", and adds no new facts.
  2. **Re-check:** the rewrite is checked again.
  - Arm D's text is the rewrite plus its check note, which starts "✎ Rewritten by SearchIQ's check: …". This is exactly what the live app shows.
- **When nothing is flagged,** D's text is identical to C's, so D **inherits C's verdict**. It isn't re-judged, so unchanged answers add no judge noise.

**Settings.** Same model as C (`openai/gpt-oss-20b`, temperature 0), same cached search results, same locked judge and rubric, graded like B and C.

**Confidence on rewritten answers.** A rewritten answer's label comes from the re-check, with the premise cap lifted, because the rewrite now corrects the premise. Answers that weren't rewritten keep C's label.

**Calibration** (reported for C and D; this is the core feature): for each label (HIGH / MEDIUM / LOW / NONE = no checkable claims), the answers with that label as shown, and how many the judge scored fully right (correctness 2): "right X of N".
- A useful label means **HIGH is right much more often than LOW**.

**Expectation, written before the run.**
- **Trick questions:** correctness should rise, because the premise is now corrected in the answer text.
- **Elsewhere:** few answers are rewritten. A rewrite can also break a right answer that had a false-alarm flag, and that will be reported.
