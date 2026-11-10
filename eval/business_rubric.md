# SearchIQ business-research eval (set 2): locked addendum

Set 2 is `business_questions.jsonl`: 20 questions in 4 groups of 5 (companies, markets, earnings, acquisitions). The user approved it on 2026-11-09, and it was committed before any answer was generated.

It uses **the same arms, models, prompts, judge and criteria** as `rubric.md`, with three differences:

- **Trick questions** are marked by `"expected": "decline_or_correct"` (K5, E5, A5) rather than by a group of their own. They pass when the answer declines or corrects the premise.
- **Reporting:** per group (companies, markets, earnings, acquisitions), plus a separate trick-question line and an overall figure.
- **Grading tolerances:** each `grading_notes` field defines them. For example, A1 is graded on the completion date only, because its deal value depends on how it's counted. E1–E3 accept ±2% as partial credit.

Post-training-cutoff items, which happened after about June 2024: E1, E2, A2, A3, A4.

Budget: 20 Tavily credits plus about 40 answer calls and 40 judge calls, under the same 90K tokens per model per day cap.
