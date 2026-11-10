# SearchIQ business-research eval (set 2): locked addendum

Set 2 is `business_questions.jsonl`: 20 questions in 4 groups of 5 (companies, markets, earnings, acquisitions). The user approved it on 2026-11-09, revised it the same day, and it was committed before any answer was generated.

It uses **the same arms, models, prompts, judge and criteria** as `rubric.md`, with these differences:

- **Required vs optional elements.** Each question lists `required` elements, which are the only ones correctness is graded on, and `optional` context, which never raises or lowers correctness. The rule is written into each question's `grading_notes`, which the locked judge prompt already passes to the judge, so the judge prompt itself is unchanged:
  - 2 = all required elements are correct
  - 1 = some, but not all
  - 0 = none, or, on a trick question, accepting the false premise

  A question with a single required element can only score 0 or 2. Numeric tolerances are written into the element itself, e.g. "accept $130–131 billion". Groundedness still covers every factual claim, including optional ones.
- **Trick questions** are marked by `"expected": "decline_or_correct"` (K5, E5, A5). They pass when the answer declines or corrects the premise.
- **Post-cutoff (†) group.** Items with `"post_cutoff": true` (E1, E2, A2, A3, A4) happened after the answer model's training cutoff of about June 2024. They are reported as their own group alongside the overall results.
- **Reporting:** per group (companies, markets, earnings, acquisitions), plus trick-question handling, the † group and an overall figure.
- **Sources** are primary where one exists: company newsroom or investor-relations pages, SEC filings and IDC. Wikipedia is used only for founding facts and headquarters.

Budget: 20 Tavily credits plus about 40 answer calls and 40 judge calls, under the same 90K tokens per model per day cap.
