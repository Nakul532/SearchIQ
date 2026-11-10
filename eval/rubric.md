# SearchIQ eval: locked rubric

This file, `questions.jsonl` and the prompts in `searchiq_eval.py` were committed **before** any answer was generated. None of them may change after the run starts.

## Setup

| | Arm A: no retrieval | Arm B: SearchIQ + Tavily |
|---|---|---|
| Answer model | `openai/gpt-oss-20b` (Groq) | same |
| Settings | temperature 0, reasoning effort low, max 1,400 output tokens | same |
| System prompt | SearchIQ's prompt + concise style (STYLE_A); the grounding paragraph is replaced with "answer from your own knowledge; say so if unsure or unknowable" | SearchIQ's prompt + STYLE_A, unchanged |
| User message | `Today's date: <run date>` + question | `Today's date: <run date>` + top-3 Tavily results (basic search, 1 credit) + question |

Temperature 0 makes the results repeatable. **The live app uses 0.4.**

## Judge

`openai/gpt-oss-120b` on Groq, temperature 0, reasoning effort medium. It makes one call per answer (80 in total) and returns a JSON verdict.

The judge sees:
- the question and its group
- the answer key and grading notes
- the answer, and whether it had sources
- reference snippets: the same Tavily results arm B saw. These are given for **both** arms, so the judge can check arm A's claims too.

## Criteria

1. **Correctness** against the answer key:
   - 2 = fully correct
   - 1 = partly correct, or correct but with a wrong extra claim
   - 0 = wrong or missing

   For trick questions, "correct" means the expected behaviour: declining, or correcting the false premise.
2. **Groundedness:**
   - **Arm B:** every factual claim is supported by a cited source in the provided results. 2 = all, 1 = some, 0 = none or no citations. An answer that only declines, with no factual claims, scores 2.
   - **Arm A:** there are no sources, so this scores whether its factual claims are true, using the answer key, the reference snippets and well-established knowledge. 2 = all true, 1 = some false, 0 = mostly false.
3. **Declined** (true/false): the answer does not give a direct answer. It refuses, says the answer is unknown or unknowable, or rejects the question's premise.
   - Trick questions: **pass** = declined is true.
   - Other questions: **unneeded refusal** = declined is true.
4. **Latency:** wall-clock seconds for the answer. Arm B includes the Tavily search.
5. **Estimated cost:** actual prompt and completion tokens × Groq list price for gpt-oss-20b ($0.075 / $0.30 per 1M tokens, checked 2026-11-09). Arm B also uses 1 Tavily credit, which is reported as a count, not in dollars.

## Reporting

- Each metric is reported **per group** (recent, stable, multi-step, trick) and overall.
- Stable facts are the **control**: retrieval should neither help nor hurt much there.
- Correctness is shown as the % of possible points (score / 2).

## Human agreement

10 of the 80 answers are drawn at random (`random.Random(42)`) and scored by hand on the same criteria, without seeing the judge's scores. Agreement is the % of exact matches for correctness, groundedness and declined, plus the overall % across the 30 labels.

## Budget

- 40 Tavily credits in total, cached so a rerun never searches again.
- The runner stops at 90,000 Groq tokens per model per UTC day (free tier is about 200K) and resumes on the next run. The live site shares the same key.
