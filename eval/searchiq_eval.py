"""SearchIQ evaluation: no retrieval (A) vs SearchIQ + Tavily (B), judged by gpt-oss-120b.

Runs locally against the providers directly — never through the live site, so
the public daily caps aren't touched. Every step is resumable: results are
appended to eval/results/*.jsonl after each call, and finished items are skipped.

    python eval/searchiq_eval.py answer     # 40 Tavily searches + 80 answers
    python eval/searchiq_eval.py judge      # 80 judge verdicts
    python eval/searchiq_eval.py sample     # writes results/human_scores.csv (10 random)
    python eval/searchiq_eval.py agree      # judge-vs-human agreement
    python eval/searchiq_eval.py report     # eval/README.md results + eval/chart.svg

Needs GROQ_API_KEY and TAVILY_API_KEY (from the repo's .env).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import random
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import providers  # noqa: E402
import search_agent as sa  # noqa: E402

RESULTS = os.environ.get("EVAL_RESULTS_DIR") or os.path.join(HERE, "results")  # override for dry runs
QUESTIONS = os.path.join(HERE, "questions.jsonl")
ANSWERS = os.path.join(RESULTS, "answers.jsonl")
SEARCHES = os.path.join(RESULTS, "searches.jsonl")
JUDGE = os.path.join(RESULTS, "judge.jsonl")
BUDGET = os.path.join(RESULTS, "budget.json")
HUMAN = os.path.join(RESULTS, "human_scores.csv")

ANSWER_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-120b"
DAILY_TOKEN_CAP = int(os.environ.get("EVAL_DAILY_TOKEN_CAP", 90_000))  # per model per UTC day
PRICES = {"openai/gpt-oss-20b": (0.075, 0.30), "openai/gpt-oss-120b": (0.15, 0.60)}
GROUPS = ["recent", "stable", "multistep", "trick"]
GROUP_NAMES = {"recent": "Recent events", "stable": "Stable facts (control)",
               "multistep": "Multi-step", "trick": "Trick / unanswerable"}

# --------------------------------------------------------------------------- #
# Prompts (LOCKED — part of the committed rubric)
# --------------------------------------------------------------------------- #
_B_SYSTEM = sa.build_system_prompt([], style=sa.STYLE_A)
_GROUNDING_B = _B_SYSTEM[_B_SYSTEM.index("GROUNDING REQUIREMENT:"):_B_SYSTEM.index("\n\n" + sa.STYLE_A)]
_GROUNDING_A = ("KNOWLEDGE: You have no web search for this question. Answer from your own "
                "knowledge. If you are not sure, or the answer cannot be known, say so plainly "
                "instead of guessing.")
_A_SYSTEM = (_B_SYSTEM.replace("Given a user's search query and fresh web search results, ",
                               "Given a user's search query, ")
             .replace(_GROUNDING_B, _GROUNDING_A))
assert _A_SYSTEM != _B_SYSTEM and "GROUNDING REQUIREMENT" not in _A_SYSTEM
SYSTEM = {"A": _A_SYSTEM, "B": _B_SYSTEM}

JUDGE_SYSTEM = """You are a strict, fair grader for a search-answer evaluation. Grade ONE answer.
Return ONLY a JSON object, no prose, with exactly these keys:
{"correctness": 0|1|2, "groundedness": 0|1|2, "declined": true|false, "rationale": "<one or two sentences>"}

correctness — judged against the ANSWER KEY and GRADING NOTES:
  2 = fully correct; 1 = partly correct, or correct but with a wrong extra claim; 0 = wrong/missing.
  For TRICK questions, "correct" means the expected behaviour: declining, or correcting the
  false premise. Accepting the false premise or inventing an answer = 0.
groundedness:
  If the answer HAD SOURCES: are its factual claims supported by the cited [n] sources in the
  REFERENCE SNIPPETS? 2 = all supported and cited; 1 = some unsupported or uncited; 0 = mostly
  unsupported or no citations. A pure decline with no factual claims = 2.
  If the answer had NO SOURCES: are its factual claims true (use the answer key, the reference
  snippets, and well-established knowledge)? 2 = all true; 1 = some false; 0 = mostly false.
declined: true if the answer does NOT give a direct answer — it refuses, says it doesn't know /
  can't be known, or rejects the question's premise. Otherwise false.
The reference snippets are evidence for YOU; the answer may or may not have seen them."""


def judge_user(q: dict, arm: str, answer: str, results: list[dict]) -> str:
    refs = "\n\n".join(f"[{i}] {r.get('title')} — {r.get('url')}\n{r.get('content')}"
                       for i, r in enumerate(results, 1)) or "(none)"
    return (f"QUESTION GROUP: {q['group']}{' (TRICK question)' if q['group'] == 'trick' else ''}\n"
            f"QUESTION: {q['question']}\n"
            f"ANSWER KEY: {q['answer_key']}\nGRADING NOTES: {q['grading_notes']}\n"
            f"ANSWER HAD SOURCES: {'yes (numbered [1]-[3], same as the reference snippets)' if arm == 'B' else 'no'}\n\n"
            f"REFERENCE SNIPPETS:\n{refs}\n\nANSWER TO GRADE:\n{answer}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def questions() -> list[dict]:
    return load_jsonl(QUESTIONS)


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def _budget() -> dict:
    try:
        with open(BUDGET) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def spend(model: str, tokens: int) -> None:
    b = _budget()
    key = f"{_today()}|{model}"
    b[key] = b.get(key, 0) + tokens
    with open(BUDGET, "w") as f:
        json.dump(b, f, indent=1, sort_keys=True)


def room(model: str, need: int) -> bool:
    return _budget().get(f"{_today()}|{model}", 0) + need <= DAILY_TOKEN_CAP


def call(model: str, system: str, user: str, max_tokens: int, effort: str) -> tuple[str, dict, float]:
    """Groq call at temperature 0 with patient retry on per-minute limits."""
    for attempt in range(6):
        t0 = time.perf_counter()
        try:
            text, usage = providers.chat_full(model, [{"role": "user", "content": user}],
                                              max_tokens=max_tokens, system=system,
                                              effort=effort, temperature=0)
            secs = time.perf_counter() - t0
            spend(model, usage.get("total_tokens", 0))
            return text, usage, secs
        except providers.ProviderLimit:
            wait = 20 * (attempt + 1)
            print(f"   rate-limited on {model}; waiting {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"Still rate-limited on {model} — likely the daily limit. Rerun later to resume.")


def cost(model: str, usage: dict) -> float:
    pin, pout = PRICES[model]
    return usage.get("prompt_tokens", 0) / 1e6 * pin + usage.get("completion_tokens", 0) / 1e6 * pout


# --------------------------------------------------------------------------- #
# Step 1: answers
# --------------------------------------------------------------------------- #
def cmd_answer() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    searches = {r["id"]: r for r in load_jsonl(SEARCHES)}
    done = {(r["id"], r["arm"]) for r in load_jsonl(ANSWERS)}
    run_date = _today()
    for q in questions():
        # One Tavily search per question, cached forever (exactly 40 credits).
        if q["id"] not in searches:
            t0 = time.perf_counter()
            results = providers.search(q["question"], max_results=sa.MAX_SOURCES)
            row = {"id": q["id"], "results": results, "search_secs": round(time.perf_counter() - t0, 3),
                   "searched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
            append_jsonl(SEARCHES, row)
            searches[q["id"]] = row
        for arm in ("A", "B"):
            if (q["id"], arm) in done:
                continue
            if not room(ANSWER_MODEL, 4000):
                print(f"Daily token cap for {ANSWER_MODEL} reached ({DAILY_TOKEN_CAP}). Rerun tomorrow to resume.")
                return
            s = searches[q["id"]]
            body = (f"Web search results:\n\n{sa._format_results(s['results'])}\n\n" if arm == "B" else "")
            user = f"Today's date: {run_date}\n\n{body}Question: {q['question']}"
            text, usage, secs = call(ANSWER_MODEL, SYSTEM[arm], user, max_tokens=1400, effort="low")
            latency = secs + (s["search_secs"] if arm == "B" else 0.0)
            append_jsonl(ANSWERS, {
                "id": q["id"], "group": q["group"], "arm": arm, "model": ANSWER_MODEL,
                "answer": text, "sources": sa.sources_from_results(s["results"]) if arm == "B" else [],
                "usage": usage, "latency_s": round(latency, 3), "cost_usd": round(cost(ANSWER_MODEL, usage), 7),
                "tavily_credits": 1 if arm == "B" else 0, "run_date": run_date,
            })
            print(f"{q['id']}/{arm}: {secs:.1f}s, {usage.get('total_tokens')} tok", flush=True)
    print("All 80 answers done.")


# --------------------------------------------------------------------------- #
# Step 2: judge
# --------------------------------------------------------------------------- #
def parse_verdict(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON")
    v = json.loads(m.group(0))
    assert v["correctness"] in (0, 1, 2) and v["groundedness"] in (0, 1, 2)
    assert isinstance(v["declined"], bool)
    return v


def cmd_judge() -> None:
    qs = {q["id"]: q for q in questions()}
    searches = {r["id"]: r for r in load_jsonl(SEARCHES)}
    done = {(r["id"], r["arm"]) for r in load_jsonl(JUDGE)}
    answers = load_jsonl(ANSWERS)
    if len(answers) < 80:
        print(f"Only {len(answers)}/80 answers so far — judging what exists.")
    for a in answers:
        if (a["id"], a["arm"]) in done:
            continue
        if not room(JUDGE_MODEL, 4000):
            print(f"Daily token cap for {JUDGE_MODEL} reached ({DAILY_TOKEN_CAP}). Rerun tomorrow to resume.")
            return
        q = qs[a["id"]]
        user = judge_user(q, a["arm"], a["answer"], searches[a["id"]]["results"])
        for attempt in range(2):
            text, usage, _ = call(JUDGE_MODEL, JUDGE_SYSTEM, user, max_tokens=1500, effort="medium")
            try:
                v = parse_verdict(text)
                break
            except (ValueError, KeyError, AssertionError, json.JSONDecodeError):
                v = None
        if v is None:
            print(f"{a['id']}/{a['arm']}: unparseable verdict, will retry next run")
            continue
        append_jsonl(JUDGE, {"id": a["id"], "arm": a["arm"], **v, "judge_model": JUDGE_MODEL,
                             "usage": usage})
        print(f"{a['id']}/{a['arm']}: C{v['correctness']} G{v['groundedness']} D{int(v['declined'])}", flush=True)
    print("Judging done.")


# --------------------------------------------------------------------------- #
# Step 3: human sample + agreement
# --------------------------------------------------------------------------- #
def cmd_sample() -> None:
    answers = sorted(load_jsonl(ANSWERS), key=lambda r: (r["id"], r["arm"]))
    assert len(answers) == 80, "run `answer` first"
    qs = {q["id"]: q for q in questions()}
    pick = random.Random(42).sample(answers, 10)
    with open(HUMAN, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "arm", "group", "question", "answer_key", "grading_notes", "answer", "sources",
                    "your_correctness_0_1_2", "your_groundedness_0_1_2", "your_declined_true_false"])
        for a in pick:
            q = qs[a["id"]]
            src = " | ".join(s["url"] for s in a["sources"]) or "(no sources — arm A)"
            w.writerow([a["id"], a["arm"], q["group"], q["question"], q["answer_key"], q["grading_notes"],
                        a["answer"], src, "", "", ""])
    print(f"Wrote {HUMAN} with 10 answers. Judge scores are not shown in it.")


def agreement() -> dict | None:
    if not os.path.exists(HUMAN):
        return None
    judge = {(r["id"], r["arm"]): r for r in load_jsonl(JUDGE)}
    rows, per = [], {"correctness": [0, 0], "groundedness": [0, 0], "declined": [0, 0]}
    with open(HUMAN, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c, g, d = (r["your_correctness_0_1_2"].strip(), r["your_groundedness_0_1_2"].strip(),
                       r["your_declined_true_false"].strip().lower())
            if not (c and g and d):
                return None  # not filled in yet
            j = judge[(r["id"], r["arm"])]
            human = {"correctness": int(c), "groundedness": int(g), "declined": d in ("true", "t", "yes", "1")}
            for k in per:
                per[k][0] += int(human[k] == j[k])
                per[k][1] += 1
            rows.append({"id": r["id"], "arm": r["arm"], "human": human,
                         "judge": {k: j[k] for k in per}})
    total = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    return {"per_criterion": {k: v[0] / v[1] for k, v in per.items()},
            "overall": total[0] / total[1], "n_answers": len(rows), "rows": rows}


def cmd_agree() -> None:
    a = agreement()
    if not a:
        print(f"Fill in the three 'your_*' columns in {HUMAN} first.")
        return
    print(json.dumps({k: v for k, v in a.items() if k != "rows"}, indent=1))


# --------------------------------------------------------------------------- #
# Step 4: report
# --------------------------------------------------------------------------- #
def summarize() -> dict:
    answers = {(r["id"], r["arm"]): r for r in load_jsonl(ANSWERS)}
    judge = {(r["id"], r["arm"]): r for r in load_jsonl(JUDGE)}
    out = {}
    for g in GROUPS + ["all"]:
        for arm in ("A", "B"):
            keys = [k for k, a in answers.items() if k[1] == arm and (g == "all" or a["group"] == g)]
            js = [judge[k] for k in keys if k in judge]
            ans = [answers[k] for k in keys]
            n = len(js)
            trick = [judge[k] for k in keys if k in judge and answers[k]["group"] == "trick"]
            non = [judge[k] for k in keys if k in judge and answers[k]["group"] != "trick"]
            out[(g, arm)] = {
                "n": n,
                "correct_pct": 100 * sum(j["correctness"] for j in js) / (2 * n) if n else None,
                "fully_correct": sum(j["correctness"] == 2 for j in js),
                "grounded_pct": 100 * sum(j["groundedness"] for j in js) / (2 * n) if n else None,
                "trick_pass": (sum(j["declined"] for j in trick), len(trick)),
                "unneeded_refusals": (sum(j["declined"] for j in non), len(non)),
                "latency_med": statistics.median(a["latency_s"] for a in ans) if ans else None,
                "cost_mean": statistics.mean(a["cost_usd"] for a in ans) if ans else None,
                "tavily": sum(a["tavily_credits"] for a in ans),
            }
    return out


def chart_svg(s: dict) -> str:
    """Grouped bar chart: correctness % per group, A vs B (no dependencies)."""
    W, H, L, B_, T = 640, 300, 56, 56, 34
    plot_w, plot_h = W - L - 20, H - B_ - T
    groups = GROUPS + ["all"]
    names = {"recent": "Recent", "stable": "Stable (control)", "multistep": "Multi-step",
             "trick": "Trick", "all": "Overall"}
    col = {"A": "#94a3b8", "B": "#4f46e5"}
    slot = plot_w / len(groups)
    bw = slot * 0.32
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" font-size="12">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
             f'<text x="{L}" y="20" font-size="14" font-weight="700" fill="#111827">Correctness vs answer key (% of points), by question group</text>']
    for p in (0, 25, 50, 75, 100):
        y = T + plot_h * (1 - p / 100)
        parts.append(f'<line x1="{L}" x2="{W-20}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{L-8}" y="{y+4:.1f}" text-anchor="end" fill="#6b7280">{p}%</text>')
    for i, g in enumerate(groups):
        x0 = L + i * slot + slot * 0.16
        for j, arm in enumerate(("A", "B")):
            v = s[(g, arm)]["correct_pct"] or 0
            h = plot_h * v / 100
            x = x0 + j * (bw + 4)
            parts.append(f'<rect x="{x:.1f}" y="{T+plot_h-h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{col[arm]}" rx="2"/>')
            parts.append(f'<text x="{x+bw/2:.1f}" y="{T+plot_h-h-4:.1f}" text-anchor="middle" fill="#111827">{v:.0f}</text>')
        parts.append(f'<text x="{L+i*slot+slot/2:.1f}" y="{T+plot_h+18:.1f}" text-anchor="middle" fill="#374151">{names[g]}</text>')
    lx = L
    for arm, label in (("A", "A: no retrieval"), ("B", "B: SearchIQ + Tavily")):
        parts.append(f'<rect x="{lx}" y="{H-18}" width="12" height="12" fill="{col[arm]}" rx="2"/>')
        parts.append(f'<text x="{lx+18}" y="{H-8}" fill="#374151">{label}</text>')
        lx += 170
    parts.append("</svg>")
    return "\n".join(parts)


def cmd_report() -> None:
    s = summarize()
    chart_dir = os.environ.get("EVAL_RESULTS_DIR") or HERE
    with open(os.path.join(chart_dir, "chart.svg"), "w") as f:
        f.write(chart_svg(s))
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump({f"{g}|{a}": v for (g, a), v in s.items()}, f, indent=1)
    print("Wrote eval/chart.svg and results/summary.json")
    print(json.dumps({f"{g}|{a}": v for (g, a), v in s.items()}, indent=1))
    ag = agreement()
    if ag:
        print("agreement:", json.dumps({k: v for k, v in ag.items() if k != "rows"}))


if __name__ == "__main__":
    cmds = {"answer": cmd_answer, "judge": cmd_judge, "sample": cmd_sample,
            "agree": cmd_agree, "report": cmd_report}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        print(__doc__)
        raise SystemExit(2)
    if sys.argv[1] in ("answer", "judge") and not (os.environ.get("GROQ_API_KEY") and
                                                    (os.environ.get("TAVILY_API_KEY") or sys.argv[1] == "judge")):
        if not providers.fake_mode():
            raise SystemExit("Set GROQ_API_KEY and TAVILY_API_KEY in .env first.")
    cmds[sys.argv[1]]()
