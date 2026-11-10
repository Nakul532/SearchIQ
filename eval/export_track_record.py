"""Build public/static/track_record.json (the data behind /track-record) from the
eval results. Run after `report` for both sets:

    python eval/searchiq_eval.py report
    EVAL_SET=business python eval/searchiq_eval.py report
    python eval/export_track_record.py

Reads only committed eval files; makes no API calls.
"""

from __future__ import annotations

import datetime as dt
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "public", "static", "track_record.json")
REPO = "https://github.com/Nakul532/SearchIQ"

SETS = {
    "business": {
        "label": "Business research set", "dir": "results_business", "questions": "business_questions.jsonl",
        "groups": ["companies", "markets", "earnings", "acquisitions", "trick", "postcutoff"],
        "blurb": "20 analyst-style questions on companies, markets, earnings and acquisitions. Trick and † post-cutoff "
                 "questions overlap those groups. Graded on required elements only.",
    },
    "core": {
        "label": "General set", "dir": "results", "questions": "questions.jsonl",
        "groups": ["recent", "stable", "multistep", "trick"],
        "blurb": "40 questions: recent events, stable facts (the control: search shouldn't help), multi-step and "
                 "trick/unanswerable questions.",
    },
}
NAMES = {"companies": "Companies", "markets": "Markets", "earnings": "Earnings", "acquisitions": "Acquisitions",
         "trick": "Trick questions", "postcutoff": "† After the model's cutoff", "recent": "Recent events",
         "stable": "Stable facts (control)", "multistep": "Multi-step", "all": "Overall"}
ARMS = {"A": "Model alone (no search)", "B": "SearchIQ: search + cited answer", "C": "SearchIQ v2: B + claim check",
        "D": "SearchIQ v2: the check rewrites flagged answers (live today)"}


def jl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def js(path):
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return None


def build_set(key: str, cfg: dict) -> dict:
    d = os.path.join(HERE, cfg["dir"])
    summary = js(os.path.join(d, "summary.json")) or {}
    qs = {q["id"]: q for q in jl(os.path.join(HERE, cfg["questions"]))}
    judge = {(r["id"], r["arm"]): r for r in jl(os.path.join(d, "judge.jsonl"))}
    arm_c = {r["id"]: r for r in jl(os.path.join(d, "arm_c.jsonl"))}
    arm_d = {r["id"]: r for r in jl(os.path.join(d, "arm_d.jsonl"))}
    arms = [a for a in ("A", "B", "C", "D") if f"all|{a}" in summary]

    def row(g):
        return {"key": g, "name": NAMES[g],
                "arms": {a: {k: summary[f"{g}|{a}"][k] for k in
                             ("n", "correct_pct", "fully_correct", "grounded_pct", "trick_pass",
                              "unneeded_refusals", "latency_med", "cost_mean")}
                         for a in arms if f"{g}|{a}" in summary}}

    # Misses: every question where SearchIQ (B) wasn't fully correct, or where
    # search made it worse than the model alone. Shown with the judge's reason.
    misses = []
    for qid, q in qs.items():
        jb, ja = judge.get((qid, "B")), judge.get((qid, "A"))
        if not jb:
            continue
        jd = judge.get((qid, "D"))
        if jb["correctness"] < 2 or (ja and ja["correctness"] > jb["correctness"]) or (jd and jd["correctness"] < 2):
            jc = judge.get((qid, "C"))
            c = (arm_c.get(qid) or {}).get("check")
            misses.append({
                "id": qid, "group": NAMES.get(q["group"], q["group"]), "question": q["question"],
                "scores": {a: j["correctness"] for a, j in (("A", ja), ("B", jb), ("C", jc), ("D", jd)) if j},
                "rewritten": bool((arm_d.get(qid) or {}).get("revised")),
                "why": jb.get("rationale", ""),
                "check": None if not c else {"level": c["sides"]["B"]["confidence"]["level"],
                                             "premise_flagged": not c["premise"]["ok"]},
            })
    return {
        "key": key, "label": cfg["label"], "blurb": cfg["blurb"], "n_questions": len(qs), "arms": arms,
        "groups": [row(g) for g in cfg["groups"] if f"{g}|A" in summary],
        "overall": row("all") if "all|A" in summary else None,
        "misses": misses,
        "checker": js(os.path.join(d, "checker_diagnostics.json")),
        "calibration": js(os.path.join(d, "calibration.json")),
        "rewritten": sum(bool(r.get("revised")) for r in arm_d.values()),
    }


def main() -> None:
    agreement = js(os.path.join(HERE, "results", "agreement.json"))
    data = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "run_dates": "2026-11-10 (UTC)",
        "arms": ARMS,
        "models": {"answers": "GPT-OSS 20B (temperature 0)", "judge": "GPT-OSS 120B", "host": "Groq"},
        "locked": [{"what": "General set + rubric", "commit": "b4f6182"},
                   {"what": "Business set", "commit": "dac4e28"},
                   {"what": "Arm C definition", "commit": "e8736d8"},
                   {"what": "Arm D + calibration", "commit": "d86c89c"}],
        "links": {"method": f"{REPO}/blob/main/eval/README.md", "raw": f"{REPO}/tree/main/eval",
                  "rubric": f"{REPO}/blob/main/eval/rubric.md"},
        "agreement": agreement,
        "sets": [build_set(k, v) for k, v in SETS.items()],
    }
    # Calibration across both sets (the headline for the confidence labels).
    both = {}
    for st in data["sets"]:
        for arm, levels in (st.get("calibration") or {}).items():
            for lv, v in levels.items():
                t = both.setdefault(arm, {}).setdefault(lv, {"right": 0, "partly": 0, "n": 0})
                for k in t:
                    t[k] += v[k]
    data["calibration_all"] = both
    data["known_issues"] = [{
        "id": "K5", "status": "open", "fix": "planned as arm E (locked separately before it runs)",
        "title": "Forecasts treated as facts",
        "detail": ("Asked for Nvidia's closing price on 31 December 2035, SearchIQ correctly said it can't be known. "
                   "Its check then marked that as contradicted, because the search results contained price "
                   "forecasts. The rewrite replaced the correct answer with $619–$20,000 forecasts. Until arm E "
                   "ships, treat any forward-looking number in a rewritten answer as a forecast, not a fact."),
        "affects": "Questions about future prices, earnings or events (rewritten answers only)",
    }]
    data["calibration_note"] = (
        "Read honestly: HIGH means something (right about 9 times in 10). LOW does not yet mean \"probably wrong\". "
        "With claim checking alone (C), LOW answers were right 8 of 13. After rewrites (D), all 5 LOW answers were right: "
        "they're corrected trick answers whose correct claims (\"no human has walked on Mars\") just aren't stated in the "
        "sources. MEDIUM was never assigned. Treat LOW as \"verify this\", not as \"this is wrong\".")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
