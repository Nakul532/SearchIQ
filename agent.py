"""SearchIQ v2 agent: check the drafted answers against the search results.

One extra fast-model call per live query (~2–3K tokens) takes the question, the
numbered search results and both drafts, and returns:

  * premise    — whether the question assumes something the sources contradict
                 (the baseline eval showed retrieval made the model accept false
                 premises more often, so this is checked explicitly);
  * claims     — the key factual claims in each draft, each labelled
                 backed (with source numbers) / unsupported / contradicted;
  * disagreements — facts on which A and B differ, and which side the sources favour.

Confidence labels are computed here, deterministically, from the claim labels —
not asked of the model — so the same check always gives the same label.

Any failure (rate limit, bad JSON) returns None and the answers are shown
without a check: the check is additive and never blocks an answer.
"""

from __future__ import annotations

import json
import re

import providers

CHECK_SYSTEM = """You are SearchIQ's fact checker for business research. You receive a user's question, numbered web search results, and one or two draft answers (A and B). Check the drafts ONLY against the numbered search results — not against your own knowledge.

1. PREMISE. Does the question assume something false (an event that has not happened, a wrong fact, an impossible quantity)? Set premise.ok=false ONLY if the search results or basic logic show the assumption is false; explain in premise.note in one sentence. Otherwise ok=true and note="".
2. CLAIMS. For each draft, list its key factual claims (names, numbers, dates, outcomes) — at most 6 per draft, most important first, each restated in under 20 words. Label each:
   - "backed": a numbered result states it; give those numbers in "sources".
   - "contradicted": a numbered result says otherwise; give the numbers and say what it says in "note".
   - "unsupported": no result states it either way.
   A draft that only declines or says it doesn't know has no claims.
3. DISAGREEMENTS. Facts on which A and B state different things (not just different detail). For each: the topic, what A says, what B says, and which side the results support ("A", "B", "both" or "neither") with source numbers. Empty if they agree or only one draft is given.

Reply with ONLY this JSON, no prose:
{"premise":{"ok":true,"note":""},"claims":[{"side":"A","text":"...","status":"backed","sources":[1],"note":""}],"disagreements":[{"topic":"...","A":"...","B":"...","supported":"B","sources":[2]}]}"""

STATUSES = ("backed", "unsupported", "contradicted")
MAX_CLAIMS_PER_SIDE = 6


def _format_results(results: list[dict], limit: int) -> str:
    if not results:
        return "(no search results)"
    blocks = []
    for i, r in enumerate(results[:limit], 1):
        date = f" — published {r['published_date']}" if r.get("published_date") else ""
        blocks.append(f"[{i}] {r.get('title') or r.get('url')}{date}\n{r.get('url')}\n{(r.get('content') or '')[:1200]}")
    return "\n\n".join(blocks)


def check_user_prompt(question: str, results: list[dict], answers: dict[str, str], n_sources: int = 3) -> str:
    drafts = "\n\n".join(f"=== Draft {side} ===\n{text[:2500]}" for side, text in sorted(answers.items()))
    return (f"Question: {question}\n\nSearch results:\n\n{_format_results(results, n_sources)}\n\n{drafts}")


def _default_llm(system: str, user: str, max_tokens: int, effort: str) -> str:
    return providers.chat(providers.FAST_MODEL, [{"role": "user", "content": user}],
                          max_tokens=max_tokens, system=system, effort=effort)


def _ints(xs, n_sources: int) -> list[int]:
    out = []
    for x in xs if isinstance(xs, list) else []:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= n_sources and i not in out:
            out.append(i)
    return out


def _s(x, n: int = 240) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()[:n]


def confidence(claims: list[dict], premise_ok: bool) -> dict:
    """Deterministic label from claim labels: high / medium / low / none.

    ``premise_ok`` is False when the question's premise is flagged and the answer
    hasn't been rewritten to correct it (a rewritten answer passes True)."""
    n = len(claims)
    backed = sum(c["status"] == "backed" for c in claims)
    contra = sum(c["status"] == "contradicted" for c in claims)
    if not premise_ok:
        level = "low"
    elif n == 0:
        level = "none"
    elif contra:
        level = "low"
    elif backed / n >= 0.8:
        level = "high"
    elif backed / n >= 0.5:
        level = "medium"
    else:
        level = "low"
    return {"level": level, "backed": backed, "total": n, "contradicted": contra}


def normalize(raw: dict, sides: list[str], n_sources: int, revised: tuple = ()) -> dict:
    prem = raw.get("premise") if isinstance(raw.get("premise"), dict) else {}
    premise_ok = prem.get("ok") is not False
    claims_by_side: dict[str, list[dict]] = {s: [] for s in sides}
    for c in raw.get("claims") or []:
        if not isinstance(c, dict):
            continue
        side = str(c.get("side", "")).upper()
        status = str(c.get("status", "")).lower()
        if side not in claims_by_side or status not in STATUSES or not _s(c.get("text")):
            continue
        if len(claims_by_side[side]) >= MAX_CLAIMS_PER_SIDE:
            continue
        srcs = _ints(c.get("sources"), n_sources)
        if status == "backed" and not srcs:
            status = "unsupported"  # "backed" must point at a source
        claims_by_side[side].append({"text": _s(c.get("text")), "status": status,
                                     "sources": srcs, "note": _s(c.get("note"))})
    disagreements = []
    if len(sides) == 2:
        for d in raw.get("disagreements") or []:
            if not isinstance(d, dict) or not _s(d.get("topic")):
                continue
            sup = str(d.get("supported", "neither"))
            sup = sup.upper() if sup.upper() in ("A", "B") else (sup.lower() if sup.lower() in ("both", "neither") else "neither")
            disagreements.append({"topic": _s(d.get("topic"), 120), "A": _s(d.get("A")), "B": _s(d.get("B")),
                                  "supported": sup, "sources": _ints(d.get("sources"), n_sources)})
            if len(disagreements) >= 4:
                break
    return {
        "premise": {"ok": premise_ok, "note": "" if premise_ok else _s(prem.get("note"))},
        "sides": {s: {"claims": claims_by_side[s],
                      "confidence": confidence(claims_by_side[s], premise_ok or s in revised)}
                  for s in sides},
        "disagreements": disagreements,
    }


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("no JSON in check output")
    return json.loads(m.group(0))


def check_answers(question: str, results: list[dict], answers: dict[str, str],
                  llm=None, n_sources: int = 3, revised: tuple = ()) -> dict | None:
    """Run the check; returns the normalized check dict, or None on any failure.

    ``llm(system, user, max_tokens, effort) -> text`` defaults to the live app's
    Groq fast model; the eval passes its own (temperature 0) caller.
    """
    answers = {s: a for s, a in answers.items() if a and a.strip()}
    if not answers:
        return None
    llm = llm or _default_llm
    user = check_user_prompt(question, results, answers, n_sources)
    try:
        raw = parse(llm(CHECK_SYSTEM, user, 1800, "low"))
    except (providers.ProviderLimit, providers.ProviderError, ValueError, json.JSONDecodeError):
        return None
    return normalize(raw, sorted(answers), min(n_sources, len(results or [])), revised=revised)


# --------------------------------------------------------------------------- #
# Rewrite: make the warning act (arm D)
# --------------------------------------------------------------------------- #
REVISE_SYSTEM = """You are SearchIQ's editor for business research. You receive a question, numbered web search results, a draft answer, and a fact check of that draft. Rewrite the draft so an executive can rely on it:

1. If the check says the question's premise is false, open by correcting the premise in one plain sentence, and do not answer as if it were true. If the results give the real facts, state them with citations.
2. Remove every claim the check marked CONTRADICTED, or replace it with what the cited result actually says, citing [n].
3. Keep claims marked BACKED, with their citations.
4. Keep a claim marked UNSUPPORTED only if the answer needs it, and mark it "(not confirmed by the sources)".
5. Add no facts that are not in the numbered results. Keep the draft's style and roughly its length.

Reply with ONLY the rewritten answer."""


def needs_revision(check: dict | None, side: str) -> bool:
    """Rewrite when the premise is flagged or any claim is contradicted."""
    if not check or side not in check.get("sides", {}):
        return False
    return (not check["premise"]["ok"]) or any(
        c["status"] == "contradicted" for c in check["sides"][side]["claims"])


def revision_reasons(check: dict, side: str) -> list[str]:
    out = []
    if not check["premise"]["ok"]:
        out.append("Corrected the question's premise" + (f": {check['premise']['note']}" if check["premise"]["note"] else ""))
    for c in check["sides"][side]["claims"]:
        if c["status"] == "contradicted":
            out.append(f"Removed or corrected a claim the sources contradict: {c['text']}")
    return out


def _check_for_editor(check: dict, side: str) -> str:
    lines = [f"PREMISE: {'OK' if check['premise']['ok'] else 'FALSE: ' + check['premise']['note']}"]
    for c in check["sides"][side]["claims"]:
        refs = ",".join(str(i) for i in c["sources"])
        lines.append(f"- {c['status'].upper()}{' [' + refs + ']' if refs else ''}: {c['text']}"
                     + (f" ({c['note']})" if c["note"] else ""))
    return "\n".join(lines)


def revise_answer(question: str, results: list[dict], answer: str, check: dict, side: str,
                  llm=None, n_sources: int = 3) -> str | None:
    """Rewrite one draft per the check; None on failure (keep the original)."""
    llm = llm or _default_llm
    user = (f"Question: {question}\n\nSearch results:\n\n{_format_results(results, n_sources)}\n\n"
            f"=== Draft ===\n{answer[:2500]}\n\n=== Fact check of the draft ===\n{_check_for_editor(check, side)}")
    try:
        text = (llm(REVISE_SYSTEM, user, 1600, "low") or "").strip()
    except (providers.ProviderLimit, providers.ProviderError):
        return None
    return text or None


def check_and_revise(question: str, results: list[dict], answers: dict[str, str],
                     llm=None, n_sources: int = 3) -> tuple[dict | None, dict[str, dict]]:
    """Check, rewrite flagged drafts, re-check the rewrites.

    Returns (check, revisions). ``check`` describes the answers as finally shown
    (rewritten sides re-checked, their confidence no longer capped by the premise
    flag because the rewrite addresses it). ``revisions`` maps side →
    {answer, original, reasons}."""
    check = check_answers(question, results, answers, llm=llm, n_sources=n_sources)
    if not check:
        return None, {}
    return revise_flagged(question, results, answers, check, llm=llm, n_sources=n_sources)


def revise_flagged(question: str, results: list[dict], answers: dict[str, str], check: dict,
                   llm=None, n_sources: int = 3) -> tuple[dict, dict[str, dict]]:
    """The rewrite half of check_and_revise, given the first check."""
    import copy
    check = copy.deepcopy(check)
    revisions = {}
    for side in sorted(check["sides"]):
        if needs_revision(check, side):
            new = revise_answer(question, results, answers[side], check, side, llm=llm, n_sources=n_sources)
            if new:
                revisions[side] = {"answer": new, "original": answers[side],
                                   "reasons": revision_reasons(check, side)}
    if revisions:
        recheck = check_answers(question, results, {s: r["answer"] for s, r in revisions.items()},
                                llm=llm, n_sources=n_sources, revised=tuple(revisions))
        if recheck:
            for s in revisions:
                check["sides"][s] = recheck["sides"][s]
        else:  # re-check failed: show the rewrite with an honest "none" label
            for s in revisions:
                check["sides"][s] = {"claims": [], "confidence": confidence([], True)}
        check["revised"] = sorted(revisions)
    return check, revisions


def render_note(check: dict, side: str, reasons: list[str] | None = None) -> str:
    """The check as the user sees it under one answer, in markdown (also what
    eval arm C appends to the answer before judging)."""
    s = check["sides"].get(side)
    if not s:
        return ""
    conf = s["confidence"]
    lines = []
    if reasons:
        lines.append("✎ **Rewritten by SearchIQ's check:** " + "; ".join(reasons))
    elif not check["premise"]["ok"]:
        lines.append(f"⚠ **Check the question's premise:** {check['premise']['note']}")
    label = {"high": "High", "medium": "Medium", "low": "Low", "none": "No checkable claims"}[conf["level"]]
    head = f"**SearchIQ check — confidence: {label}**"
    if conf["total"]:
        head += f" ({conf['backed']}/{conf['total']} claims backed by the sources)"
    lines.append(head)
    for c in s["claims"]:
        refs = "".join(f"[{i}]" for i in c["sources"])
        if c["status"] == "backed":
            lines.append(f"- ✓ Backed {refs}: {c['text']}")
        elif c["status"] == "contradicted":
            lines.append(f"- ✗ Contradicted {refs}: {c['text']}" + (f" — {c['note']}" if c["note"] else ""))
        else:
            lines.append(f"- ⚠ Double-check (not in the sources): {c['text']}")
    return "\n".join(lines)
