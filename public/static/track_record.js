"use strict";

const ARM_COLORS = { A: "var(--tr-a)", B: "var(--tr-b)", C: "var(--tr-c)" };
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v) => (v == null ? "–" : `${Math.round(v)}%`);
const SCORE = ["✗ wrong", "◐ partly right", "✓ right"];

function card(label, value, sub) {
  return `<div class="tr-card"><div class="tr-card-val">${value}</div><div class="tr-card-label">${esc(label)}</div>${sub ? `<div class="tr-card-sub">${sub}</div>` : ""}</div>`;
}

function summaryCards(data) {
  const out = [];
  for (const s of data.sets) {
    if (!s.overall) continue;
    const o = s.overall.arms;
    const best = o.C ? "C" : "B";
    out.push(card(`${s.label}: SearchIQ correct`, pct(o[best].correct_pct),
      `vs ${pct(o.A.correct_pct)} for the model alone · ${s.n_questions} questions`));
  }
  const business = data.sets.find((s) => s.key === "business");
  const pc = business && business.groups.find((g) => g.key === "postcutoff");
  if (pc) out.push(card("Facts after the model's cutoff", pct(pc.arms.B.correct_pct),
    `vs ${pct(pc.arms.A.correct_pct)} without search`));
  const core = data.sets.find((s) => s.key === "core");
  const trick = core && core.groups.find((g) => g.key === "trick");
  if (trick) {
    const arm = trick.arms.C ? "C" : "B";
    out.push(card("Trick questions handled", pct(trick.arms[arm].correct_pct),
      `vs ${pct(trick.arms.A.correct_pct)} without search: <b>search can mislead here</b>`));
  }
  return out.join("");
}

function bars(arms, armsShown) {
  return armsShown.filter((a) => arms[a]).map((a) => {
    const v = arms[a].correct_pct || 0;
    return `<div class="tr-bar-row"><span class="tr-bar-arm">${a}</span>
      <span class="tr-bar-track"><span class="tr-bar" style="width:${v}%;background:${ARM_COLORS[a]}"></span></span>
      <span class="tr-bar-val">${pct(v)}</span></div>`;
  }).join("");
}

function setSection(s, armNames) {
  const legend = s.arms.map((a) => `<span class="tr-legend-item"><i style="background:${ARM_COLORS[a]}"></i><b>${a}</b> ${esc(armNames[a])}</span>`).join("");
  const rows = [...s.groups, ...(s.overall ? [s.overall] : [])].map((g) => {
    const n = (g.arms.B || g.arms.A).n;
    const grounded = s.arms.map((a) => g.arms[a] ? `${a} ${pct(g.arms[a].grounded_pct)}` : "").join(" · ");
    return `<div class="tr-group${g.key === "all" ? " overall" : ""}">
      <div class="tr-group-name">${esc(g.name)} <span class="tr-n">n=${n}</span></div>
      <div class="tr-bars">${bars(g.arms, s.arms)}</div>
      <div class="tr-group-sub">Grounded in sources: ${grounded}</div>
    </div>`;
  }).join("");
  const misses = s.misses.length ? s.misses.map((m) => {
    const sc = ["A", "B", "C"].filter((a) => m.scores[a] != null)
      .map((a) => `<span class="tr-score s${m.scores[a]}">${a}: ${SCORE[m.scores[a]]}</span>`).join("");
    const chk = m.check ? `<div class="tr-miss-check">v2 check: ${esc(m.check.level)} confidence${m.check.premise_flagged ? " · <b>premise flagged</b>" : ""}</div>` : "";
    return `<li class="tr-miss"><div class="tr-miss-q"><span class="tr-tag">${esc(m.group)}</span> ${esc(m.question)}</div>
      <div class="tr-scores">${sc}</div><div class="tr-miss-why">Judge on B: ${esc(m.why)}</div>${chk}</li>`;
  }).join("") : `<li class="tr-miss">No misses in this set.</li>`;
  let checker = "";
  const c = s.checker;
  if (c && c.n) {
    const r = (a, b) => (b ? `${a}/${b}` : "–");
    checker = `<div class="tr-checker"><h3>Does the v2 claim check flag the right answers?</h3>
      <p class="tr-note">Exploratory: not part of the locked rubric. "Flagged" means confidence below High, or a premise warning.</p>
      <div class="tr-cards small">
        ${card("Wrong answers it flagged", r(c.wrong_b_flagged, c.wrong_b), "higher is better")}
        ${card("Right answers it flagged anyway", r(c.right_b_flagged, c.right_b), "lower is better")}
        ${card("Trick questions: premise flagged", r(c.trick_premise_flagged, c.trick), "")}
        ${card("Other questions: premise flagged", r(c.nontrick_premise_flagged, c.nontrick), "false alarms")}
      </div></div>`;
  }
  return `<section class="tr-panel">
    <h2>${esc(s.label)}</h2><p class="tr-note">${esc(s.blurb)}</p>
    <div class="tr-legend">${legend}</div>
    <h3>Correct vs the answer key, by question type</h3>
    <div class="tr-groups">${rows}</div>
    ${checker}
    <details class="tr-misses"><summary>Every miss (${s.misses.length}): where SearchIQ wasn't fully right, or search made it worse</summary><ul>${misses}</ul></details>
  </section>`;
}

function method(data) {
  const ag = data.agreement;
  const agree = ag && ag.overall != null
    ? `A person re-scored 10 random answers; they matched the judge on ${pct(100 * ag.overall)} of labels.`
    : `A person is re-scoring 10 random answers to check the judge. <b>Until that's done, the judge's scores are unverified.</b>`;
  const locked = data.locked.map((l) => `${esc(l.what)} <code>${esc(l.commit)}</code>`).join(", ");
  return `<h2>How this is measured</h2>
    <ul class="tr-method">
      <li><b>Arms.</b> ${Object.entries(data.arms).map(([k, v]) => `<b>${k}</b>: ${esc(v)}`).join("; ")}. Same model and prompt; only search (and, for C, the check) differ.</li>
      <li><b>Locked first.</b> Questions, answer keys, rubric and the arm C definition were committed before they ran: ${locked}.</li>
      <li><b>Grading.</b> ${esc(data.models.judge)} grades each answer against the key: correct 0/1/2, grounded in the cited sources 0/1/2. Answers by ${esc(data.models.answers)}, all on ${esc(data.models.host)}. Run ${esc(data.run_dates)}.</li>
      <li><b>Judge check.</b> ${agree}</li>
      <li><b>Small samples.</b> 5–10 questions per group, so one answer moves a group 10–20 points. Treat gaps under ~20 points as noise. The live app samples at temperature 0.4, so its answers vary more.</li>
    </ul>
    <p class="tr-links"><a href="${esc(data.links.method)}" target="_blank" rel="noopener">Full method &amp; limitations</a> · <a href="${esc(data.links.raw)}" target="_blank" rel="noopener">Raw answers, searches and verdicts</a> · <a href="${esc(data.links.rubric)}" target="_blank" rel="noopener">Rubric</a></p>`;
}

(async function load() {
  const sum = document.getElementById("tr-summary");
  try {
    const res = await fetch("/static/track_record.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    sum.innerHTML = summaryCards(data);
    document.getElementById("tr-sets").innerHTML = data.sets.map((s) => setSection(s, data.arms)).join("");
    document.getElementById("tr-method").innerHTML = method(data);
  } catch (e) {
    sum.innerHTML = `<div class="tr-loading">Couldn't load the results right now. The raw data is on <a href="https://github.com/Nakul532/SearchIQ/tree/main/eval">GitHub</a>.</div>`;
  }
})();
