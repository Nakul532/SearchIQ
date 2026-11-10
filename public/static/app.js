"use strict";

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const resetBtn = document.getElementById("resetBtn");
const historyListEl = document.getElementById("historyList");
const newChatBtn = document.getElementById("newChatBtn");
const sidebarEl = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebarToggle");
const sidebarScrim = document.getElementById("sidebarScrim");

let busy = false;

// Default budget for the small JSON endpoints (metrics, prefer, history, …).
const REQUEST_TIMEOUT_MS = 60000;
// /api/query streams answers (NDJSON); the server caps each draft at 90s, run in
// parallel, so the outer client budget just needs headroom over that.
const STREAM_TIMEOUT_MS = 150000;
const MIN_QUERY_LENGTH = 2;

// Default loading label; complex queries get a more reassuring message.
const TYPING_LABEL_DEFAULT = "Searching the web & drafting two answers…";
const TYPING_LABEL_COMPLEX =
  "This is a complex query — searching and generating two detailed answers, please wait…";

// Mirror of the server's complexity classifier (search_agent.classify_query) so
// the client can pick the right timeout + loading message BEFORE sending (the
// server only classifies once it receives the request). Kept deliberately simple;
// it only needs to agree on the "complex" tier, which is what gets the long wait.
const COMPLEX_SIGNALS = [
  "analyze", "analyse", "analysis", "evaluate", "assess", "research",
  "implication", "impact of", "consequence", "pros and cons", "trade-off",
  "tradeoff", "compare and contrast", "in depth", "step by step", "step-by-step",
  "strategy", "architect", "design a", "design an", "framework for",
  "why does", "why do", "why is", "why are", "how would", "what would happen",
  "deep dive", "critique", "synthesize", "synthesis", "rank ", "recommend",
  "news", "latest news",
];
function looksComplex(query) {
  const q = (query || "").toLowerCase().trim();
  if (!q) return false;
  const wordCount = q.split(/\s+/).length;
  const questionMarks = (q.match(/\?/g) || []).length;
  if (wordCount > 24 || questionMarks > 1) return true;
  return COMPLEX_SIGNALS.some((s) => q.includes(s));
}

// --------------------------------------------------------------------------- //
// Conversation identity (per-chat; the USER identity comes from the auth cookie)
// --------------------------------------------------------------------------- //
function uuid() {
  return (crypto.randomUUID && crypto.randomUUID()) ||
    "id-" + Date.now().toString(36) + "-" + Math.random().toString(16).slice(2);
}
let conversationId = localStorage.getItem("searchiq_conversation_id") || "";
function setConversation(id) {
  conversationId = id;
  localStorage.setItem("searchiq_conversation_id", id);
}
function startNewConversation() {
  setConversation(uuid());
}
if (!conversationId) startNewConversation();

// fetch() has no built-in timeout — wrap it in an AbortController so a slow
// request fails cleanly (AbortError) instead of hanging. The signed session
// cookie carries identity — a Google user or a guest id (same-origin requests
// include it automatically).
async function fetchWithTimeout(url, options = {}, timeout = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

// Friendly message for a request that failed (timeout vs. generic). The timeout
// budget varies (60s normal, 120s complex), so report the actual limit.
function describeFetchError(err, timeoutMs = REQUEST_TIMEOUT_MS) {
  if (err && err.name === "AbortError") {
    const secs = Math.round(timeoutMs / 1000);
    return `That request took longer than ${secs}s and timed out. Web searches can be slow — please try again.`;
  }
  return (err && err.message) || "Network error — please try again.";
}

// --------------------------------------------------------------------------- //
// Lightweight markdown → HTML (escaped first, safe subset). No external deps.
// --------------------------------------------------------------------------- //
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function escapeAttr(s) {
  return escapeHtml(s).replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function hostname(url) {
  try { return new URL(url).hostname.replace(/^www\./, ""); }
  catch (_) { return url; }
}

// Render the RAG "Sources" section as citation cards (favicon · title · domain).
function renderSources(sources) {
  if (!sources || !sources.length) return "";
  const items = sources
    .map((s, i) => {
      const domain = hostname(s.url);
      const favicon = `https://www.google.com/s2/favicons?domain=${encodeURIComponent(domain)}&sz=64`;
      const letter = (domain[0] || "?").toUpperCase();
      return `
      <a class="source" href="${escapeAttr(s.url)}" target="_blank" rel="noopener noreferrer">
        <span class="src-fav">
          <img src="${escapeAttr(favicon)}" alt="" loading="lazy"
               onerror="this.style.display='none';this.nextElementSibling.style.display='grid';" />
          <span class="src-fav-fallback">${escapeHtml(letter)}</span>
        </span>
        <span class="src-text">
          <span class="src-title">${escapeHtml(s.title)}</span>
          <span class="src-domain">${escapeHtml(domain)}</span>
        </span>
        <span class="src-num">${i + 1}</span>
      </a>`;
    })
    .join("");
  return `
    <div class="sources">
      <div class="sources-label">Sources · web search</div>
      <div class="sources-list">${items}</div>
    </div>`;
}
// Render the pre-processor's routing decision as a small badge under an answer:
// "⚡ Simple · <fast model> · Fast" (green) / "🔍 Medium · <fast model> · Balanced"
// (blue) / "🧠 Complex · <deep model> · Deep" (purple). Web search always runs; the badge
// reflects which model aggregated the results.
function renderRoutingBadge(routing) {
  if (!routing || !routing.badge) return "";
  const cls = routing.complexity || "medium";
  return `
    <div class="route-badge ${cls}" title="Pre-processor routing decision (web search always runs)">
      <span class="rb-text">${escapeHtml(routing.badge)}</span>
    </div>`;
}
function renderMarkdown(md) {
  // Code fences first.
  const blocks = [];
  md = md.replace(/```[\w]*\n?([\s\S]*?)```/g, (_, code) => {
    blocks.push(`<pre><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });

  md = escapeHtml(md);
  const lines = md.split("\n");
  let html = "";
  let listType = null;
  const closeList = () => { if (listType) { html += `</${listType}>`; listType = null; } };

  for (let raw of lines) {
    const line = raw.trimEnd();
    let m;
    if ((m = line.match(/^###?\s+(.*)/))) { closeList(); html += `<h3>${inline(m[1])}</h3>`; continue; }
    if ((m = line.match(/^[-*]\s+(.*)/))) { if (listType !== "ul") { closeList(); html += "<ul>"; listType = "ul"; } html += `<li>${inline(m[1])}</li>`; continue; }
    if ((m = line.match(/^\d+\.\s+(.*)/))) { if (listType !== "ol") { closeList(); html += "<ol>"; listType = "ol"; } html += `<li>${inline(m[1])}</li>`; continue; }
    if (line.trim() === "") { closeList(); continue; }
    if (/^\u0000\d+\u0000$/.test(line.trim())) { closeList(); html += line.trim(); continue; }
    closeList(); html += `<p>${inline(line)}</p>`;
  }
  closeList();
  html = html.replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[Number(i)]);
  return html;
}
function inline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
}

// --------------------------------------------------------------------------- //
// DOM builders
// --------------------------------------------------------------------------- //
function clearWelcome() {
  const w = messagesEl.querySelector(".welcome");
  if (w) w.remove();
}
function scrollToBottom() {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: "smooth" });
}

function addUserMessage(text) {
  clearWelcome();
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="who">You</div><div class="bubble"></div>`;
  el.querySelector(".bubble").textContent = text;
  messagesEl.appendChild(el);
  scrollToBottom();
}

function addTyping(label = TYPING_LABEL_DEFAULT) {
  const el = document.createElement("div");
  el.className = "msg ai typing-wrap";
  el.innerHTML = `
    <div class="who">SearchIQ</div>
    <div class="typing">
      <span class="typing-dots"><i></i><i></i><i></i></span>
      <span class="typing-label"></span>
    </div>`;
  el.querySelector(".typing-label").textContent = label;
  messagesEl.appendChild(el);
  scrollToBottom();
  return el;
}

// The 3 reasons offered after a preference: [UI label, value sent to backend].
const REASON_OPTIONS = [
  ["More accurate", "more accurate"],
  ["Clearer explanation", "clearer explanation"],
  ["Better sources", "better sources"],
];

// The agent's natural clarifying question, rendered as an ordinary chat turn.
// The user just types their answer in the composer — no chips, no buttons. Their
// reply (and this question) live in the conversation history, so the next turn
// has the context to answer the original query.
function addClarifyQuestion(text) {
  const el = document.createElement("div");
  el.className = "msg ai";
  el.innerHTML = `<div class="who">SearchIQ</div><div class="bubble">${renderMarkdown(text)}</div>`;
  messagesEl.appendChild(el);
  scrollToBottom();
}

// "✨ Applied N lessons from past feedback" badge shown above an answer pair.
function appliedBadgeHtml(count) {
  if (!count) return "";
  return `<div class="applied-badge" title="Lessons learned from your past preferences">
    ✨ Applied ${count} lesson${count > 1 ? "s" : ""} from past feedback</div>`;
}

// "🗂 Cached answer from <date>" — shown when the pair was served from the cache.
function fmtDate(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return "";
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}
function cachedBadgeHtml(meta) {
  if (!meta || !meta.cached) return "";
  return `<div class="cached-badge" title="Served from the answer cache — no live search used">
    🗂 Cached answer from ${escapeHtml(fmtDate(meta.cached_at))}</div>`;
}

// "✨ Personalized for you" badge — shown when the answer used the user's profile.
function personalizedBadgeHtml(on) {
  if (!on) return "";
  return `<div class="personalized-badge" title="Tailored to your personality profile">
    ✨ Personalized for you</div>`;
}

// Model labels from the server (/api/me) — the fast and deep Groq models.
let MODEL_LABELS = { fast: "GPT-OSS 20B", deep: "GPT-OSS 120B" };

// Per-side header tag: model + style (e.g. "⚡ GPT-OSS 20B · Concise").
function sideStyleTag(routing, side) {
  const label = (routing && routing.model_label) || MODEL_LABELS.fast;
  const emoji = label === MODEL_LABELS.deep ? "🧠" : "⚡";
  const styleWord = side === "A" ? "Concise" : "Detailed";
  return `${emoji} ${escapeHtml(label)} · ${styleWord}`;
}

// Build the two-column A/B shell from the `meta` event, with both sides showing a
// shimmer until their `answer` events arrive. Returns a controller { el, data } the
// stream consumer fills incrementally (see fillAnswerSide / enablePicks).
function startComparison(meta) {
  const el = document.createElement("div");
  el.className = "msg ai comparison";
  const data = {
    comparison_id: null,
    applied_lessons: meta.applied_lessons || [],
    applied_count: meta.applied_count || 0,
    original_query: meta.original_query,
    followups: [],
  };
  const sideCard = (side, routing) => `
    <div class="ab-answer pending" data-side="${side}">
      <div class="ab-head">
        <span class="ab-label">Answer ${side}</span>
        <span class="ab-style">${sideStyleTag(routing, side)}</span>
      </div>
      <div class="bubble"><span class="ab-shimmer">Searching &amp; writing…</span></div>
      <div class="ab-check" hidden></div>
      <div class="ab-sources"></div>
      <button class="ab-pick" disabled>This one is better 👍</button>
    </div>`;

  el.innerHTML = `
    <div class="who">SearchIQ · two approaches, you pick</div>
    ${cachedBadgeHtml(meta)}
    ${personalizedBadgeHtml(meta.personalized)}
    ${appliedBadgeHtml(data.applied_count)}
    ${renderRoutingBadge(meta.routing)}
    <ol class="steps" aria-label="What SearchIQ did"></ol>
    <div class="premise-flag" hidden></div>
    <div class="ab-compare">
      ${sideCard("A", meta.routing_a)}
      ${sideCard("B", meta.routing_b)}
    </div>
    <div class="disagree" hidden></div>
    <div class="ab-reason" hidden></div>`;

  el.querySelectorAll(".ab-answer").forEach((card) => {
    card.querySelector(".ab-pick").addEventListener("click", () =>
      choosePreferred(data, card.dataset.side, el)
    );
  });

  messagesEl.appendChild(el);
  scrollToBottom();
  return { el, data };
}

// Fill one side's bubble + sources once its `answer` event arrives.
function fillAnswerSide(comp, side, answer, sources) {
  if (!comp) return;
  const card = comp.el.querySelector(`.ab-answer[data-side="${side}"]`);
  if (!card) return;
  card.classList.remove("pending");
  if (sources && sources.length) comp.data.sources = sources;
  card.querySelector(".bubble").innerHTML = renderMarkdown(answer);
  const srcWrap = card.querySelector(".ab-sources");
  if (srcWrap) srcWrap.innerHTML = renderSources(sources);
  scrollToBottom();
}

// --------------------------------------------------------------------------- //
// v2 agent: steps panel, claim check, confidence labels, disagreement view
// --------------------------------------------------------------------------- //
const STEP_LABELS = { cache: "Cache", search: "Search", draft: "Draft", check: "Check claims", revise: "Rewrite" };
const CONF_LABELS = { high: "High confidence", medium: "Medium confidence", low: "Low confidence", none: "No checkable claims" };

function updateStep(comp, ev) {
  if (!comp) return;
  const list = comp.el.querySelector(".steps");
  let li = list.querySelector(`li[data-step="${ev.step}"]`);
  if (!li) {
    li = document.createElement("li");
    li.dataset.step = ev.step;
    list.appendChild(li);
  }
  li.className = `step ${ev.status}`;
  const icon = ev.status === "done" ? "✓" : ev.status === "skipped" ? "–" : "";
  li.innerHTML = `<span class="step-icon" aria-hidden="true">${icon}</span>` +
    `<span class="step-name">${escapeHtml(STEP_LABELS[ev.step] || ev.step)}</span>` +
    (ev.detail ? `<span class="step-detail">${escapeHtml(ev.detail)}</span>` : "");
}

function srcRefs(nums, sources) {
  return (nums || []).map((n) => {
    const s = (sources || [])[n - 1];
    return s
      ? `<a class="claim-ref" href="${escapeAttr(s.url)}" target="_blank" rel="noopener" title="${escapeAttr(s.title || s.url)}">[${n}]</a>`
      : `<span class="claim-ref">[${n}]</span>`;
  }).join("");
}

function renderClaims(side, sources) {
  const c = side.confidence;
  const counts = c.total ? ` · ${c.backed}/${c.total} claims backed` : "";
  const rows = side.claims.map((cl) => {
    const tag = cl.status === "backed" ? ["✓", "Backed", "ok"]
      : cl.status === "contradicted" ? ["✗", "Contradicted", "bad"] : ["⚠", "Double-check", "warn"];
    const note = cl.status === "contradicted" && cl.note ? `<div class="claim-note">${escapeHtml(cl.note)}</div>`
      : cl.status === "unsupported" ? `<div class="claim-note">Not stated in the sources: verify before you use it.</div>` : "";
    return `<li class="claim ${tag[2]}"><span class="claim-tag">${tag[0]} ${tag[1]} ${srcRefs(cl.sources, sources)}</span>` +
      `<span class="claim-text">${escapeHtml(cl.text)}</span>${note}</li>`;
  }).join("");
  const open = c.level === "low" || c.contradicted ? " open" : "";
  return `<details class="check-box"${open}>
      <summary><span class="conf conf-${c.level}">${CONF_LABELS[c.level] || c.level}</span><span class="conf-count">${counts}</span></summary>
      ${rows ? `<ul class="claims">${rows}</ul>` : `<div class="claim-note">This answer makes no factual claims to check.</div>`}
    </details>`;
}

// The check acted: swap in the rewritten draft, say why, keep the original one click away.
function applyRevision(comp, ev) {
  if (!comp) return;
  const card = comp.el.querySelector(`.ab-answer[data-side="${ev.side}"]`);
  if (!card) return;
  card.querySelector(".bubble").innerHTML = renderMarkdown(ev.answer);
  let box = card.querySelector(".revised-box");
  if (!box) {
    box = document.createElement("div");
    box.className = "revised-box";
    card.querySelector(".bubble").before(box);
  }
  const reasons = (ev.reasons || []).map((r) => `<li>${escapeHtml(r)}</li>`).join("");
  box.innerHTML = `<div class="revised-head">✎ Rewritten by SearchIQ's check</div>` +
    (reasons ? `<ul class="revised-reasons">${reasons}</ul>` : "") +
    `<details class="revised-original"><summary>Show the original draft</summary><div class="revised-original-body">${renderMarkdown(ev.original || "")}</div></details>`;
  scrollToBottom();
}

function applyCheck(comp, check) {
  if (!comp || !check || !check.sides) return;
  const sources = comp.data.sources || [];
  Object.entries(check.sides).forEach(([side, s]) => {
    const box = comp.el.querySelector(`.ab-answer[data-side="${side}"] .ab-check`);
    if (!box) return;
    box.innerHTML = renderClaims(s, sources);
    box.hidden = false;
  });
  const flag = comp.el.querySelector(".premise-flag");
  if (check.premise && check.premise.ok === false) {
    flag.innerHTML = `<b>⚠ Check the question itself.</b> ${escapeHtml(check.premise.note || "The sources suggest the question rests on a false assumption.")}`;
    flag.hidden = false;
  }
  const dis = comp.el.querySelector(".disagree");
  const ds = check.disagreements || [];
  if (Object.keys(check.sides).length === 2) {
    const favor = (d) => d.supported === "A" || d.supported === "B"
      ? `Sources back ${d.supported} ${srcRefs(d.sources, sources)}`
      : d.supported === "both" ? `Sources back both ${srcRefs(d.sources, sources)}` : "Sources don’t settle it: double-check";
    dis.innerHTML = ds.length
      ? `<div class="disagree-head">Where A and B disagree</div>` + ds.map((d) => `
          <div class="disagree-row">
            <div class="disagree-topic">${escapeHtml(d.topic)}</div>
            <div class="disagree-sides">
              <div class="${d.supported === "A" ? "favored" : ""}"><b>A:</b> ${escapeHtml(d.A)}</div>
              <div class="${d.supported === "B" ? "favored" : ""}"><b>B:</b> ${escapeHtml(d.B)}</div>
            </div>
            <div class="disagree-verdict">${favor(d)}</div>
          </div>`).join("")
      : `<div class="disagree-none">✓ A and B agree on the key facts.</div>`;
    dis.hidden = false;
  }
  scrollToBottom();
}

// Both drafts are in and the comparison is recorded — enable the pick buttons.
function enablePicks(comp, comparisonId) {
  if (!comp) return;
  comp.data.comparison_id = comparisonId;
  comp.el.querySelectorAll(".ab-answer .ab-pick").forEach((b) => { b.disabled = false; });
}

// One draft failed/timed out: keep the survivor, mark the other, drop the picks,
// and offer follow-ups (the survivor is already committed server-side).
function collapseToSingle(comp, ev) {
  if (!comp) { addSingleAnswer(ev); return; }
  comp.el.classList.add("single");
  comp.el.querySelectorAll(".ab-answer").forEach((card) => {
    const pick = card.querySelector(".ab-pick");
    if (pick) pick.remove();
    if (card.dataset.side !== ev.side) {
      card.classList.remove("pending");
      card.classList.add("failed");
      card.querySelector(".bubble").innerHTML =
        `<span class="ab-failed">This draft didn’t come back this time.</span>`;
    }
  });
  const note = document.createElement("div");
  note.className = "single-note";
  note.textContent = "⚠ Showing one answer — the second draft didn’t come back this time.";
  comp.el.appendChild(note);
  if (ev.followups && ev.followups.length) addFollowups(ev.followups);
  scrollToBottom();
}

// Fallback render: only one answer was generated (the other call failed), so
// there's nothing to compare — show the single answer and offer follow-ups.
// The answer is already committed server-side, so no prefer/reason step.
function addSingleAnswer(data) {
  const el = document.createElement("div");
  el.className = "msg ai";
  el.innerHTML = `
    <div class="who">SearchIQ</div>
    ${appliedBadgeHtml(data.applied_count)}
    ${renderRoutingBadge(data.routing)}
    <div class="single-note">⚠ Showing one answer — the second draft didn’t come back this time.</div>
    <div class="bubble">${renderMarkdown(data.answer)}</div>
    ${renderSources(data.sources)}`;
  messagesEl.appendChild(el);
  scrollToBottom();
  if (data.followups && data.followups.length) addFollowups(data.followups);
}

async function choosePreferred(data, side, compareEl) {
  if (compareEl.classList.contains("chosen")) return;
  compareEl.classList.add("chosen");

  // Visually mark winner / loser.
  compareEl.querySelectorAll(".ab-answer").forEach((card) => {
    card.querySelector(".ab-pick").disabled = true;
    if (card.dataset.side === side) card.classList.add("winner");
    else card.classList.add("loser");
  });

  try {
    const res = await fetchWithTimeout("/api/prefer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comparison_id: data.comparison_id, preferred: side }),
    });
    const out = await res.json().catch(() => ({}));
    if (out.limit) { addLimitCard(out.limit); return; }
    if (!res.ok) { addError(out.error || `Request failed (${res.status})`); return; }
    if (out.metrics) { try { applyMetrics(out.metrics); } catch (e) { console.error(e); } }
    if (out.policy_update) showPolicyUpdate(out.policy_update);
    data.followups = out.followups || [];  // shown after the reason step
    // The winning answer is now committed as the conversation's assistant turn.
    loadConversations();
    showReasonChips(data, compareEl);
  } catch (err) {
    addError(describeFetchError(err));
  }
}

function showReasonChips(data, compareEl) {
  const wrap = compareEl.querySelector(".ab-reason");
  const chips = REASON_OPTIONS
    .map(([label, value]) => `<button class="reason-chip" data-value="${escapeAttr(value)}">${escapeHtml(label)}</button>`)
    .join("");
  wrap.innerHTML = `<div class="reason-head">Why did you prefer this?</div><div class="reason-chips">${chips}</div>`;
  wrap.hidden = false;
  wrap.querySelectorAll(".reason-chip").forEach((b) => {
    b.addEventListener("click", () => chooseReason(data, b.dataset.value, wrap));
  });
  scrollToBottom();
}

async function chooseReason(data, reason, wrapEl) {
  if (wrapEl.classList.contains("done")) return;
  wrapEl.classList.add("done");
  wrapEl.querySelectorAll(".reason-chip").forEach((b) => {
    b.disabled = true;
    if (b.dataset.value === reason) b.classList.add("chosen");
  });

  try {
    const res = await fetchWithTimeout("/api/prefer_reason", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comparison_id: data.comparison_id, reason }),
    });
    const out = await res.json().catch(() => ({}));
    if (out.limit) { addLimitCard(out.limit); return; }
    if (!res.ok) { addError(out.error || `Request failed (${res.status})`); return; }
    if (out.learning_card) addLearningCard(out.learning_card);
    if (out.metrics) { try { applyMetrics(out.metrics); } catch (e) { console.error(e); } }
    if (out.lesson) highlightLessons([out.lesson.id]);
    // Offer follow-up questions to continue the conversation.
    if (data.followups && data.followups.length) addFollowups(data.followups);
  } catch (err) {
    addError(describeFetchError(err));
  }
}

// The "Learning Card" — plain-English statement of what the system just learned.
function addLearningCard(text) {
  const el = document.createElement("div");
  el.className = "learning-card";
  el.innerHTML = `<span class="lc-icon">📚</span><div class="lc-text"><b>Learned:</b> ${escapeHtml(text)}</div>`;
  messagesEl.appendChild(el);
  scrollToBottom();
}

// Follow-up question chips — tapping one immediately asks it (same conversation).
function addFollowups(questions) {
  const el = document.createElement("div");
  el.className = "followups";
  const chips = questions
    .map((q) => `<button class="followup-chip">${escapeHtml(q)}</button>`)
    .join("");
  el.innerHTML = `<div class="followups-head">💡 Continue the conversation</div><div class="followups-chips">${chips}</div>`;
  el.querySelectorAll(".followup-chip").forEach((b) => {
    b.addEventListener("click", () => {
      el.querySelectorAll(".followup-chip").forEach((x) => (x.disabled = true));
      // force: already specific; followup: rewards the prior answer (+0.2).
      sendQuery(b.textContent.trim(), true, true);
    });
  });
  messagesEl.appendChild(el);
  scrollToBottom();
}

// --------------------------------------------------------------------------- //
// API flows
// --------------------------------------------------------------------------- //
let awaitingClarification = false;  // the last turn was a clarifying question

async function sendQuery(forcedQuery, force = false, followup = false) {
  if (busy) return;
  const query = (forcedQuery != null ? forcedQuery : inputEl.value).trim();

  // Reject empty or too-short queries (e.g. a lone "?") before sending.
  if (query.length < MIN_QUERY_LENGTH) {
    rejectQuery(query.length === 0 ? "Type a question to search." : "Please enter at least 2 characters.");
    return;
  }

  // If we just asked a clarifying question, this message is the user's answer to
  // it — skip the clarify check and let the conversation history carry the context.
  if (awaitingClarification) { force = true; awaitingClarification = false; }

  // Complex queries take longer (deep-model draft B), so use a reassuring loading
  // message until the first answer streams in.
  const complex = looksComplex(query);

  setBusy(true);
  addUserMessage(query);
  if (forcedQuery == null) { inputEl.value = ""; autoGrow(); }
  let typing = addTyping(complex ? TYPING_LABEL_COMPLEX : TYPING_LABEL_DEFAULT);
  const clearTyping = () => { if (typing) { typing.remove(); typing = null; } };

  // Stream state for this turn.
  let comp = null;             // the A/B comparison controller (after `meta`)
  let appliedLessons = [];     // lessons applied to this answer (pulsed at the end)
  let finished = false;        // a comparison/single arrived → refresh metrics after

  const handleEvent = (ev) => {
    switch (ev.type) {
      case "clarify":
        clearTyping();
        awaitingClarification = true;
        if (ev.cached) addNotice(`🗂 Cached answer from ${fmtDate(ev.cached_at)}`);
        addClarifyQuestion(ev.question);
        loadConversations();  // the clarify exchange was saved as a turn
        break;
      case "meta":
        clearTyping();
        awaitingClarification = false;
        appliedLessons = ev.applied_lessons || [];
        comp = startComparison(ev);
        break;
      case "answer":
        fillAnswerSide(comp, ev.side, ev.answer, ev.sources);
        break;
      case "step":
        updateStep(comp, ev);
        break;
      case "check":
        applyCheck(comp, ev);
        break;
      case "revise":
        applyRevision(comp, ev);
        break;
      case "comparison":
        enablePicks(comp, ev.comparison_id);
        finished = true;
        break;
      case "single":
        collapseToSingle(comp, ev);
        finished = true;
        break;
      case "limit":
        clearTyping();
        if (comp) { comp.el.remove(); comp = null; }  // drop the half-built A/B shell
        addLimitCard(ev);
        loadConversations();
        break;
      case "quota":
        applyQuota(ev);
        break;
      case "error":
        clearTyping();
        addError(ev.error || "Something went wrong — please try again.");
        break;
      default:
        break;
    }
  };

  try {
    const res = await fetchWithTimeout("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, force, followup, conversation_id: conversationId }),
    }, STREAM_TIMEOUT_MS);
    if (!res.ok) {
      const out = await res.json().catch(() => ({}));
      if (out.limit) { clearTyping(); addLimitCard(out.limit); return; }
      throw new Error(out.error || `Request failed (${res.status})`);
    }

    // Read the NDJSON stream line by line, dispatching each event as it arrives.
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        let ev;
        try { ev = JSON.parse(line); } catch (_) { continue; }
        handleEvent(ev);
      }
    }

    if (finished) {
      // Refresh the feed first, THEN pulse the lessons applied to this answer
      // (refreshMetrics rebuilds the feed DOM, so the highlight must come after).
      await refreshMetrics();
      if (appliedLessons.length) highlightLessons(appliedLessons.map((l) => l.id));
      loadConversations();  // this turn may have created/renamed the conversation
      loadMe();             // the profile rebuilds every 5 queries — reflect it
    }
  } catch (err) {
    clearTyping();
    addError(describeFetchError(err, STREAM_TIMEOUT_MS));
  } finally {
    setBusy(false);
  }
}

function addError(msg) {
  const el = document.createElement("div");
  el.className = "msg ai";
  el.innerHTML = `<div class="who">Agent</div><div class="bubble" style="border-color:var(--down);color:var(--down)">⚠ ${escapeHtml(msg)}</div>`;
  messagesEl.appendChild(el);
  scrollToBottom();
}

// Friendly "limit reached" card — shown instead of an error when a daily cap or a
// provider's free-tier limit is hit. Offers the demo video and cached examples.
function addLimitCard(info) {
  const el = document.createElement("div");
  el.className = "msg ai";
  const resets = info.resets_at
    ? new Date(info.resets_at).toLocaleString(undefined, { weekday: "short", hour: "numeric", minute: "2-digit" })
    : "";
  const demo = info.demo_url
    ? `<a class="limit-demo" href="${escapeAttr(info.demo_url)}" target="_blank" rel="noopener">▶ Watch the demo video</a>`
    : "";
  const signIn = info.reason === "guest" && ME.oauth_configured
    ? `<a class="limit-signin" href="/auth/login">Sign in with Google for more</a>` : "";
  const ex = (info.examples || [])
    .map((e) => `<button class="chip limit-example" data-q="${escapeAttr(e.query)}">${escapeHtml(e.query)}</button>`)
    .join("");
  el.innerHTML = `
    <div class="who">SearchIQ</div>
    <div class="limit-card">
      <div class="limit-title">⏳ ${escapeHtml(info.title || "Daily limit reached")}</div>
      <div class="limit-body">${escapeHtml(info.message || "")}</div>
      ${resets ? `<div class="limit-reset">Resets ${escapeHtml(resets)} (your time)</div>` : ""}
      <div class="limit-actions">${demo}${signIn}</div>
      ${ex ? `<div class="limit-ex-head">Try a cached example (free):</div><div class="limit-examples">${ex}</div>` : ""}
    </div>`;
  el.querySelectorAll(".limit-example").forEach((b) =>
    b.addEventListener("click", () => sendQuery(b.dataset.q))
  );
  messagesEl.appendChild(el);
  scrollToBottom();
}

// "2 of 3 free searches left today" pill in the header.
function applyQuota(q) {
  const pill = document.getElementById("quotaPill");
  if (!pill || !q || q.cap == null) return;
  pill.hidden = false;
  pill.textContent = `${q.remaining}/${q.cap} ${q.signed_in ? "" : "free "}searches left today`;
  pill.classList.toggle("empty", q.remaining <= 0);
}

function addNotice(msg) {
  const el = document.createElement("div");
  el.className = "notice";
  el.textContent = msg;
  messagesEl.appendChild(el);
  scrollToBottom();
}

// --------------------------------------------------------------------------- //
// Metrics + animations
// --------------------------------------------------------------------------- //
let lastRate = 0;
function animateNumber(el, to, suffix = "") {
  const from = lastRate;
  const start = performance.now();
  const dur = 700;
  function step(now) {
    const t = Math.min(1, (now - start) / dur);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(from + (to - from) * eased) + suffix;
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
  lastRate = to;
}

function applyMetrics(m) {
  const rate = Math.round(m.completion_rate);
  animateNumber(document.getElementById("completionRate"), rate);
  document.getElementById("northstarBar").style.width = rate + "%";
  document.getElementById("totalQueries").textContent = m.total_queries;
  document.getElementById("ratedQueries").textContent = m.rated_queries;
  document.getElementById("improvedCount").textContent = m.improved_count;
  document.getElementById("upCount").textContent = m.thumbs_up;
  document.getElementById("downCount").textContent = m.thumbs_down;

  // Routing economics: share of answers on the fast model + estimated cost per
  // query at Groq list prices (an estimate — this demo runs on the free tier).
  if (m.model_labels) MODEL_LABELS = m.model_labels;
  const shareEl = document.getElementById("fastShare");
  if (shareEl) shareEl.textContent = Math.round(m.fast_share_pct ?? 0) + "%";
  const shareLbl = document.getElementById("fastShareLabel");
  if (shareLbl) shareLbl.textContent = `⚡ routed to ${MODEL_LABELS.fast}`;
  const cpqEl = document.getElementById("costPerQuery");
  if (cpqEl) cpqEl.textContent = "~$" + Number(m.est_cost_per_query_usd ?? 0).toFixed(4);

  // bar chart
  const chart = document.getElementById("feedbackChart");
  chart.innerHTML = "";
  if (!m.recent_feedback.length) {
    chart.innerHTML = `<div class="empty-chart">No feedback yet</div>`;
  } else {
    m.recent_feedback.forEach((fb, i) => {
      const bar = document.createElement("div");
      bar.className = "bar " + fb;
      bar.style.height = (fb === "up" ? 100 : 55) + "%";
      bar.style.animationDelay = i * 25 + "ms";
      bar.title = fb === "up" ? "Helpful" : "Not helpful";
      chart.appendChild(bar);
    });
  }

  // improved list
  const list = document.getElementById("improvedList");
  list.innerHTML = "";
  if (!m.most_improved.length) {
    list.innerHTML = `<li class="empty">No improvements yet</li>`;
  } else {
    m.most_improved.slice(-5).reverse().forEach((q) => {
      const li = document.createElement("li");
      li.textContent = q;
      list.appendChild(li);
    });
  }

  // Training progress: total lessons learned + total times applied.
  const tp = m.training_progress || { lessons_learned: 0, lessons_applied: 0 };
  const learnedEl = document.getElementById("lessonsLearned");
  const appliedEl = document.getElementById("lessonsApplied");
  if (learnedEl) learnedEl.textContent = tp.lessons_learned;
  if (appliedEl) appliedEl.textContent = tp.lessons_applied;

  // Lessons Learned feed.
  renderLessonsFeed(m.lessons_feed || []);

  // Reinforcement-learning reward panel.
  renderReward(m.reward, m.policy);
}

// --------------------------------------------------------------------------- //
// Reward dashboard panel (reinforcement-learning signals)
// --------------------------------------------------------------------------- //
function fmtScore(v) {
  const n = Number(v || 0);
  return (n >= 0 ? "+" : "") + n.toFixed(2);
}

// Diverging horizontal bars from a center baseline: positive grows right (green),
// negative grows left (red). Scaled against the ±1 reward range so groups compare.
function renderRewardBars(containerId, data, labels) {
  const el = document.getElementById(containerId);
  if (!el) return;
  const entries = Object.keys(labels).map((k) => [labels[k], Number(data[k] || 0)]);
  const maxAbs = Math.max(1, ...entries.map(([, v]) => Math.abs(v)));
  el.innerHTML = entries
    .map(([name, v]) => {
      const pos = v >= 0;
      const w = (Math.abs(v) / maxAbs) * 50;
      const fill = pos ? `left:50%;width:${w}%` : `right:50%;width:${w}%`;
      return `
      <div class="rb-row">
        <span class="rb-key">${escapeHtml(name)}</span>
        <span class="rb-track"><span class="rb-fill ${pos ? "pos" : "neg"}" style="${fill}"></span></span>
        <span class="rb-val ${pos ? "pos" : "neg"}">${fmtScore(v)}</span>
      </div>`;
    })
    .join("");
}

// Line chart of the running average reward over time (no chart library — raw SVG).
function drawRewardChart(timeline) {
  const svg = document.getElementById("rewardChart");
  const empty = document.getElementById("rewardChartEmpty");
  if (!svg) return;
  if (!timeline || timeline.length < 2) {
    svg.innerHTML = "";
    if (empty) empty.style.display = "grid";
    return;
  }
  if (empty) empty.style.display = "none";
  const W = 300, H = 96, pad = 10;
  const avgs = timeline.map((t) => t.avg);
  let lo = Math.min(0, ...avgs), hi = Math.max(0, ...avgs);
  if (hi - lo < 0.001) { hi += 0.5; lo -= 0.5; }
  const x = (i) => pad + (W - 2 * pad) * (i / (timeline.length - 1));
  const y = (v) => pad + (H - 2 * pad) * (1 - (v - lo) / (hi - lo));
  const pts = timeline.map((t, i) => `${x(i).toFixed(1)},${y(t.avg).toFixed(1)}`).join(" ");
  const area = `${x(0).toFixed(1)},${y(lo).toFixed(1)} ${pts} ${x(timeline.length - 1).toFixed(1)},${y(lo).toFixed(1)}`;
  const last = timeline[timeline.length - 1];
  const dotColor = last.avg >= 0 ? "var(--up)" : "var(--down)";
  const zeroY = y(0).toFixed(1);
  svg.innerHTML = `
    <polygon points="${area}" class="rc-area" />
    <line x1="${pad}" y1="${zeroY}" x2="${W - pad}" y2="${zeroY}" class="rc-zero" />
    <polyline points="${pts}" class="rc-line" />
    <circle cx="${x(timeline.length - 1).toFixed(1)}" cy="${y(last.avg).toFixed(1)}" r="3.4" fill="${dotColor}" />`;
}

function renderReward(reward, policy) {
  if (!reward) return;
  const avgEl = document.getElementById("rewardAvg");
  if (avgEl) avgEl.textContent = fmtScore(reward.avg_score);
  const vel = Number(reward.learning_velocity || 0);
  const velEl = document.getElementById("rewardVelocity");
  if (velEl) {
    velEl.textContent = fmtScore(vel);
    velEl.classList.toggle("pos", vel >= 0);
    velEl.classList.toggle("neg", vel < 0);
  }
  const posEl = document.getElementById("rewardPos");
  const negEl = document.getElementById("rewardNeg");
  if (posEl) posEl.textContent = reward.positive_count;
  if (negEl) negEl.textContent = reward.negative_count;

  drawRewardChart(reward.timeline);
  renderRewardBars("rewardByModel", reward.by_model || {}, {
    [MODEL_LABELS.fast]: "⚡ " + MODEL_LABELS.fast,
    [MODEL_LABELS.deep]: "🧠 " + MODEL_LABELS.deep,
  });
  renderRewardBars("rewardByType", reward.by_query_type || {}, { simple: "Simple", medium: "Medium", complex: "Complex" });
  renderRewardBars("rewardByStyle", reward.by_style || {}, { concise: "Concise", detailed: "Detailed" });

  const pol = document.getElementById("rewardPolicy");
  if (pol) {
    if (policy && policy.text) {
      pol.hidden = false;
      pol.innerHTML = `<span class="rp-icon">⚙️</span><div><div class="rp-head">Active policy</div><div class="rp-text">${escapeHtml(policy.text)}</div></div>`;
    } else {
      pol.hidden = true;
      pol.innerHTML = "";
    }
  }
}

// In-chat "Policy Updated" notification when the learned policy changes.
function showPolicyUpdate(p) {
  if (!p || !p.text) return;
  const el = document.createElement("div");
  el.className = "policy-card";
  el.innerHTML = `<span class="pc-icon">⚙️</span><div class="pc-text"><b>Policy Updated</b> · Learned: ${escapeHtml(p.text)}</div>`;
  messagesEl.appendChild(el);
  scrollToBottom();
}

// Render the live "Lessons Learned" feed. Each row carries data-lesson-id so it
// can be pulse-highlighted when that lesson is applied to a new answer.
function renderLessonsFeed(lessons) {
  const feed = document.getElementById("lessonsFeed");
  if (!feed) return;
  feed.innerHTML = "";
  if (!lessons.length) {
    feed.innerHTML = `<li class="empty">No lessons yet — pick a winner to teach SearchIQ</li>`;
    return;
  }
  lessons.forEach((l) => {
    const li = document.createElement("li");
    li.className = "lesson";
    li.dataset.lessonId = l.id;
    li.innerHTML = `
      <div class="lesson-text">${escapeHtml(l.text)}</div>
      <div class="lesson-meta">
        <span class="lesson-time">${escapeHtml(relativeTime(l.created_at))}</span>
        <span class="lesson-applied" title="Times applied to new answers">✨ ${l.applied_count}×</span>
      </div>`;
    feed.appendChild(li);
  });
}

// Pulse-highlight lessons in the feed when they're applied/learned.
function highlightLessons(ids) {
  if (!ids || !ids.length) return;
  ids.forEach((id) => {
    const el = document.querySelector(`#lessonsFeed [data-lesson-id="${id}"]`);
    if (!el) return;
    el.classList.remove("pulse");
    void el.offsetWidth;       // restart the animation
    el.classList.add("pulse");
    setTimeout(() => el.classList.remove("pulse"), 1800);
  });
}

function relativeTime(iso) {
  if (!iso) return "";
  const then = new Date(iso.replace(" ", "T"));
  const secs = Math.max(0, (Date.now() - then.getTime()) / 1000);
  if (secs < 60) return "just now";
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

async function refreshMetrics() {
  try {
    const res = await fetchWithTimeout("/api/metrics", {}, 15000);
    applyMetrics(await res.json());
  } catch (_) { /* ignore */ }
}

// --------------------------------------------------------------------------- //
// History sidebar + conversation loading
// --------------------------------------------------------------------------- //
async function loadConversations() {
  try {
    const res = await fetchWithTimeout("/api/conversations", {}, 15000);
    const data = await res.json();
    renderHistory(data.conversations || []);
  } catch (_) { /* non-critical */ }
}

// Group conversations into Today / Yesterday / Last Week / Older by last activity.
function dateBucket(iso) {
  if (!iso) return "Older";
  const d = new Date(iso.replace(" ", "T"));
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const dayMs = 86400000;
  if (d >= startOfToday) return "Today";
  if (d >= new Date(startOfToday - dayMs)) return "Yesterday";
  if (d >= new Date(startOfToday - 7 * dayMs)) return "Last Week";
  return "Older";
}

function renderHistory(conversations) {
  historyListEl.innerHTML = "";
  if (!conversations.length) {
    historyListEl.innerHTML = `<div class="history-empty">No conversations yet</div>`;
    return;
  }
  const order = ["Today", "Yesterday", "Last Week", "Older"];
  const groups = {};
  conversations.forEach((c) => {
    (groups[dateBucket(c.last_at)] ||= []).push(c);
  });
  order.forEach((bucket) => {
    if (!groups[bucket]) return;
    const head = document.createElement("div");
    head.className = "history-group";
    head.textContent = bucket;
    historyListEl.appendChild(head);
    groups[bucket].forEach((c) => {
      const item = document.createElement("button");
      item.className = "history-item" + (c.conversation_id === conversationId ? " active" : "");
      item.dataset.conversationId = c.conversation_id;
      item.title = c.title;
      item.textContent = c.title;
      item.addEventListener("click", () => selectConversation(c.conversation_id));
      historyListEl.appendChild(item);
    });
  });
}

function markActiveConversation() {
  historyListEl.querySelectorAll(".history-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.conversationId === conversationId);
  });
}

// Render a historical assistant turn (a previously-chosen answer + its sources).
function addHistoricalAnswer(content, sources) {
  const el = document.createElement("div");
  el.className = "msg ai";
  el.innerHTML = `
    <div class="who">SearchIQ</div>
    <div class="bubble">${renderMarkdown(content)}</div>
    ${renderSources(sources)}`;
  messagesEl.appendChild(el);
}

async function selectConversation(id) {
  if (busy) return;
  try {
    const res = await fetchWithTimeout(`/api/conversation/${encodeURIComponent(id)}`, {}, 15000);
    const data = await res.json();
    setConversation(id);
    awaitingClarification = false;
    closeSidebarMobile();
    messagesEl.innerHTML = "";
    const msgs = data.messages || [];
    if (!msgs.length) {
      messagesEl.innerHTML = WELCOME_HTML;
    } else {
      msgs.forEach((m) => {
        if (m.role === "user") addUserMessage(m.content);
        else addHistoricalAnswer(m.content, m.sources);
      });
    }
    markActiveConversation();
    scrollToBottom();
  } catch (err) {
    addError(describeFetchError(err));
  }
}

function newChat() {
  startNewConversation();
  messagesEl.innerHTML = WELCOME_HTML;
  awaitingClarification = false;
  markActiveConversation();
  closeSidebarMobile();
  inputEl.focus();
}

// Mobile sidebar drawer
function openSidebarMobile() { sidebarEl.classList.add("open"); sidebarScrim.hidden = false; }
function closeSidebarMobile() { sidebarEl.classList.remove("open"); sidebarScrim.hidden = true; }
function toggleSidebarMobile() {
  if (sidebarEl.classList.contains("open")) closeSidebarMobile(); else openSidebarMobile();
}

// --------------------------------------------------------------------------- //
// UI plumbing
// --------------------------------------------------------------------------- //
function setBusy(b) {
  busy = b;
  sendBtn.disabled = b;
}
function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
}

// Reject a too-short query: shake the composer and flash a transient hint.
let hintTimer = null;
function rejectQuery(message) {
  const inner = document.querySelector(".composer-inner");
  const hint = document.querySelector(".composer-hint");
  if (inner) {
    inner.classList.remove("invalid");
    void inner.offsetWidth; // restart the animation
    inner.classList.add("invalid");
    setTimeout(() => inner.classList.remove("invalid"), 500);
  }
  if (hint) {
    const original = hint.dataset.original || (hint.dataset.original = hint.innerHTML);
    hint.textContent = message;
    hint.classList.add("hint-warn");
    clearTimeout(hintTimer);
    hintTimer = setTimeout(() => { hint.innerHTML = original; hint.classList.remove("hint-warn"); }, 2200);
  }
  inputEl.focus();
}

inputEl.addEventListener("input", autoGrow);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuery(); }
});
sendBtn.addEventListener("click", () => sendQuery());

// Delegated so suggestion chips keep working after a reset re-renders them.
messagesEl.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (chip) { inputEl.value = chip.textContent.trim(); autoGrow(); inputEl.focus(); }
});

const WELCOME_HTML = `
  <div class="welcome">
    <div class="welcome-logo">
      <svg viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="white" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="10.5" cy="10.5" r="6.4" /><path d="M20 20l-4.2-4.2" />
      </svg>
    </div>
    <h2>Research a company, market or deal</h2>
    <p>Built for analysts doing business and competitive research. Every answer cites
       its web sources, so you can see which facts are <b>backed and safe to put in front
       of executives</b> — and anything without a source is your cue to double-check.
       Compare two drafts, pick the better one, and SearchIQ learns how you like your briefs.</p>
    <div class="suggestions">
      <button class="chip">Who founded Nvidia, and in what year?</button>
      <button class="chip">What was Apple's total net sales in fiscal year 2023?</button>
      <button class="chip">When did Microsoft complete its acquisition of Activision Blizzard?</button>
    </div>
  </div>`;

resetBtn.addEventListener("click", async () => {
  if (!confirm("Reset your data? This clears YOUR conversations, history, metrics, learning, and profile. Your account stays.")) return;
  await fetchWithTimeout("/api/reset", { method: "POST" }, 15000);
  startNewConversation();
  messagesEl.innerHTML = WELCOME_HTML;
  lastRate = 0;
  refreshMetrics();
  loadConversations();
  loadMe();  // profile was cleared
});

newChatBtn.addEventListener("click", newChat);
sidebarToggle.addEventListener("click", toggleSidebarMobile);
sidebarScrim.addEventListener("click", closeSidebarMobile);

// --------------------------------------------------------------------------- //
// Account header + personality profile card
// --------------------------------------------------------------------------- //
const userChipEl = document.getElementById("userChip");
const userAvatarEl = document.getElementById("userAvatar");
const userNameEl = document.getElementById("userName");
const profileCardEl = document.getElementById("profileCard");

function renderProfileCard(profile) {
  if (!profileCardEl) return;
  if (!profile) {
    profileCardEl.hidden = true;
    return;
  }
  profileCardEl.hidden = false;
  const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : "—");
  document.getElementById("pfExpertise").textContent = cap(profile.expertise_level);
  document.getElementById("pfStyle").textContent =
    profile.preferred_style === "detailed" ? "Detailed answers" : "Concise answers";
  const topics = profile.top_topics || [];
  const topicsEl = document.getElementById("pfTopics");
  topicsEl.innerHTML = topics.length
    ? topics.map((t) => `<span class="pf-chip">${escapeHtml(t)}</span>`).join("")
    : `<span class="pf-empty">learning…</span>`;
  const model = profile.preferred_model;
  document.getElementById("pfStatus").textContent =
    "✨ Personalizing your answers" + (model ? ` · best model: ${model}` : "");
}

// Who's here: a signed-in Google user, or a guest (sign-in is optional).
let ME = { guest: true, oauth_configured: false };
async function loadMe() {
  try {
    const res = await fetchWithTimeout("/api/me", {}, 15000);
    const data = await res.json();
    ME = data;
    if (data.models) MODEL_LABELS = data.models;
    const signInBtn = document.getElementById("signInBtn");
    if (data.user) {
      const u = data.user;
      if (userNameEl) userNameEl.textContent = u.name || "";
      if (userAvatarEl) {
        if (u.picture) { userAvatarEl.src = u.picture; userAvatarEl.style.display = ""; }
        else { userAvatarEl.style.display = "none"; }
      }
      if (userChipEl) userChipEl.hidden = false;
      if (signInBtn) signInBtn.hidden = true;
    } else {
      if (userChipEl) userChipEl.hidden = true;
      if (signInBtn) signInBtn.hidden = !data.oauth_configured;
    }
    renderProfileCard(data.profile);
    applyQuota(data.quota);
  } catch (_) { /* offline — the page still works for cached answers */ }
}

// Startup: load account + profile, metrics + sidebar, and reopen the current
// conversation if it has history (per-user, server-side).
loadMe();
refreshMetrics();
loadConversations();
(async function restoreCurrentConversation() {
  try {
    const res = await fetchWithTimeout(`/api/conversation/${encodeURIComponent(conversationId)}`, {}, 15000);
    const data = await res.json();
    if (data.messages && data.messages.length) {
      messagesEl.innerHTML = "";
      data.messages.forEach((m) => {
        if (m.role === "user") addUserMessage(m.content);
        else addHistoricalAnswer(m.content, m.sources);
      });
      markActiveConversation();
      scrollToBottom();
    }
  } catch (_) { /* fresh start */ }
})();
