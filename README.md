# Search Evaluation Agent

A command-line agent that answers search queries with Claude, learns from
thumbs up/down feedback, and tracks a North Star Metric (Task Completion Rate).

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

### Web UI (recommended for demos)

```bash
python app.py
# open http://127.0.0.1:5000
```

A dark-theme single-page app: a chat interface on the left, a live North Star
dashboard on the right (Task Completion Rate as a big number, total/rated/improved
counts, a recent-feedback bar chart, and a "most improved" list). Thumbs-down
triggers a visible "🧠 Learning…" state while the agent rephrases and retries.
Use **Reset demo** in the top bar to start from a clean slate before a presentation.

The web app reuses the same `search_agent.py` backend and `search_agent.db`.

### CLI

```bash
python search_agent.py
```

## Retrieval-Augmented Generation (web search)

The agent is instructed to run Anthropic's built-in **web search tool**
(`web_search_20260209`) on **every** query before answering — it grounds each
response in live results rather than relying on training memory alone, then
blends in its own knowledge for context. Answers cite their sources:

> Enforcement note: "always search" is enforced via a hard system-prompt mandate
> (kept alongside adaptive thinking), not `tool_choice`. The API does not allow
> forced `tool_choice` together with extended/adaptive thinking, and the web
> search docs steer "always search" through the system prompt — so this is the
> reliable mechanism that preserves reasoning quality and won't error.

- **Web UI** — a "🔗 Sources" section appears under each answer with clickable links.
- **Terminal** — sources are printed as a numbered list under the answer.

The retrieval logic lives in one shared function (`search_agent.search_and_answer`)
used by both the CLI and the web app, so behavior is identical in both.

## What it does

1. **Query in** — you type a search query at the prompt.
2. **Search + answer out** — Claude searches the web when the query needs current
   info, then streams a direct answer grounded in those results (with sources).
3. **Feedback** — you give a 👍 / 👎 and an optional comment.
4. **Storage** — every query, answer, and piece of feedback lands in
   `search_agent.db` (SQLite).
5. **Learning** —
   - A 👎 triggers an automatic **rephrase-and-retry** (up to 2 rounds): Claude
     rewrites the query using your comment and the failed answer, then tries again.
   - Recent 👎 comments are fed back into the system prompt as *lessons learned*,
     so future answers avoid repeating the same mistakes.
6. **North Star Metric** — **Task Completion Rate** = % of rated queries that
   eventually earned a 👍.
7. **Dashboard** — printed automatically every 5 queries (and on exit), showing
   total queries, completion rate, and most-improved queries (👎 → 👍 after retry).

## Commands at the prompt

- any text — run it as a search query
- `dashboard` / `stats` — show the dashboard on demand
- `quit` / `exit` / `q` — exit (prints final stats)

## Data model

One table, `interactions`, with one row per attempt. Retries of the same query
share a `query_group`, so the dashboard can measure per-query completion and
detect improvements across attempts.
