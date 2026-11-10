# SearchIQ

**Sourced answers for business and competitive research.** SearchIQ is for analysts researching companies, markets, earnings and deals. Every answer is grounded in live web search and cites its sources inline. That shows you which facts are backed and safe to put in front of executives, and which to double-check before they go into a brief.

Every query gets **two drafts (A/B)**: one concise, one detailed. Pick the better one and say why. SearchIQ turns that into lessons, rewards and a learned per-user policy, so answers drift toward how you like your briefs. It tracks a North Star metric, **Task Completion Rate**.

It runs at **$0** on free tiers:

| Piece | Service | Free-tier limit | SearchIQ's own cap |
|---|---|---|---|
| Hosting | Vercel Hobby (static `public/` + one Python function) | 300s per call | `maxDuration: 120` |
| LLM | Groq: `openai/gpt-oss-20b` (fast), `openai/gpt-oss-120b` (deep) | ~200K tokens/day per model | 150K tokens/day per model |
| Web search | Tavily basic search (1 credit) | 1,000 credits/month | 25/day, 900/month |
| Database | Turso (libSQL over HTTP) | free plan | n/a |

> Groq retired `llama-3.1-8b-instant` for free keys on 2026-08-16. `openai/gpt-oss-20b` is Groq's recommended replacement. You can override either model with `GROQ_FAST_MODEL` / `GROQ_DEEP_MODEL`.

## How a query flows

1. **Cache.** The first turn of a conversation is served from the answer cache if possible. That's free and doesn't count against any cap, and the UI shows "Cached answer from <date>". Time-sensitive queries (today, latest, current, news, prices, …) skip the cache. The three example questions ("Who founded Nvidia, and in what year?", "What was Apple's total net sales in fiscal year 2023?", "When did Microsoft complete its acquisition of Activision Blizzard?") are pinned in the cache, so they're always served cached.
2. **Clarify.** A vague query ("best tools") gets one clarifying question. No search is spent.
3. **Caps.** Guests get 3 live searches/day per cookie, plus a cap of 10/day per hashed IP, so clearing cookies doesn't reset it. Signed-in users get 10/day. There's also a site-wide daily/monthly cap and a per-model Groq token budget. Any limit, including a real 429 from Groq or Tavily, shows a friendly limit card with cached examples. It's never an error page.
4. **Live.** One Tavily search runs, then Groq drafts A (concise) and B (detailed) in parallel from the same results. Simple/medium queries use GPT-OSS 20B. Complex queries split A = GPT-OSS 20B, B = GPT-OSS 120B.
5. **Check (v2 agent).** One more GPT-OSS 20B call (~2–3K tokens) checks both drafts against the numbered sources. Each key claim is labelled **✓ Backed [n]**, **⚠ Double-check** (not in the sources) or **✗ Contradicted [n]**. Each answer gets a confidence label (High/Medium/Low), computed from those labels rather than asked of the model. A false premise in the question is flagged above the answers, and a **Where A and B disagree** panel shows which side the sources support. A **steps panel** shows what ran (search → draft → check). If the check fails or hits a limit, the answers still show, just without labels. The check is cached with the answer.
6. **Learn.** Pick a winner, then give a reason. That produces rewards, lessons and a policy. A 👎 retry reuses the original sources, so no new search is spent.

Guests can do everything. Google sign-in is optional and adds a personality profile that personalizes answers.

## Deploy to Vercel

1. Create free accounts at Groq, Tavily and Turso. **Don't add a payment card**; free tiers then hard-stop instead of billing you.
2. Create a Turso database (`turso db create searchiq`), then get its URL (`turso db show searchiq --url`) and a token (`turso db tokens create searchiq`).
3. Import this repo in Vercel. Under **Settings → Environment Variables**, set:
   - `GROQ_API_KEY`, `TAVILY_API_KEY`
   - `TURSO_DATABASE_URL`, `TURSO_AUTH_TOKEN`
   - `FLASK_SECRET_KEY` (required): generate with `python3 -c "import secrets; print(secrets.token_hex(32))"`
   - Optional: `OPENROUTER_API_KEY`, a prepaid key used only to rewrite flagged answers (pinned to Groq, no fallbacks). If it's missing or out of credit, rewrites use the free Groq key and wait up to 30s for capacity.
   - Optional: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and any cap from `.env.example`
4. Google sign-in (optional): add `https://<app>.vercel.app/auth/callback` as an authorized redirect URI, and **publish the OAuth consent screen** ("In production", not "Testing"). In Testing mode, only listed test users can sign in.

Keys live **only** in Vercel's environment variables. `.env` is gitignored, and `.vercelignore` keeps local data out of uploads.

## Local development

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
cp .env.example .env        # add keys, or set SEARCHIQ_FAKE_PROVIDERS=1 for canned answers
.venv/bin/python app.py     # http://127.0.0.1:5001  (uses searchiq_local.db)
.venv/bin/python -m pytest -q tests
SEARCHIQ_TEST_TURSO=1 .venv/bin/python -m pytest -q tests   # same suite via the Turso HTTP adapter
```

The CLI still works: `python search_agent.py`.

## Files

- `app.py`: Flask app and the Vercel entrypoint. Holds the JSON/NDJSON API, guest/Google identity and the limit card.
- `agent.py`: the v2 claim check (prompt, normalization, confidence labels).
- `search_agent.py`: routing, A/B generation, lessons, rewards, policy, profiles, metrics.
- `providers.py`: Groq and Tavily clients. A 429 is raised as `ProviderLimit`.
- `limits.py`: caps, token budgets, answer cache.
- `db.py`: Turso HTTP connection (sqlite3-compatible), with a SQLite fallback.
- `public/static/`: frontend, served by Vercel's CDN. `templates/`: HTML.
- `archive/` (gitignored): the pre-migration Railway data, kept locally and not deployed.
