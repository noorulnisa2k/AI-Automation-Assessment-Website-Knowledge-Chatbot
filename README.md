# PlacidWay Knowledge Chatbot

A **citation-first, refuse-by-default** RAG assistant. It answers patient questions only from seven PlacidWay pages, links the page each answer came from, and says "I don't know" (with a free-quote pointer) instead of guessing.

> **Pitch in one line:** *"A small chatbot you can trust: every answer is traceable to a PlacidWay page, every price is verified against the source text in code, and anything it can't prove, it declines."*

## Design choices (and why)

| Decision | Choice | Why |
|---|---|---|
| Approach | Classic RAG, not agent / live browsing | Code, not the LLM, controls the evidence; deterministic guardrails; respects 1 req/s and robots.txt |
| LLM | OpenAI `gpt-4o-mini` (configurable) | Reliable JSON output, low cost, strong multilingual |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (local, ONNX) | No per-query embedding cost, no torch, small RAM footprint |
| Vector DB | ChromaDB (persistent) + BM25 | Hybrid search: embeddings for meaning, BM25 for exact terms and figures |
| API / UI | FastAPI + one static HTML/CSS page | No build step, fast to review and explain |

## How a question is answered

1. **Rewrite**: the LLM turns the message plus chat history into a standalone English query and detects the user's language (this powers follow-ups and Spanish/Arabic questions over English pages).
2. **Hybrid retrieve**: vector + BM25, fused with Reciprocal Rank Fusion.
3. **Grounding gate**: if the best similarity is below `MIN_SIMILARITY`, the answering model is **never called**; a fixed "not in our content, request a free quote" reply is returned (translated). Health-related questions bypass the gate so the model can refuse to give medical advice properly.
4. **Generate**: strict system prompt, JSON output (`type`, `answer`, `sources`).
5. **Post-checks in code**: cited URLs must be among the retrieved pages; **every money figure in the answer must appear in the retrieved text**; failure triggers the safe fallback. This is what defeats "I heard it costs $50. Confirm that."
6. **Extras**: one-time medical disclaimer per session, lead capture form, unanswered-question log.

## Ingestion

`app/ingest.py` fetches only the URLs in `sources.json`, with a clear User-Agent, robots.txt check, and ~1 request/second. Text is extracted with headings preserved, tables kept as `a | b | c` rows, and chunked by heading (max ~1200 chars; tables and lists are not split). Every chunk carries URL, title, section and page type. Pages are hashed on cleaned text, so a refresh only re-embeds pages that changed.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # set OPENAI_API_KEY and ADMIN_TOKEN
uvicorn app.main:app --reload   # first boot builds the index (about a minute)
# open http://localhost:8000
python -m pytest                # unit tests for guards and chunking
python -m tests.run_questions   # runs the 8 test questions -> TEST_RESULTS.md
```

## Refresh content

- HTTP: `curl -X POST "$URL/admin/refresh" -H "X-Admin-Token: $ADMIN_TOKEN"` (add `?force=true` to re-embed everything), then `GET /admin/status`.
- CLI: `python -m scripts.refresh [--force]`

## Admin endpoints (header `X-Admin-Token`)

`POST /admin/refresh`, `GET /admin/status`, `GET /admin/unanswered` (content gaps), `GET /admin/leads`.

## Configuration

See `.env.example`. Key knobs: `MIN_SIMILARITY` (grounding gate; tune by running `tests/run_questions.py` and checking `/admin/unanswered`), `OPENAI_MODEL`, `RATE_LIMIT_PER_MINUTE`. No secrets live in code.

## Deploy on Render (with a Namecheap domain)

1. Push this folder to a GitHub repo (`.env` and `data/` are git-ignored).
2. Render dashboard → **New → Blueprint** → select the repo. Render reads `render.yaml` (Docker, Starter plan, 1 GB disk at `/data`).
3. When prompted, set `OPENAI_API_KEY`. `ADMIN_TOKEN` is generated automatically (copy it from the service's Environment tab). Also set `CRAWLER_USER_AGENT` with your real contact email.
4. Deploy. Watch the logs for the first-boot index build, then open `https://<service>.onrender.com/healthz` (it shows the chunk count) and the chat page.
5. **Custom domain**: Service → **Settings → Custom Domains → Add** `bot.yourdomain.com`.
6. **Namecheap**: Domain List → **Manage → Advanced DNS** → add `CNAME Record`, Host `bot`, Value `<service>.onrender.com`, TTL Automatic. Delete any conflicting parking or URL-redirect record for the same host.
7. Back in Render click **Verify**. HTTPS is issued automatically once DNS propagates (minutes to a few hours). For a bare domain (`yourdomain.com`) use the values Render displays in the dashboard, typically an A record on Host `@`, and add a `www` CNAME as well.
8. Keep it up for 7+ days: use the **Starter** plan (the free plan sleeps after 15 minutes idle and would look broken to a reviewer). Optionally add an uptime monitor on `/healthz`.

## Known limitations

- Extraction assumes server-rendered HTML; if a page is JS-rendered the ingest log warns (a Playwright fallback would fix it).
- The money check validates figures, not their meaning; a wrong-but-present number could still be quoted, which is why the prompt forces qualifiers and the page link.
- Medical-question detection is regex plus prompt rules, not a classifier.
- In-memory rate limiting and a single worker: fine for one instance, not for horizontal scaling.
- Only 7 pages: the similarity threshold is tuned by hand, not learned.

## With another week

Playwright fallback, a labelled eval set (retrieval recall and refusal precision) with a CI gate, a cross-encoder reranker, an LLM-as-judge faithfulness check on every answer, Redis rate limiting, streaming responses, and an agent for multi-step comparison questions.
