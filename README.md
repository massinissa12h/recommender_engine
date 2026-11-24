# For You — Recommendation Engine

A small FastAPI service that powers the "For You" page of your Next.js +
Supabase store. It reads `products`, `product_tags`, `interactions`, and
`friendships` from Supabase, and serves:

- `GET /recommend/homepage?user_id=&n=` — personalized picks (falls back to
  trending for logged-out users or users with no history yet)
- `GET /recommend/item?item_id=&user_id=&n=` — "because you viewed X"
- `GET /recommend/friends?user_id=&n=` — bonus endpoint: products your
  accepted friends purchased/saved (your frontend may already do this
  directly against Supabase — use this only if you want to centralize it)
- `GET /health` — status + row counts, for uptime checks
- `POST /admin/refresh` — force an immediate data reload

It blends **content-based** similarity (TF-IDF over name/description/
category/tags) with **collaborative** similarity (co-interaction patterns
from `interactions`, weighted by action: purchase > save > cart > share >
seen), plus a recency-decayed **trending** score. Data is loaded into memory
on startup and refreshed automatically every 5 minutes (configurable), so it
stays fast without hitting Postgres on every request.

## 1. Local setup

```bash
cd recommender-engine
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
```

Edit `.env`:

```
SUPABASE_URL=https://ubenriftfsfwbejriomn.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<paste from Supabase dashboard>
ALLOWED_ORIGINS=http://localhost:3000
```

Get the service role key from your Supabase dashboard: **Project Settings →
API → service_role key**. Use the *service role* key, not `anon` — every
table here has RLS enabled, so the anon key can't read across users. Never
put the service role key in your Next.js app or any browser-exposed `.env` —
it only belongs in this backend.

Run it:

```bash
uvicorn app.main:app --reload --port 8000
```

Check it's alive:

```bash
curl http://localhost:8000/health
curl "http://localhost:8000/recommend/homepage?n=5"
```

The first request after startup may take a second or two while it loads and
vectorizes your catalog — that's expected, everything after is served from
memory.

## 2. Wire it into your Next.js app

Your `page.tsx` already expects `lib/recommender.ts` to call this service and
shows "Recommender offline. Start the FastAPI service on :8000" when it can't
reach it — that error message tells us the contract already assumes exactly
this shape (a base URL + these two query params). In `lib/recommender.ts`,
point the base URL at an environment variable instead of hardcoding
`localhost:8000`:

```ts
const RECOMMENDER_URL = process.env.NEXT_PUBLIC_RECOMMENDER_URL ?? 'http://localhost:8000'

export async function fetchHomepage({ userId, n, signal }: { userId?: string; n: number; signal?: AbortSignal }) {
  const params = new URLSearchParams({ n: String(n) })
  if (userId) params.set('user_id', userId)
  const res = await fetch(`${RECOMMENDER_URL}/recommend/homepage?${params}`, { signal })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<RecommendedProduct[]>
}

export async function fetchRecommendForItem({ userId, itemId, n }: { userId?: string; itemId: string; n: number }) {
  const params = new URLSearchParams({ item_id: itemId, n: String(n) })
  if (userId) params.set('user_id', userId)
  const res = await fetch(`${RECOMMENDER_URL}/recommend/item?${params}`)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<RecommendedProduct[]>
}
```

If you already have a `lib/recommender.ts` with different code, just update
its base URL constant to read from `NEXT_PUBLIC_RECOMMENDER_URL` — share that
file with me and I'll align it exactly instead of guessing.

Add to your Next.js `.env.local`:

```
NEXT_PUBLIC_RECOMMENDER_URL=http://localhost:8000
```

## 3. Deploy it so it's always on

The engine is a plain Docker container, so any of these work. **Render** is
the simplest if you want a click-and-go setup:

### Option A: Render (easiest)

1. Push the `recommender-engine/` folder to a GitHub repo (or a subfolder of
   your existing repo).
2. In the [Render dashboard](https://dashboard.render.com) → **New → Web
   Service** → connect that repo.
3. Render auto-detects the `Dockerfile`. Set:
   - **Instance type**: at least the paid "Starter" tier if you want the
     service to stay warm 24/7 — Render's **free tier spins down after ~15
     minutes idle** and takes ~30s to cold-start on the next request, which
     will make your "For You" page feel slow the first time someone loads it
     after a lull.
4. Add environment variables (from your `.env`): `SUPABASE_URL`,
   `SUPABASE_SERVICE_ROLE_KEY`, `ALLOWED_ORIGINS` (set this to your deployed
   Next.js domain, e.g. `https://your-app.vercel.app`), and optionally
   `ENGINE_API_KEY`.
5. Deploy. Render gives you a URL like `https://your-engine.onrender.com`.

### Option B: Railway

Same idea — `New Project → Deploy from GitHub`, it picks up the Dockerfile,
add the same environment variables. Railway's usage-based pricing keeps
services warm without Render's free-tier sleep behavior.

### Option C: Fly.io

```bash
fly launch          # detects the Dockerfile, creates fly.toml
fly secrets set SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... ALLOWED_ORIGINS=...
fly deploy
```

Fly's smallest always-on VM is inexpensive and doesn't sleep, which suits a
small always-running recommender well.

### After deploying

1. Copy the live URL Render/Railway/Fly gives you.
2. In your Next.js project's Vercel dashboard (or wherever it's hosted), set
   `NEXT_PUBLIC_RECOMMENDER_URL=https://your-engine.onrender.com`.
3. Redeploy the frontend so it picks up the new env var.
4. Double check `ALLOWED_ORIGINS` on the engine includes your production
   frontend's exact domain — CORS will silently block the browser requests
   otherwise (you'll see it in the browser console, not as a friendly error).

## 4. Keeping recommendations fresh

- The in-memory data refreshes automatically every `REFRESH_INTERVAL_SECONDS`
  (default 300s / 5 min). New products or interactions show up within that
  window without a restart.
- Call `POST /admin/refresh` (with the `X-API-Key` header if you set
  `ENGINE_API_KEY`) to force an immediate reload right after a bulk import.
- If your catalog grows past a few thousand products, the dense similarity
  matrix approach here will get slow/memory-heavy — at that point move to an
  approximate nearest-neighbor index (e.g. `pgvector` in Supabase itself, or
  `faiss`) instead of recomputing a full cosine-similarity matrix on refresh.

## 5. Security notes

- `SUPABASE_SERVICE_ROLE_KEY` bypasses Row Level Security. Keep it only in
  this backend's environment variables — never in frontend code, never
  committed to git (`.env` is already in `.gitignore`).
- Set `ENGINE_API_KEY` once you deploy publicly, and have your Next.js
  backend (an API route, not client-side code) attach it as an `X-API-Key`
  header, so the recommender isn't a fully open public endpoint. If
  `lib/recommender.ts` calls this service directly from the browser, the key
  would be visible to users anyway — in that case rely on `ALLOWED_ORIGINS`
  CORS restriction instead, or proxy the calls through a Next.js API route
  that holds the key server-side.
