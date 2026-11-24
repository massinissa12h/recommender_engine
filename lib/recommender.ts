import { supabase } from './supabaseClient'

const RECOMMENDER_URL =
  process.env.NEXT_PUBLIC_RECOMMENDER_URL ?? 'http://localhost:8000'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Lightweight result from the FastAPI engine: just an id + score + reason. */
export interface RecommendedProduct {
  id: string
  score: number
  reason?: string
}

/** A RecommendedProduct filled in with the actual product row from Supabase. */
export interface HydratedProduct extends RecommendedProduct {
  name: string
  description: string | null
  category: string | null
  price: number
  image_url: string | null
  seller_id: string | null
  is_featured: boolean
}

// ---------------------------------------------------------------------------
// Engine calls (FastAPI, :8000 locally)
// ---------------------------------------------------------------------------

async function engineFetch(path: string, params: Record<string, string>, signal?: AbortSignal) {
  const qs = new URLSearchParams(params)
  let res: Response
  try {
    res = await fetch(`${RECOMMENDER_URL}${path}?${qs}`, { signal })
  } catch (e) {
    // Normalize browser fetch errors ("Failed to fetch", "NetworkError when
    // attempting to fetch resource", etc.) into a message describeEngineError
    // in page.tsx knows how to recognize as "the engine is unreachable".
    if ((e as Error)?.name === 'AbortError') throw e
    throw new Error(`fetch failed: ${(e as Error)?.message ?? 'unknown network error'}`)
  }
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`)
  }
  return res.json()
}

export async function fetchHomepage({
  userId,
  n,
  signal,
}: {
  userId?: string
  n: number
  signal?: AbortSignal
}): Promise<RecommendedProduct[]> {
  const params: Record<string, string> = { n: String(n) }
  if (userId) params.user_id = userId
  return engineFetch('/recommend/homepage', params, signal)
}

export async function fetchRecommendForItem({
  userId,
  itemId,
  n,
  signal,
}: {
  userId?: string
  itemId: string
  n: number
  signal?: AbortSignal
}): Promise<RecommendedProduct[]> {
  const params: Record<string, string> = { item_id: itemId, n: String(n) }
  if (userId) params.user_id = userId
  return engineFetch('/recommend/item', params, signal)
}

// ---------------------------------------------------------------------------
// Hydration: turn engine ids into full product cards via Supabase
// ---------------------------------------------------------------------------

const PRODUCT_COLUMNS =
  'id,name,description,category,price,image_url,seller_id,is_featured,status'

export async function hydrateRecommendations(
  recs: RecommendedProduct[],
): Promise<HydratedProduct[]> {
  if (recs.length === 0) return []

  const ids = recs.map((r) => r.id)
  const { data, error } = await supabase
    .from('products')
    .select(PRODUCT_COLUMNS)
    .in('id', ids)
    .eq('status', 'active')

  if (error) throw new Error(`Supabase hydrate error: ${error.message}`)

  const byId = new Map((data ?? []).map((p) => [p.id as string, p]))

  // preserve the engine's ranking order; silently drop ids that no longer
  // exist or aren't active (deleted/archived since the engine last refreshed)
  const hydrated: HydratedProduct[] = []
  for (const rec of recs) {
    const p = byId.get(rec.id)
    if (!p) continue
    hydrated.push({
      ...rec,
      name: p.name,
      description: p.description ?? null,
      category: p.category ?? null,
      price: Number(p.price ?? 0),
      image_url: p.image_url ?? null,
      seller_id: p.seller_id ?? null,
      is_featured: Boolean(p.is_featured),
    })
  }
  return hydrated
}

// ---------------------------------------------------------------------------
// Friends-liked and recent-views: queried directly from Supabase, since
// they're social/history reads rather than model-scored recommendations.
// ---------------------------------------------------------------------------

export async function fetchFriendsLiked(
  userId: string,
  n: number,
): Promise<HydratedProduct[]> {
  const { data: friendRows, error: friendErr } = await supabase
    .from('friendships')
    .select('requester_id,addressee_id')
    .eq('status', 'accepted')
    .or(`requester_id.eq.${userId},addressee_id.eq.${userId}`)

  if (friendErr) throw new Error(`Supabase friendships error: ${friendErr.message}`)

  const friendIds = Array.from(
    new Set(
      (friendRows ?? []).map((f) =>
        f.requester_id === userId ? f.addressee_id : f.requester_id,
      ),
    ),
  )
  if (friendIds.length === 0) return []

  const { data: interactionRows, error: interErr } = await supabase
    .from('interactions')
    .select('product_id,action,created_at')
    .in('user_id', friendIds)
    .in('action', ['purchase', 'save'])
    .order('created_at', { ascending: false })
    .limit(200) // wide net; we'll dedupe + rank below, then trim to n

  if (interErr) throw new Error(`Supabase interactions error: ${interErr.message}`)

  const weight: Record<string, number> = { purchase: 2, save: 1 }
  const scoreByProduct = new Map<string, number>()
  for (const row of interactionRows ?? []) {
    const w = weight[row.action as string] ?? 1
    scoreByProduct.set(row.product_id, (scoreByProduct.get(row.product_id) ?? 0) + w)
  }

  const ranked: RecommendedProduct[] = Array.from(scoreByProduct.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([id, score]) => ({ id, score, reason: 'Loved by your friends' }))

  return hydrateRecommendations(ranked)
}

export async function fetchRecentViews(
  userId: string,
  n: number,
): Promise<HydratedProduct[]> {
  const { data, error } = await supabase
    .from('interactions')
    .select('product_id,created_at')
    .eq('user_id', userId)
    .eq('action', 'seen')
    .order('created_at', { ascending: false })
    .limit(50) // fetch extra so we can dedupe down to n distinct products

  if (error) throw new Error(`Supabase recent views error: ${error.message}`)

  const seen = new Set<string>()
  const recs: RecommendedProduct[] = []
  for (const row of data ?? []) {
    if (seen.has(row.product_id)) continue
    seen.add(row.product_id)
    recs.push({ id: row.product_id, score: 0 })
    if (recs.length >= n) break
  }

  return hydrateRecommendations(recs)
}
