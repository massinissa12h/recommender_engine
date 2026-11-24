from __future__ import annotations

import numpy as np

from .config import settings
from .data_store import DataStore
from .schemas import RecommendedProduct


def _seller_of(store: DataStore, product_id: str) -> str | None:
    idx = store.id_to_idx.get(product_id)
    if idx is None:
        return None
    return store.products.iloc[idx]["seller_id"]


def _name_of(store: DataStore, product_id: str) -> str:
    idx = store.id_to_idx.get(product_id)
    if idx is None:
        return ""
    value = store.products.iloc[idx]["name"]
    return "" if value is None else str(value)


def _round_sources(sources: dict[str, float]) -> dict[str, float]:
    """Keep only signals that actually contributed, rounded for readability."""
    return {k: round(float(v), 4) for k, v in sources.items() if float(v) > 1e-9}


def _explain(content: float, collab: float) -> str:
    """
    Plain-language "why this?" line, chosen by whichever signal dominated.
    This is what the card renders under the product name.
    """
    if content <= 1e-9 and collab <= 1e-9:
        return "A fresh pick from the catalogue"
    if content >= collab * 1.5:
        return "Similar to items you've been looking at"
    if collab >= content * 1.5:
        return "Popular with shoppers who like what you like"
    return "Matches your taste and what similar shoppers bought"


# --------------------------------------------------------------------------- #
# Trending (also the cold-start / logged-out fallback)
# --------------------------------------------------------------------------- #
def get_trending(
    store: DataStore, n: int, exclude: set[str] | None = None
) -> list[RecommendedProduct]:
    exclude = exclude or set()
    if store.trending_scores.empty:
        return []

    ranked = store.trending_scores.sort_values(ascending=False)
    out: list[RecommendedProduct] = []
    for pid, score in ranked.items():
        if pid in exclude:
            continue
        out.append(
            RecommendedProduct(
                item_id=pid,
                name=_name_of(store, pid),
                score=float(score),
                sources=_round_sources({"trending": float(score)}),
                explanation="Trending right now",
                personalised=False,
            )
        )
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# Homepage / "Picked for you"
# --------------------------------------------------------------------------- #
def get_homepage(
    store: DataStore, user_id: str | None, n: int
) -> tuple[list[RecommendedProduct], str]:
    """Returns (recs, mode) where mode is 'personalized' or 'trending'."""
    if store.hybrid_sim is None or len(store.products) == 0:
        return [], "trending"

    if not user_id:
        return get_trending(store, n), "trending"

    user_items = store.user_weighted_items.get(user_id)
    if not user_items:
        # cold start: no interaction history yet
        return get_trending(store, n), "trending"

    n_items = len(store.products)

    # Weight vector over the catalogue: how much this user has engaged with
    # each item (purchase 5 > save 3 > cart 2 > share 1.5 > seen 1).
    weights = np.zeros(n_items, dtype=float)
    for pid, weight in user_items.items():
        idx = store.id_to_idx.get(pid)
        if idx is not None:
            weights[idx] += float(weight)

    # Score every item by similarity to what the user already engaged with,
    # keeping the two signals separate so we can explain the result.
    # Subtracting the diagonal term matches hybrid_sim, whose diagonal is zeroed.
    content_scores = weights @ store.content_sim - weights * np.diag(store.content_sim)
    collab_scores = weights @ store.collab_sim - weights * np.diag(store.collab_sim)

    alpha = settings.CONTENT_WEIGHT
    total_scores = alpha * content_scores + (1.0 - alpha) * collab_scores

    # don't recommend back items the user already engaged with -- they've seen
    # or bought them, showing them again as a "pick" isn't useful
    already_seen = {pid for pid in user_items if pid in store.id_to_idx}

    ranked_idx = np.argsort(-total_scores)

    out: list[RecommendedProduct] = []
    for idx in ranked_idx:
        idx = int(idx)
        pid = store.idx_to_id[idx]
        if pid in already_seen:
            continue
        if _seller_of(store, pid) == user_id:
            continue
        if total_scores[idx] <= 0:
            break

        content_part = alpha * float(content_scores[idx])
        collab_part = (1.0 - alpha) * float(collab_scores[idx])
        out.append(
            RecommendedProduct(
                item_id=pid,
                name=_name_of(store, pid),
                score=float(total_scores[idx]),
                sources=_round_sources(
                    {"content": content_part, "collaborative": collab_part}
                ),
                explanation=_explain(content_part, collab_part),
                personalised=True,
            )
        )
        if len(out) >= n:
            break

    if len(out) < n:
        # top off with trending items not already included / already seen
        have = {r.item_id for r in out} | already_seen
        out.extend(get_trending(store, n - len(out), exclude=have))

    return out, "personalized"


# --------------------------------------------------------------------------- #
# "Because you viewed X"
# --------------------------------------------------------------------------- #
def get_item_recs(
    store: DataStore, user_id: str | None, item_id: str, n: int
) -> list[RecommendedProduct]:
    idx = store.id_to_idx.get(item_id)
    if idx is None or store.hybrid_sim is None:
        return []

    anchor_name = _name_of(store, item_id)
    alpha = settings.CONTENT_WEIGHT

    hybrid_row = store.hybrid_sim[idx]
    content_row = store.content_sim[idx]
    collab_row = store.collab_sim[idx]

    # don't suggest things this user has already engaged with
    seen = set((store.user_weighted_items.get(user_id) or {}).keys()) if user_id else set()

    out: list[RecommendedProduct] = []
    for j in np.argsort(-hybrid_row):
        j = int(j)
        if j == idx:
            continue
        if hybrid_row[j] <= 0:
            break
        pid = store.idx_to_id[j]
        if pid in seen:
            continue
        if user_id and _seller_of(store, pid) == user_id:
            continue

        content_part = alpha * float(content_row[j])
        collab_part = (1.0 - alpha) * float(collab_row[j])
        out.append(
            RecommendedProduct(
                item_id=pid,
                name=_name_of(store, pid),
                score=float(hybrid_row[j]),
                sources=_round_sources(
                    {"content": content_part, "collaborative": collab_part}
                ),
                explanation=(
                    f"Similar to {anchor_name}" if anchor_name else "Similar to an item you viewed"
                ),
                personalised=bool(user_id),
            )
        )
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# Friends
# --------------------------------------------------------------------------- #
def get_friends_recs(store: DataStore, user_id: str, n: int) -> list[RecommendedProduct]:
    """
    Products this user's accepted friends have purchased or saved recently.
    The frontend currently queries this directly from Supabase; this endpoint
    exists so the logic can be centralised in the engine instead.
    """
    friend_ids = {b for (a, b) in store.friend_pairs if a == user_id}
    if not friend_ids or store.interactions.empty:
        return []

    df = store.interactions[
        store.interactions["user_id"].isin(friend_ids)
        & store.interactions["action"].isin(["purchase", "save"])
    ]
    if df.empty:
        return []

    mine = set((store.user_weighted_items.get(user_id) or {}).keys())

    weights = df["action"].map({"purchase": 2.0, "save": 1.0}).fillna(1.0)
    scored = (
        df.assign(weight=weights)
        .groupby("product_id")["weight"]
        .sum()
        .sort_values(ascending=False)
    )
    friend_counts = df.groupby("product_id")["user_id"].nunique()

    out: list[RecommendedProduct] = []
    for pid, score in scored.items():
        if pid not in store.id_to_idx or pid in mine:
            continue
        count = int(friend_counts.get(pid, 1))
        out.append(
            RecommendedProduct(
                item_id=pid,
                name=_name_of(store, pid),
                score=float(score),
                sources=_round_sources({"social": float(score)}),
                explanation=(
                    "Loved by one of your friends"
                    if count == 1
                    else f"Loved by {count} of your friends"
                ),
                personalised=True,
            )
        )
        if len(out) >= n:
            break
    return out
