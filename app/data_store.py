"""
Pulls products / interactions / product_tags / friendships from Supabase
into memory and keeps them fresh with a background refresh loop.

Kept as an in-memory store (not re-querying the DB per request) because the
catalog is small (dozens-hundreds of products). If your catalog grows into
the tens of thousands, swap the similarity computation for a proper vector
index (e.g. pgvector) instead of recomputing a dense matrix on every refresh.
"""
from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import Client, create_client

from .config import settings

logger = logging.getLogger("recommender.data_store")

PAGE_SIZE = 1000


def _paginate_all(query_fn, page_size: int = PAGE_SIZE) -> list[dict]:
    """Supabase caps rows per request; page through everything."""
    rows: list[dict] = []
    start = 0
    while True:
        resp = query_fn().range(start, start + page_size - 1).execute()
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return rows


class DataStore:
    def __init__(self) -> None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set "
                "(use the service role key, not the anon key)."
            )
        self.client: Client = create_client(
            settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY
        )

        self.products: pd.DataFrame = pd.DataFrame()
        self.interactions: pd.DataFrame = pd.DataFrame()
        self.friend_pairs: set[tuple[str, str]] = set()

        self.content_sim: Optional[np.ndarray] = None      # item-item, content-based
        self.collab_sim: Optional[np.ndarray] = None        # item-item, co-interaction based
        self.hybrid_sim: Optional[np.ndarray] = None        # blended
        self.id_to_idx: dict[str, int] = {}
        self.idx_to_id: list[str] = []

        self.user_weighted_items: dict[str, dict[str, float]] = {}  # user -> {product_id: weight}
        self.trending_scores: pd.Series = pd.Series(dtype=float)

        self.last_refresh: Optional[str] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _fetch_raw(self) -> dict[str, list[dict]]:
        products = _paginate_all(
            lambda: self.client.table("products")
            .select("id,name,description,category,price,image_url,seller_id,stock,status,is_featured,created_at")
        )
        tags = _paginate_all(
            lambda: self.client.table("product_tags").select("product_id,tag")
        )
        interactions = _paginate_all(
            lambda: self.client.table("interactions")
            .select("user_id,product_id,action,created_at")
        )
        friendships = _paginate_all(
            lambda: self.client.table("friendships")
            .select("requester_id,addressee_id,status")
            .eq("status", "accepted")
        )
        return {
            "products": products,
            "tags": tags,
            "interactions": interactions,
            "friendships": friendships,
        }

    def refresh(self) -> None:
        raw = self._fetch_raw()

        products = pd.DataFrame(raw["products"])
        if products.empty:
            logger.warning("No products returned from Supabase.")
            self.products = products
            return

        # only recommend live, in-stock-ish listings
        products = products[products["status"] == "active"].copy()

        tags_df = pd.DataFrame(raw["tags"])
        if not tags_df.empty:
            tags_by_product = (
                tags_df.groupby("product_id")["tag"]
                .apply(lambda s: " ".join(str(t) for t in s if t))
            )
        else:
            tags_by_product = pd.Series(dtype=str)

        products["tags_text"] = products["id"].map(tags_by_product).fillna("")
        products["text_blob"] = (
            products["name"].fillna("") + " " +
            products["category"].fillna("") + " " +
            products["tags_text"].fillna("") + " " +
            products["description"].fillna("")
        ).str.strip()

        self.products = products.reset_index(drop=True)
        self.id_to_idx = {pid: i for i, pid in enumerate(self.products["id"])}
        self.idx_to_id = list(self.products["id"])

        interactions = pd.DataFrame(raw["interactions"])
        # drop interactions pointing at products we no longer consider active
        if not interactions.empty:
            interactions = interactions[interactions["product_id"].isin(self.id_to_idx)]
        self.interactions = interactions.reset_index(drop=True)

        friendships = raw["friendships"]
        pairs: set[tuple[str, str]] = set()
        for f in friendships:
            a, b = f["requester_id"], f["addressee_id"]
            pairs.add((a, b))
            pairs.add((b, a))
        self.friend_pairs = pairs

        self._build_content_similarity()
        self._build_collaborative_similarity()
        self._blend_similarity()
        self._build_user_weights()
        self._build_trending()

        self.last_refresh = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Refreshed data_store: %d products, %d interactions, %d friend pairs",
            len(self.products), len(self.interactions), len(self.friend_pairs),
        )

    # ------------------------------------------------------------------ #
    # Similarity building
    # ------------------------------------------------------------------ #
    def _build_content_similarity(self) -> None:
        corpus = self.products["text_blob"].tolist()
        if len(corpus) < 2 or all(not c.strip() for c in corpus):
            n = len(corpus)
            self.content_sim = np.zeros((n, n))
            return
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        matrix = vectorizer.fit_transform(corpus)
        self.content_sim = cosine_similarity(matrix)

    def _build_collaborative_similarity(self) -> None:
        n = len(self.products)
        if self.interactions.empty or n < 2:
            self.collab_sim = np.zeros((n, n))
            return

        weights = self.interactions["action"].map(settings.ACTION_WEIGHTS).fillna(1.0)
        df = self.interactions.assign(weight=weights)
        df = df[df["product_id"].isin(self.id_to_idx)]
        df["item_idx"] = df["product_id"].map(self.id_to_idx)

        # user-item implicit matrix (sum weights if a user did multiple actions
        # on the same product)
        pivot = df.pivot_table(
            index="user_id", columns="item_idx", values="weight", aggfunc="sum", fill_value=0.0
        )
        # make sure every product column exists even if untouched
        for idx in range(n):
            if idx not in pivot.columns:
                pivot[idx] = 0.0
        pivot = pivot[sorted(pivot.columns)]

        matrix = pivot.to_numpy()
        if matrix.shape[0] == 0:
            self.collab_sim = np.zeros((n, n))
            return

        self.collab_sim = cosine_similarity(matrix.T)

    def _blend_similarity(self) -> None:
        alpha = settings.CONTENT_WEIGHT
        self.hybrid_sim = alpha * self.content_sim + (1 - alpha) * self.collab_sim
        np.fill_diagonal(self.hybrid_sim, 0.0)

    def _build_user_weights(self) -> None:
        self.user_weighted_items = {}
        if self.interactions.empty:
            return
        weights = self.interactions["action"].map(settings.ACTION_WEIGHTS).fillna(1.0)
        df = self.interactions.assign(weight=weights)
        for user_id, group in df.groupby("user_id"):
            agg = group.groupby("product_id")["weight"].sum()
            self.user_weighted_items[user_id] = agg.to_dict()

    def _build_trending(self) -> None:
        n = len(self.products)
        scores = np.zeros(n)
        if not self.interactions.empty:
            halflife = max(settings.TRENDING_HALFLIFE_DAYS, 0.1)
            decay_lambda = math.log(2) / halflife
            now = datetime.now(timezone.utc)

            weights = self.interactions["action"].map(settings.ACTION_WEIGHTS).fillna(1.0)
            df = self.interactions.assign(weight=weights)
            created = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
            age_days = (now - created).dt.total_seconds() / 86400.0
            age_days = age_days.fillna(age_days.max() if len(age_days) else 0)
            decayed = df["weight"] * np.exp(-decay_lambda * age_days.clip(lower=0))

            df = df.assign(decayed=decayed)
            for pid, val in df.groupby("product_id")["decayed"].sum().items():
                idx = self.id_to_idx.get(pid)
                if idx is not None:
                    scores[idx] = val

        # small featured-listing boost so new/curated items aren't invisible
        featured_boost = self.products["is_featured"].fillna(False).astype(bool).to_numpy() * 0.5
        scores = scores + featured_boost

        self.trending_scores = pd.Series(scores, index=self.products["id"])

    # ------------------------------------------------------------------ #
    # Background refresh loop
    # ------------------------------------------------------------------ #
    async def refresh_loop(self) -> None:
        while True:
            try:
                async with self._lock:
                    await asyncio.to_thread(self.refresh)
            except Exception:
                logger.exception("Background refresh failed; keeping previous data.")
            await asyncio.sleep(settings.REFRESH_INTERVAL_SECONDS)


store = DataStore()
