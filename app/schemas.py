from typing import Optional

from pydantic import BaseModel, Field, model_validator


class RecommendedProduct(BaseModel):
    """
    One scored recommendation.

    The frontend (lib/recommender.ts -> hydrateRecommendations) reads
    `item_id`, `name`, `score`, `sources` and `explanation`, then re-fetches
    the full product row (price/image/etc.) from Supabase so cards always show
    live data.

    `sources` is the "why this?" breakdown: how much each signal contributed to
    the score (content similarity, collaborative signal, trending, social).

    `id` and `reason` are mirrors of `item_id` and `explanation`, kept so that
    any client written against the earlier {id, score, reason} shape keeps
    working without changes.
    """

    item_id: str
    name: str = ""
    score: float
    sources: dict[str, float] = Field(default_factory=dict)
    explanation: str = ""
    personalised: bool = False

    # --- backwards-compatible mirrors, filled automatically ---
    id: str = ""
    reason: str = ""

    @model_validator(mode="after")
    def _mirror_legacy_fields(self) -> "RecommendedProduct":
        if not self.id:
            self.id = self.item_id
        if not self.reason:
            self.reason = self.explanation
        return self


class HealthResponse(BaseModel):
    status: str
    products_loaded: int
    interactions_loaded: int
    users_with_history: int
    last_refresh: Optional[str] = None
