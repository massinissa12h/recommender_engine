import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import engine
from .config import settings
from .data_store import store
from .schemas import HealthResponse, RecommendedProduct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommender")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load data once before serving traffic, then keep it fresh in the background.
    await asyncio.to_thread(store.refresh)
    task = asyncio.create_task(store.refresh_loop())
    yield
    task.cancel()


app = FastAPI(title="For You Recommendation Engine", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _check_api_key(x_api_key: str | None) -> None:
    if settings.ENGINE_API_KEY and x_api_key != settings.ENGINE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        products_loaded=len(store.products),
        interactions_loaded=len(store.interactions),
        users_with_history=len(store.user_weighted_items),
        last_refresh=store.last_refresh,
    )


@app.get("/recommend/homepage", response_model=list[RecommendedProduct])
async def recommend_homepage(
    user_id: str | None = Query(default=None),
    n: int = Query(default=8, ge=1, le=50),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    recs, _mode = engine.get_homepage(store, user_id, n)
    return recs


@app.get("/recommend/item", response_model=list[RecommendedProduct])
async def recommend_item(
    item_id: str = Query(...),
    user_id: str | None = Query(default=None),
    n: int = Query(default=6, ge=1, le=50),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    # An unknown/inactive item isn't a caller error -- the product may have been
    # archived since the last refresh. Return nothing so the "Because you
    # viewed..." row renders empty instead of surfacing an error to the user.
    if item_id not in store.id_to_idx:
        return []
    return engine.get_item_recs(store, user_id, item_id, n)


@app.get("/recommend/friends", response_model=list[RecommendedProduct])
async def recommend_friends(
    user_id: str = Query(...),
    n: int = Query(default=8, ge=1, le=50),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    return engine.get_friends_recs(store, user_id, n)


@app.post("/admin/refresh")
async def force_refresh(x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key)
    await asyncio.to_thread(store.refresh)
    return {"status": "refreshed", "last_refresh": store.last_refresh}
