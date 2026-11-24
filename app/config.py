import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- Supabase ---
    # Use the SERVICE ROLE key here, not the anon key. RLS is enabled on every
    # table in this project, so the anon key would only see rows the current
    # (unauthenticated) request is allowed to see, which is nothing useful for
    # a server-side recommender. The service role key must never be shipped to
    # the browser -- it only lives in this backend's environment.
    SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    # --- CORS ---
    # Comma-separated list of origins allowed to call this API, e.g.
    # "http://localhost:3000,https://your-app.vercel.app"
    ALLOWED_ORIGINS: list[str] = [
        o.strip()
        for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ]

    # --- Engine tuning ---
    # How much weight content-based similarity (name/description/category/tags)
    # gets vs. collaborative similarity (co-interaction patterns), 0..1.
    CONTENT_WEIGHT: float = float(os.environ.get("CONTENT_WEIGHT", "0.55"))

    # Implicit-feedback weight per interaction action.
    ACTION_WEIGHTS: dict[str, float] = {
        "purchase": 5.0,
        "save": 3.0,
        "cart": 2.0,
        "share": 1.5,
        "seen": 1.0,
    }

    # Trending score half-life, in days. Older interactions count for less.
    TRENDING_HALFLIFE_DAYS: float = float(os.environ.get("TRENDING_HALFLIFE_DAYS", "7"))

    # How often (seconds) the engine reloads data from Supabase in the
    # background, so new products/interactions show up without a restart.
    REFRESH_INTERVAL_SECONDS: int = int(os.environ.get("REFRESH_INTERVAL_SECONDS", "300"))

    # API key clients must send to call this service. Leave blank to disable.
    ENGINE_API_KEY: str = os.environ.get("ENGINE_API_KEY", "")


settings = Settings()
