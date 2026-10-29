from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    admin_token: str = ""

    data_dir: Path = Path("data")
    sources_file: Path = Path("sources.json")
    contact_url: str = "https://www.placidway.com"

    crawler_user_agent: str = "PlacidWayKnowledgeBot/1.0 (assessment candidate; respects robots.txt)"
    crawl_delay_seconds: float = 1.0

    embed_model: str = "BAAI/bge-small-en-v1.5"
    min_similarity: float = 0.50   # grounding gate: best vector cosine similarity must reach this
    top_k: int = 6
    history_turns: int = 6
    rate_limit_per_minute: int = 20
    allowed_origins: str = "*"


@lru_cache
def get_settings() -> Settings:
    return Settings()
