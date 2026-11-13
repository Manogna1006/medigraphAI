"""
Centralized configuration for MediGraph AI.

All environment-dependent values are declared here via pydantic-settings so that
missing/invalid config fails fast at startup with a clear error, instead of
surfacing as a cryptic runtime exception three layers deep in a request.
"""
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_username: str = Field(default="neo4j")
    neo4j_password: str = Field(default="changeme")

    # Groq / LLM
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.1-70b-versatile")

    # Vector store
    chroma_persist_dir: str = Field(default="./chroma_data")
    chroma_collection_name: str = Field(default="medigraph_corpus")

    # Semantic firewall
    vector_distance_threshold: float = Field(default=0.35, ge=0.0, le=2.0)
    min_kg_triplets_required: int = Field(default=1, ge=0)

    # Hallucination evaluation
    hallucination_judge_model: str = Field(default="llama-3.1-70b-versatile")
    hallucination_rate_alert_threshold: float = Field(default=0.15, ge=0.0, le=1.0)

    # OpenAlex ingestion
    openalex_base_url: str = Field(default="https://api.openalex.org")
    openalex_mailto: str = Field(default="", description="Contact email for OpenAlex's polite pool (higher rate limits)")
    openalex_min_authority_score: float = Field(default=0.3, ge=0.0, le=1.0)
    openalex_max_works_per_topic: int = Field(default=50, ge=1, le=200)

    # App
    log_level: str = Field(default="INFO")
    app_env: str = Field(default="development")

    @field_validator("groq_api_key")
    @classmethod
    def warn_if_missing_key(cls, v: str) -> str:
        # Intentionally does not raise here: unit tests / offline dev must still
        # be able to import the app. Live LLM calls will fail loudly and clearly
        # at call time instead (see synthesis/llm_engine.py).
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
