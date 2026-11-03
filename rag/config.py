"""Runtime settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else int(raw)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else float(raw)


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chunk_max_tokens: int = _int("CHUNK_MAX_TOKENS", 512)
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", str(ROOT_DIR / "chroma_db")))
    collection_name: str = os.getenv("COLLECTION_NAME", "agentic_rag")
    retrieve_k: int = _int("RETRIEVE_K", 5)
    retrieve_candidate_k: int = _int("RETRIEVE_CANDIDATE_K", 20)
    bm25_weight: float = _float("BM25_WEIGHT", 0.5)
    dense_weight: float = _float("DENSE_WEIGHT", 0.5)
    flashrank_model: str = os.getenv("FLASHRANK_MODEL", "ms-marco-MiniLM-L-12-v2")
    max_retries: int = _int("MAX_RETRIES", 2)
    max_history_messages: int = _int("MAX_HISTORY_MESSAGES", 20)
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    checkpoint_path: str = os.getenv("CHECKPOINT_PATH", "")
    guardrails_enabled: bool = _bool("GUARDRAILS_ENABLED", True)
    guardrails_dir: Path = Path(os.getenv("GUARDRAILS_DIR", str(ROOT_DIR / "guardrails")))
    ragas_enabled: bool = _bool("RAGAS_ENABLED", True)
    ragas_min_faithfulness: float = _float("RAGAS_MIN_FAITHFULNESS", 0.0)

    @property
    def web_search_enabled(self) -> bool:
        return bool(self.tavily_api_key)


settings = Settings()
