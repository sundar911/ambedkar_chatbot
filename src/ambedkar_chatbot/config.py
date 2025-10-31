"""Configuration helpers for the Ambedkar chatbot project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = PROJECT_ROOT / "Ambedkar_Writings"
INDEX_FILE = DATA_DIR / "ambedkar_index.ann"
METADATA_FILE = DATA_DIR / "ambedkar_metadata.jsonl"
INDEX_INFO_FILE = DATA_DIR / "ambedkar_index_info.json"


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    embedding_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    completion_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "320"))  # number of words per chunk
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "60"))  # word overlap between chunks
    batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "32"))
    top_k: int = int(os.getenv("TOP_K", "6"))
    persona_temperature: float = float(os.getenv("CHAT_TEMPERATURE", "0.6"))

    def ensure_api_key(self) -> str:
        """Return the API key or raise a helpful error."""

        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please create a .env file or export the variable."
            )
        return self.openai_api_key


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Lazily load settings (cached)."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return Settings()


__all__ = [
    "Settings",
    "settings",
    "PROJECT_ROOT",
    "DATA_DIR",
    "PDF_DIR",
    "INDEX_FILE",
    "METADATA_FILE",
    "INDEX_INFO_FILE",
]
