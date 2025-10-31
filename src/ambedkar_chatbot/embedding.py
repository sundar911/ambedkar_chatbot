"""Wrapper around the OpenAI embeddings API with batching and retry logic."""

from __future__ import annotations

import time
from typing import Iterable, List, Sequence

from openai import APIError, OpenAI, RateLimitError, Timeout

from .config import Settings, settings


class EmbeddingClient:
    """Thin wrapper that batches embeddings and retries rate-limited calls."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings()
        api_key = self.cfg.ensure_api_key()
        self.client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings: List[List[float]] = []
        batch_size = self.cfg.batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            attempt = 0
            while True:
                try:
                    response = self.client.embeddings.create(
                        model=self.cfg.embedding_model,
                        input=list(batch),
                    )
                except (RateLimitError, Timeout) as exc:  # pragma: no cover - network timing
                    delay = min(2 ** attempt, 30)
                    time.sleep(delay)
                    attempt += 1
                    if attempt > 5:
                        raise RuntimeError(
                            "Repeated rate limit errors while requesting embeddings"
                        ) from exc
                    continue
                except APIError as exc:  # pragma: no cover - defensive
                    raise RuntimeError(f"OpenAI API error while embedding: {exc}") from exc
                embeddings.extend([item.embedding for item in response.data])
                break
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        vectors = self.embed_texts([text])
        if not vectors:
            raise RuntimeError("Failed to compute embedding for query")
        return vectors[0]


__all__ = ["EmbeddingClient"]
