"""Annoy-based vector store loader and search utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import orjson
from annoy import AnnoyIndex

from .config import INDEX_FILE, INDEX_INFO_FILE, METADATA_FILE, Settings, settings
from .embedding import EmbeddingClient


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    source: str
    page: int
    score: float


class VectorStore:
    """Simple wrapper over an Annoy index and its metadata."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings()
        if not INDEX_FILE.exists() or not METADATA_FILE.exists() or not INDEX_INFO_FILE.exists():
            raise FileNotFoundError(
                "Vector store files not found. Run `poetry run ambedkar-chatbot ingest` first."
            )
        info = orjson.loads(INDEX_INFO_FILE.read_bytes())
        self.dimension = info["dimension"]
        self.embedding_model = info["embedding_model"]
        self._index = AnnoyIndex(self.dimension, info.get("index_metric", "angular"))
        self._index.load(str(INDEX_FILE))
        self._metadata = self._load_metadata()
        self._embedder = EmbeddingClient(self.cfg)

    def _load_metadata(self) -> List[dict]:
        metadata: List[dict] = []
        with METADATA_FILE.open("rb") as fh:
            for line in fh:
                metadata.append(orjson.loads(line))
        return metadata

    def _score_from_distance(self, distance: float) -> float:
        # Annoy angular distance (0..2). Convert to an affinity score for presentation.
        return max(min(1 - distance / 2, 1.0), 0.0)

    def similarity_search(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
        top_k = top_k or self.cfg.top_k
        query_vector = self._embedder.embed_query(query)
        indices, distances = self._index.get_nns_by_vector(
            query_vector, n=top_k, include_distances=True
        )
        results: List[RetrievedChunk] = []
        for idx, distance in zip(indices, distances):
            meta = self._metadata[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=meta["chunk_id"],
                    content=meta["content"],
                    source=meta["source"],
                    page=meta["page"],
                    score=self._score_from_distance(distance),
                )
            )
        return results


__all__ = ["VectorStore", "RetrievedChunk"]
