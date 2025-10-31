"""Utilities for transforming the Ambedkar writings into a searchable vector store."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List

import orjson
from annoy import AnnoyIndex
from rich.console import Console

from .config import (
    DATA_DIR,
    INDEX_FILE,
    INDEX_INFO_FILE,
    METADATA_FILE,
    PDF_DIR,
    Settings,
    settings,
)
from .embedding import EmbeddingClient

console = Console()


@dataclass
class Chunk:
    """A small, contextual unit of text extracted from the PDFs."""

    chunk_id: str
    content: str
    source: str
    page: int


def _clean_text(raw: str) -> str:
    text = raw.replace(".-", "-")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_words(text: str, max_words: int, overlap: int) -> Iterator[str]:
    words = text.split()
    if not words:
        return

    step = max(max_words - overlap, 1)
    total = len(words)
    start = 0
    while start < total:
        end = min(total, start + max_words)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            yield chunk_text
        if end == total:
            break
        start += step


def _iter_pdf_chunks(cfg: Settings) -> Iterator[Chunk]:
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF directory not found: {PDF_DIR}")

    pdf_paths: List[Path] = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError("No PDF files found in the Ambedkar_Writings directory.")

    for pdf_path in pdf_paths:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
        except Exception as exc:  # pragma: no cover - defensive logging only
            console.print(f"[red]Failed to read {pdf_path.name}: {exc}")
            continue

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as exc:  # pragma: no cover
                console.print(f"[yellow]Skipping page {page_number} of {pdf_path.name}: {exc}")
                continue

            cleaned = _clean_text(raw_text)
            if not cleaned:
                continue

            for chunk_idx, chunk_text in enumerate(
                _chunk_words(cleaned, cfg.chunk_size, cfg.chunk_overlap), start=1
            ):
                chunk_id = f"{pdf_path.stem}_p{page_number}_c{chunk_idx}"
                yield Chunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    source=pdf_path.name,
                    page=page_number,
                )


def ingest_corpus(rebuild: bool = True) -> None:
    """Create or refresh the Annoy index from the Ambedkar writings."""

    cfg = settings()
    cfg.ensure_api_key()

    if rebuild:
        for artefact in (INDEX_FILE, METADATA_FILE, INDEX_INFO_FILE):
            if artefact.exists():
                artefact.unlink()

    chunks = list(_iter_pdf_chunks(cfg))
    if not chunks:
        console.print("[yellow]No textual content extracted from PDFs.")
        return

    embedder = EmbeddingClient(cfg)
    console.print(f"[green]Embedding {len(chunks)} chunks using {cfg.embedding_model}...")

    vectors = embedder.embed_texts([chunk.content for chunk in chunks])

    if not vectors:
        raise RuntimeError("Embedding generation returned no vectors.")

    dimension = len(vectors[0])
    index = AnnoyIndex(dimension, "angular")
    for idx, vector in enumerate(vectors):
        index.add_item(idx, vector)
    index.build(50)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    index.save(str(INDEX_FILE))

    with METADATA_FILE.open("wb") as fh:
        for idx, chunk in enumerate(chunks):
            record = {
                "int_id": idx,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page": chunk.page,
                "content": chunk.content,
            }
            fh.write(orjson.dumps(record))
            fh.write(b"\n")

    info = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "embedding_model": cfg.embedding_model,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "vector_count": len(chunks),
        "dimension": dimension,
        "index_metric": "angular",
    }
    INDEX_INFO_FILE.write_bytes(orjson.dumps(info))

    console.print(
        f"[green]Vector store ready with {len(chunks)} chunks across {len(set(c.source for c in chunks))} PDFs."
    )


__all__ = ["ingest_corpus"]
