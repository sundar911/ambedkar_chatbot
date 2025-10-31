"""Conversational interface that blends retrieval with OpenAI responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from openai import APIError, OpenAI, RateLimitError, Timeout
from rich.console import Console

from .config import Settings, settings
from .vector_store import RetrievedChunk, VectorStore

SYSTEM_PROMPT = """
You are a calm, empathetic companion representing the scholarship of Dr. B. R. Ambedkar.
Speak in clear, accessible English while staying faithful to the cited writings. Meet disagreement
with patience, curiosity, and constructive reasoning. Encourage nuanced dialogue and invite learners
into the material rather than dismissing other viewpoints. Avoid moralizing; guide with context,
logic, historical detail, and compassion.
""".strip()

console = Console()


@dataclass
class Chatbot:
    cfg: Settings = field(default_factory=settings)
    _client: OpenAI = field(init=False)
    _store: VectorStore = field(init=False)

    def __post_init__(self) -> None:
        self.cfg.ensure_api_key()
        self._client = OpenAI(api_key=self.cfg.openai_api_key)
        self._store = VectorStore(self.cfg)

    def _format_context(self, contexts: Sequence[RetrievedChunk]) -> str:
        if not contexts:
            return "Context: (no supporting passages retrieved)"
        lines: List[str] = ["Context passages (highest relevance first):"]
        for idx, chunk in enumerate(contexts, start=1):
            preview = chunk.content.strip()
            if len(preview) > 550:
                preview = preview[:550].rsplit(" ", 1)[0] + " …"
            lines.append(
                f"{idx}. Volume: {chunk.source}, page {chunk.page} | Score: {chunk.score:.2f}\n   {preview}"
            )
        return "\n".join(lines)

    def answer(
        self,
        question: str,
        history: Sequence[dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> tuple[str, List[RetrievedChunk]]:
        contexts = self._store.similarity_search(question, top_k=top_k)
        context_prompt = self._format_context(contexts)

        messages: List[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    "You have access to excerpts from Dr. Ambedkar's writings. Cite them naturally in plain language "
                    "using the volume file name and page number when they inform your answer."
                ),
            },
        ]
        if history:
            messages.extend(history)

        user_prompt = (
            f"{context_prompt}\n\n"  # retrieval context first
            f"Conversation partner: {question}\n\n"
            "Craft a thoughtful reply that references the context when relevant, acknowledges the user's perspective, "
            "and suggests concrete ways to explore Ambedkar's work further."
        )
        messages.append({"role": "user", "content": user_prompt})

        for attempt in range(5):
            try:
                response = self._client.chat.completions.create(
                    model=self.cfg.completion_model,
                    temperature=self.cfg.persona_temperature,
                    messages=messages,
                )
                answer = response.choices[0].message.content or ""
                return answer.strip(), list(contexts)
            except (RateLimitError, Timeout):  # pragma: no cover - network variability
                wait_time = min(2 ** attempt, 30)
                console.print(f"[yellow]Rate limited. Retrying in {wait_time}s...")
            except APIError as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"OpenAI API error: {exc}") from exc
        raise RuntimeError("Failed to generate a response after multiple attempts.")


__all__ = ["Chatbot"]
