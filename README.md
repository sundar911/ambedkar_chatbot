# Ambedkar Chatbot

A retrieval-augmented chatbot that helps people explore and debate the writings of Dr. B. R. Ambedkar in calm, plain English. The agent grounds every answer in the PDF corpus stored in `Ambedkar_Writings/`, and keeps the tone empathetic, mature, and constructive even when faced with disagreement.

## Features

- Vectorise Ambedkar's collected writings with OpenAI embeddings and an Annoy index stored locally in `data/`
- Fast retrieval of relevant passages for each question, with volume and page level references
- Persona-aware chat pipeline that encourages reflective, respectful dialogue around points of disagreement
- Typer-powered CLI for ingestion, status checks, and interactive conversations
- Poetry-managed environment with reproducible dependency graph

## Getting Started

1. **Install dependencies**
   ```bash
   poetry install
   ```
2. **Create a `.env` file** (or export the variables in your shell):
   ```bash
   cat <<'ENV' > .env
   OPENAI_API_KEY=sk-your-key
   # Optional overrides
   # EMBED_MODEL=text-embedding-3-small
   # CHAT_MODEL=gpt-4o-mini
   # CHUNK_SIZE=320
   # CHUNK_OVERLAP=60
   # TOP_K=6
   ENV
   ```
3. **Build the vector store** (downloads embeddings for all PDFs and writes Annoy index + metadata):
   ```bash
   poetry run ambedkar-chatbot ingest
   ```
   The first run can take a while and incurs OpenAI embedding costs. Artefacts are written to `data/`.
4. **Chat with the companion**:
   ```bash
   poetry run ambedkar-chatbot chat
   ```
   Type `exit` (or press `Ctrl+D`) to leave the conversation.

## CLI Reference

- `poetry run ambedkar-chatbot ingest --incremental` — append only new material; defaults to full rebuild
- `poetry run ambedkar-chatbot info` — quick health check for index/metadata files
- `poetry run ambedkar-chatbot chat --top-k 8` — override the number of context chunks

## Persona Guidelines

The chatbot blends the warmth of a community educator with Ambedkar's analytical rigor:

- Uses plain, inclusive English and avoids jargon when possible
- Surfaces relevant citations (volume + page) whenever passages inform the reply
- Engages disagreements patiently, validating the other person's curiosity before clarifying Ambedkar's stance
- Encourages further reading, offering actionable suggestions to continue learning

## Repository Layout

```
Ambedkar_Writings/    # Source PDFs (input corpus)
data/                 # Generated embeddings, metadata, and Annoy index (git-ignored)
src/ambedkar_chatbot/ # Python package (config, ingest pipeline, vector store, chatbot, CLI)
poetry.toml / .venv   # Poetry config and in-project virtualenv (ignored)
```

## Next Steps

- Add automated tests that mock OpenAI responses for repeatable CI runs
- Expose the persona via a small FastAPI or Streamlit frontend
- Add incremental ingestion heuristics (hashing chunks, skipping unchanged PDFs)
- Experiment with local embedding models when GPU resources are available

---
Released under the MIT License.
