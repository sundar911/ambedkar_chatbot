"""Typer-based command line interface for the Ambedkar chatbot."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from .chat import Chatbot
from .config import INDEX_FILE, INDEX_INFO_FILE, METADATA_FILE
from .ingest import ingest_corpus

app = typer.Typer(help="Utilities for preparing and conversing with the Ambedkar chatbot.")
console = Console()


@app.command()
def ingest(
    rebuild: bool = typer.Option(
        True,
        "--rebuild/--incremental",
        help="Rebuild the vector store from scratch (default) or append new material.",
    ),
) -> None:
    """Parse PDFs, create embeddings, and build the Annoy index."""

    if not rebuild and not INDEX_FILE.exists():
        console.print("[yellow]No existing index detected; performing a full rebuild instead.")
        rebuild = True

    ingest_corpus(rebuild=rebuild)


@app.command()
def chat(top_k: Optional[int] = typer.Option(None, help="Override the number of context chunks.")) -> None:
    """Start an interactive chat session in the terminal."""

    try:
        bot = Chatbot()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}")
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        console.print(f"[red]{exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        Panel(
            "Type your questions or disagreements to explore Ambedkar's ideas.\n"
            "Enter 'exit' or press Ctrl+D to leave the conversation.",
            title="Ambedkar Companion",
            expand=False,
        )
    )

    history: list[dict[str, str]] = []
    while True:
        try:
            question = typer.prompt("You")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[cyan]Goodbye! Keep exploring Ambedkar's work.")
            break

        normalized = question.strip()
        if not normalized:
            continue
        if normalized.lower() in {"exit", "quit"}:
            console.print("[cyan]Conversation ended by user.")
            break

        answer, contexts = bot.answer(normalized, history=history, top_k=top_k)
        history.append({"role": "user", "content": normalized})
        history.append({"role": "assistant", "content": answer})

        console.print(Panel(answer, title="Ambedkar Companion", border_style="cyan"))
        if contexts:
            console.print("[bold]Supporting references:[/]")
            for item in contexts:
                console.print(
                    f"  • {item.source} – page {item.page} (score {item.score:.2f})"
                )
        console.print()


@app.command()
def info() -> None:
    """Display the current status of the vector store files."""

    artefacts = {
        "Index": INDEX_FILE,
        "Metadata": METADATA_FILE,
        "Info": INDEX_INFO_FILE,
    }
    rows = []
    for name, path in artefacts.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        rows.append(f"{name:10} | {'present' if exists else 'missing ':8} | {size/1024:.1f} KiB")
    console.print("\n".join(rows))


def main() -> None:  # pragma: no cover - Typer registers this
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
