from __future__ import annotations
from pathlib import Path
from typing import Optional

import typer

from llm_ensemble.ingest.adapters.llm_judge import iter_examples
from llm_ensemble.ingest.domain.models import JudgingExample

app = typer.Typer(add_completion=False, help="LLM Ensemble – data ingest CLI")


def _json_dumps(obj: JudgingExample) -> str:
    return obj.model_dump_json()


@app.command("ingest")
def ingest(
    dataset: str = typer.Option(
        ..., "--dataset", "-d", help="Dataset adapter to use (e.g., llm-judge)"
    ),
    data_dir: Path = typer.Option(
        ..., "--data-dir", "-i", exists=True, file_okay=False, readable=True,
        help="Path to directory containing raw dataset files",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Write NDJSON to this file (defaults to stdout)",
    ),
    limit: Optional[int] = typer.Option(None, help="Process at most N examples"),
):
    """Normalize a raw IR dataset into JudgingExample NDJSON records.

    12-factor friendly: reads from files, writes to stdout by default, and can be
    configured via flags/env. Use LOG_LEVEL env var for verbosity.
    """

    dataset = dataset.lower().strip()
    if dataset not in {"llm-judge", "llm-judge-2024"}:
        raise typer.BadParameter("Unsupported dataset. Try: llm-judge")

    # Output stream selection
    if out is None:
        import sys
        sink = sys.stdout
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        sink = out.open("w", encoding="utf-8", newline="\n")

    count = 0
    try:
        for ex in iter_examples(data_dir):
            sink.write(_json_dumps(ex) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break
    finally:
        if out is not None:
            sink.close()
    typer.echo(f"Wrote {count} examples", err=True)


if __name__ == "__main__":
    app()
