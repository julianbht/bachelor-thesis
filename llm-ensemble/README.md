# LLM Ensemble (CLI-first, Apps + Libs)

This repo hosts four CLIs — `ingest`, `infer`, `aggregate`, `evaluate` — plus shared libs.
Artifacts (Parquet + preview heads) are written under `artifacts/runs/<run_id>/`.

Quick start (dev flow):
- `make ingest`     → writes samples.parquet + samples.head.jsonl
- `make infer`      → writes judgements/<model>.parquet
- `make aggregate`  → writes ensemble.parquet
- `make evaluate`   → writes metrics.json + report.html
