SHELL := /usr/bin/env bash
.PHONY: ingest infer aggregate evaluate peek

export PYTHONUNBUFFERED=1

ingest:
	@echo ">> ingest (placeholder) — writes samples.parquet + head"
	@python apps/ingest/cli/ingest_cli.py

infer:
	@echo ">> infer (placeholder) — runs models over samples"
	@python apps/infer/cli/infer_cli.py

aggregate:
	@echo ">> aggregate (placeholder) — majority vote"
	@python apps/aggregate/cli/aggregate_cli.py

evaluate:
	@echo ">> evaluate (placeholder) — metrics + report"
	@python apps/evaluate/cli/evaluate_cli.py

peek:
	@echo ">> DuckDB peek latest run (requires duckdb CLI)"
	@bash scripts/peek_latest.sh
