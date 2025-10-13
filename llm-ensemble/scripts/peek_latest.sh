#!/usr/bin/env bash
set -euo pipefail
if ! command -v duckdb >/dev/null 2>&1; then
  echo "duckdb CLI not found. Install from https://duckdb.org/"
  exit 1
fi
duckdb -c "SELECT * FROM 'artifacts/runs/*/ensemble.parquet' LIMIT 50;"
