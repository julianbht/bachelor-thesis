#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${1:-llm-ensemble}"

# helpers
mkd() { mkdir -p "$1"; }
mkf() {
  local path="$1"
  if [[ -e "$path" ]]; then return 0; fi
  mkdir -p "$(dirname "$path")"
  : > "$path"
}

echo "Scaffolding project: $PROJECT_NAME"
mkd "$PROJECT_NAME"
cd "$PROJECT_NAME"

############################################
# top-level files
############################################
cat > README.md <<'EOF'
# LLM Ensemble (CLI-first, Apps + Libs)

This repo hosts four CLIs — `ingest`, `infer`, `aggregate`, `evaluate` — plus shared libs.
Artifacts (Parquet + preview heads) are written under `artifacts/runs/<run_id>/`.

Quick start (dev flow):
- `make ingest`     → writes samples.parquet + samples.head.jsonl
- `make infer`      → writes judgements/<model>.parquet
- `make aggregate`  → writes ensemble.parquet
- `make evaluate`   → writes metrics.json + report.html
EOF

cat > .gitignore <<'EOF'
# python
__pycache__/
*.pyc
.venv/
.env
# artifacts & data
artifacts/
data/
# OS/editor
.DS_Store
.vscode/
.idea/
# caches
*.cache
EOF

cat > .dockerignore <<'EOF'
artifacts/
data/
.vscode/
.idea/
__pycache__/
*.pyc
.venv/
EOF

cat > .env.example <<'EOF'
APP_ENV=dev
DATASET_NAME=llm_judge_challenge
PROMPT_PROFILE=relevance_v1
MODELS=phi3-mini,tinyllama
ENSEMBLE_STRATEGY=weighted_majority_v1
ARTIFACTS_DIR=./artifacts
DATA_DIR=./data
CACHE_BACKEND=filesystem
CHUNK_SIZE=2000
EARLY_EXIT_THRESHOLD=0.9
# Providers
OLLAMA_BASE_URL=http://localhost:11434
HF_ENDPOINT_URL=https://api-inference.huggingface.co/models/xxx
HF_API_TOKEN=your_token_here
EOF

cat > Makefile <<'EOF'
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
EOF

# minimal pyproject (so editors/linters work)
cat > pyproject.toml <<'EOF'
[project]
name = "llm-ensemble"
version = "0.0.1"
description = "CLI-first LLM ensemble relevance judging system"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "pyarrow>=17.0.0",
  "pandas>=2.2.2",
  "polars>=1.5.0",
  "jinja2>=3.1.4",
  "pydantic>=2.7.4",
  "rich>=13.7.1"
]
EOF

############################################
# high-level dirs
############################################
mkd configs/prompts
mkd configs/models
mkd configs/ensembles
mkd configs/datasets
mkd configs/runtime

mkd data
mkd artifacts/runs

mkd scripts

############################################
# configs (tiny samples)
############################################
cat > configs/prompts/relevance_v1.jinja <<'EOF'
You are a judge assessing relevance.
Given a query and a candidate answer, return JSON with keys:
{ "label": "relevant|partially|irrelevant", "confidence": 0-1, "rationale": "short reason" }.

Query: {{ query }}
Candidate: {{ candidate }}
EOF

cat > configs/models/phi3-mini.yaml <<'EOF'
model_id: phi3-mini
provider: hf
context_window: 4096
default_params:
  temperature: 0.0
  max_tokens: 256
capabilities:
  multilingual: true
EOF

cat > configs/models/tinyllama.yaml <<'EOF'
model_id: tinyllama
provider: ollama
context_window: 2048
default_params:
  temperature: 0.0
  max_tokens: 256
capabilities:
  multilingual: false
EOF

cat > configs/ensembles/weighted_majority_v1.yaml <<'EOF'
strategy: weighted_majority
params:
  default_weight: 1.0
  per_model_weights:
    phi3-mini: 1.0
    tinyllama: 1.0
EOF

cat > configs/datasets/llm_judge_challenge.yaml <<'EOF'
name: llm_judge_challenge
version: "0.1"
source: "local"
splits:
  train: "data/llm_judge_challenge/train.jsonl"
  test: "data/llm_judge_challenge/test.jsonl"
label_space: [relevant, partially, irrelevant]
EOF

cat > configs/runtime/dev.env <<'EOF'
APP_ENV=dev
EOF
cat > configs/runtime/ci.env <<'EOF'
APP_ENV=ci
EOF
cat > configs/runtime/prod.env <<'EOF'
APP_ENV=prod
EOF

############################################
# scripts
############################################
cat > scripts/peek_duckdb.sql <<'EOF'
-- Example DuckDB query for quick peeks
SELECT * FROM 'artifacts/runs/*/ensemble.parquet' LIMIT 50;
EOF

cat > scripts/peek_latest.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if ! command -v duckdb >/dev/null 2>&1; then
  echo "duckdb CLI not found. Install from https://duckdb.org/"
  exit 1
fi
duckdb -c "SELECT * FROM 'artifacts/runs/*/ensemble.parquet' LIMIT 50;"
EOF
chmod +x scripts/peek_latest.sh

############################################
# libs (shared)
############################################
mkd libs/io
mkd libs/schemas
mkd libs/logging
mkd libs/runtime
mkd libs/cache
mkd libs/utils

cat > libs/schemas/sample.schema.json <<'EOF'
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Canonical Sample",
  "type": "object",
  "required": ["sample_id", "query", "candidate"],
  "properties": {
    "sample_id": { "type": "string" },
    "query": { "type": "string" },
    "candidate": { "type": ["string","object"] },
    "references": { "type": ["array","object","null"] },
    "gold_label": { "type": ["string","null"] },
    "metadata": { "type": "object" }
  }
}
EOF

cat > libs/schemas/judgement.schema.json <<'EOF'
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Normalized Judgement",
  "type": "object",
  "required": ["sample_id","model_id","label","confidence"],
  "properties": {
    "sample_id": { "type": "string" },
    "model_id": { "type": "string" },
    "provider": { "type": "string" },
    "label": { "type": "string", "enum": ["relevant","partially","irrelevant"] },
    "score": { "type": ["number","null"] },
    "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
    "rationale": { "type": ["string","null"] },
    "raw_text": { "type": ["string","null"] },
    "latency_ms": { "type": ["number","null"] },
    "attempts": { "type": ["integer","null"] },
    "cache_hit": { "type": ["boolean","null"] },
    "warnings": { "type": "array", "items": { "type": "string" } }
  }
}
EOF

cat > libs/schemas/ensemble.schema.json <<'EOF'
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Ensemble Result",
  "type": "object",
  "required": ["sample_id","final_label","final_confidence"],
  "properties": {
    "sample_id": { "type": "string" },
    "final_label": { "type": "string", "enum": ["relevant","partially","irrelevant"] },
    "final_confidence": { "type": "number", "minimum": 0, "maximum": 1 },
    "disagreement_metrics": { "type": "object" },
    "flags": { "type": "array", "items": { "type": "string" } }
  }
}
EOF

cat > libs/schemas/metrics.schema.json <<'EOF'
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Evaluation Metrics",
  "type": "object",
  "properties": {
    "accuracy": { "type": "number" },
    "macro_f1": { "type": "number" },
    "confusion_matrix": { "type": "array" }
  }
}
EOF

cat > libs/runtime/env.py <<'EOF'
# placeholder: parse env vars here later
EOF

cat > libs/logging/json_logger.py <<'EOF'
# placeholder: structured JSON logger later
EOF

cat > libs/io/parquet_io.py <<'EOF'
# placeholder: read/write parquet + head previews later
EOF

cat > libs/utils/chunking.py <<'EOF'
# placeholder: chunk planners later
EOF

############################################
# apps / CLIs (thin placeholders)
############################################
for app in ingest infer aggregate evaluate; do
  mkd "apps/$app/cli"
  mkd "apps/$app/domain"
  mkd "apps/$app/adapters"
  mkd "apps/$app/tests"
done

cat > apps/ingest/cli/ingest_cli.py <<'EOF'
print("ingest: placeholder CLI — implement loader, write samples.parquet + head")
EOF

cat > apps/infer/cli/infer_cli.py <<'EOF'
print("infer: placeholder CLI — implement providers, normalize judgements, chunked execution")
EOF

cat > apps/aggregate/cli/aggregate_cli.py <<'EOF'
print("aggregate: placeholder CLI — implement majority/weighted vote, partials")
EOF

cat > apps/evaluate/cli/evaluate_cli.py <<'EOF'
print("evaluate: placeholder CLI — compute metrics, render report.html")
EOF

############################################
# docker skeleton (optional now)
############################################
mkd docker/entrypoints
cat > docker/Dockerfile.cpu <<'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .
COPY . .
ENTRYPOINT ["bash","-lc"]
CMD ["echo","Build successful. Use compose for dev runs."]
EOF

cat > docker/entrypoints/run_cli.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exec "$@"
EOF
chmod +x docker/entrypoints/run_cli.sh

echo "Done. Next steps:"
echo "1) cd $PROJECT_NAME"
echo "2) python -m venv .venv && source .venv/Scripts/activate  # (Git Bash on Windows)"
echo "3) pip install -e ."
echo "4) cp .env.example .env && edit as needed"
echo "5) make ingest  # (placeholders print; replace with real logic as you develop)"
