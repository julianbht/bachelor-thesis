# LLM Ensemble – Ingest CLI (First Step)

This CLI normalizes the **LLM Judge Challenge 2024** dataset into a single NDJSON stream of `JudgingExample` records for downstream components.

## Install (dev)

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -e .
```

## Usage

```bash
# to stdout
python -m cli --help
python -m cli ingest -d llm-judge -i ./data > ./out/llm_judge.ndjson

# to a file
python -m cli ingest -d llm-judge -i ./data -o ./out/llm_judge.ndjson

# process only first N for smoke testing
python -m cli ingest -d llm-judge -i ./data -o ./out/part.ndjson --limit 100
```

## Output Schema

Each line is a JSON object:

```json
{
  "dataset": "llm-judge-2024",
  "query_id": "q1",
  "query_text": "What is AI?",
  "docid": "d1",
  "doc": "AI is...",
  "gold_relevance": 1
}
```
# .env.example
LOG_LEVEL=INFO
