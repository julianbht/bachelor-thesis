## Processes
### Ingest
**Input:** dataset spec (`DATASET_NAME`, optional split)  
**Output:** normalized samples (`JSONL`/`Parquet`) → `/artifacts/samples/...`  
**Tasks:** schema validation, text cleanup, token budgeting metadata  

### Infer
**Input:** normalized samples + model set  
**Output:** per-model judgements → `/artifacts/infer/{model_id}/...`  
**Tasks:** build prompts, dispatch to providers, parse outputs, record latency/retries, cache hits/misses  

### Aggregate
**Input:** per-model judgements  
**Output:** ensemble decisions → `/artifacts/aggregate/...` (+ optional rationale)  
**Tasks:** run ensemble strategy, compute agreement & uncertainty  

### Evaluate
**Input:** ensemble decisions + gold labels  
**Output:** metrics & reports → `/artifacts/eval/...`  
**Tasks:** metrics, error slices, disagreement sets, calibration fits, cost & latency analysis  

**Why this helps:**  
Run as subcommands in dev, split in Docker, test independently in CI.