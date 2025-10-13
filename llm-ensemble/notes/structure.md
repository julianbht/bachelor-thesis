### Reports & Artifacts

Minimize hidden state by writing:

- `runs/{run_id}/manifest.json` — everything about the run  
- `runs/{run_id}/samples.parquet`  
- `runs/{run_id}/judgements/{model_id}.parquet`  
- `runs/{run_id}/ensemble.parquet`  
- `runs/{run_id}/metrics.json`  
- `runs/{run_id}/report.html` — snapshot for your thesis appendix  

Include a **repro footer** in reports:
- Git SHA  
- run_id  
- dataset checksum  
- model registry snapshot path


