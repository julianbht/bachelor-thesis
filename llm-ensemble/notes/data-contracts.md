### Canonical Dataset Record
- **sample_id:** str  
- **query:** str  
- **candidate:** str | dict (answer/passages/etc.)  
- **references:** Optional[list[str] | dict]  
- **metadata:** dict (domain, length, etc.)  
- **gold_label:** Optional[str] (for evaluation/calibration splits)

### Canonical Model Judgement
- **model_id**, **provider**, **version**  
- **label:** {relevant, partially, irrelevant} (configurable)  
- **score:** [0,1] (mapped from model output)  
- **confidence:** [0,1] (self-reported or derived)  
- **rationale:** str  
- **raw_text:** str  
- **latency_ms**, **retries**, **cost_estimate**

### Ensemble Output
- **final_label**, **final_confidence**  
- **per_model_votes:** list[canonical model judgement]  
- **aggregation_strategy:** name + params  
- **warnings:** list[str] (ties, low agreement, parser fallbacks)
