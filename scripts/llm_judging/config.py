from dataclasses import dataclass
from typing import Optional, List

@dataclass(frozen=True)
class Pg:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "bachelor-thesis"
    user: str = "postgres"
    password: str = "123"

@dataclass(frozen=True)
class Settings:
    data_schema: str = "passagev2"
    audit_schema: str = "passagev2"
    model: str = "deepseek-r1:14b"
    temperature: float = 0.0
    max_text_chars: Optional[int] = None
    commit_every: int = 5
    retry_enabled: bool = True
    retry_attempts: int = 3
    retry_backoff_ms: int = 50
    limit_qrels: Optional[int] = 100
    official: bool = False
    user_notes: Optional[str] = None
    reasoning_enabled: bool = True  # LLM should give reason for judgement

# ---- Batch plan (edit this list for unattended sweeps) ----
RUN_SPECS: List[Settings] = [
    Settings(model="deepseek-r1:1.5b", user_notes="deepseek 1.5b sweep test run", limit_qrels=5),
    Settings(model="deepseek-r1:7b",   user_notes="deepseek 7b sweep test run",   limit_qrels=100),
]
