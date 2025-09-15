from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Pg:
    host: str = os.getenv("PGHOST", "localhost")
    port: int = int(os.getenv("PGPORT", "5432"))
    dbname: str = os.getenv("PGDATABASE", "bachelor-thesis")
    user: str = os.getenv("PGUSER", "postgres")
    password: str = os.getenv("PGPASSWORD", "123")

@dataclass(frozen=True)
class Settings:
    data_schema: str = os.getenv("DATA_SCHEMA", "passagev2")
    audit_schema: str = os.getenv("AUDIT_SCHEMA", "passagev2")
    model: str = os.getenv("LLM_MODEL", "deepseek-r1:14b")
    max_text_chars: int = int(os.getenv("MAX_TEXT_CHARS", "2000000000"))
    commit_every: int = int(os.getenv("COMMIT_EVERY", "50"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
