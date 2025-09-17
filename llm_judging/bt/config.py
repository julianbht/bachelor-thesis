from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import json
import pathlib

@dataclass(frozen=True)
class Pg:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "bachelor-thesis"
    user: str = "postgres"
    password: str = "123"
    

@dataclass(frozen=True)
class Settings:
    # Data / audit
    data_schema: str = "passagev2"
    audit_schema: str = "passagev2"

    # LLM
    model: str = "deepseek-r1:14b"
    temperature: float = 0.0
    reasoning_enabled: bool = True

    # Run behavior
    max_text_chars: Optional[int] = None
    commit_every: int = 5
    limit_qrels: Optional[int] = 100
    official: bool = False
    user_notes: Optional[str] = None

    # Retries
    retry_enabled: bool = True
    retry_attempts: int = 3
    retry_backoff_ms: int = 50


def load_settings_file(path: str | pathlib.Path) -> List[Settings]:
    """
    Load a JSON config file from the 'run_configs' folder.

    Accepts either:
      - a single settings object (dict) -> returns [Settings]
      - a list of settings objects      -> returns [Settings, Settings, ...]
    """
    path = pathlib.Path("run_configs") / pathlib.Path(path).name

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        return [_from_dict(obj)]
    if isinstance(obj, list):
        return [_from_dict(x) for x in obj]

    raise ValueError("Config JSON must be either an object or an array of objects.")



def _from_dict(d: Dict[str, Any]) -> Settings:
    allowed = {f.name for f in Settings.__dataclass_fields__.values()}
    filtered = {k: v for k, v in d.items() if k in allowed}
    return Settings(**filtered)
