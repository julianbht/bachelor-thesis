# bt/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import json
import pathlib
import os  # NEW

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

    # Provider selection
    provider: str = "ollama"  # "ollama" | "hf_hub" | "hf_endpoint"

    # Common LLM fields
    model: str = "deepseek-r1:14b"
    temperature: float = 0.0
    reasoning_enabled: bool = False
    llm_timeout_ms: Optional[int] = 120000  # 2 minutes

    # HF Inference Endpoint (used when provider == "hf_endpoint")
    hf_endpoint_url: Optional[str] = None
    hf_api_token: Optional[str] = None  # can be sourced from env
    max_new_tokens: int = 256
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None

    # Run behavior
    max_text_chars: Optional[int] = None
    commit_every: int = 5
    limit_qrels: Optional[int] = 1000
    official: bool = False
    user_notes: Optional[str] = None

    # Retries
    retry_enabled: bool = True
    retry_attempts: int = 2
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
    # allow token from env if not provided in JSON
    merged = dict(d)
    if not merged.get("hf_api_token"):
        merged["hf_api_token"] = os.getenv("HUGGINGFACE_API_TOKEN")

    allowed = {f.name for f in Settings.__dataclass_fields__.values()}
    filtered = {k: v for k, v in merged.items() if k in allowed}
    return Settings(**filtered)
