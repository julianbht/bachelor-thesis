from __future__ import annotations
from typing import Protocol, Dict, Any

class LLMClient(Protocol):
    def judge(self, prompt: str) -> tuple[int | None, str | None, Dict[str, Any], int]:
        """Return (score, reason, raw, elapsed_ms)."""
        ...
    def close(self) -> None: ...
    @property
    def model_label(self) -> str: ...
