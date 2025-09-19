from __future__ import annotations
import time, logging, requests, ollama
from requests.exceptions import Timeout as ReqTimeout
from typing import Dict, Any
from bt.call import call_with_retry
from bt.util.parsing import parse_score_and_reason
from bt.config import Settings

log = logging.getLogger("bt.llm.ollama")

class OllamaClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self._session = requests.Session()
        # Pull model only for ollama runs
        try:
            ollama.show(model=self.s.model)
        except Exception:
            log.info("Pulling Ollama model '%s'…", self.s.model)
            last = None
            for _ in range(3):
                try:
                    for __ in ollama.pull(model=self.s.model, stream=True):
                        pass
                    ollama.show(model=self.s.model)
                    break
                except Exception as e:
                    last = e
                    time.sleep(0.5)
            else:
                raise RuntimeError(f"Ollama pull failed: {last!r}")

    @property
    def model_label(self) -> str:
        return f"ollama:{self.s.model}"

    def _single_call(self, prompt: str):
        t0 = time.time()
        payload = {
            "model": self.s.model,
            "prompt": prompt,
            "options": {"temperature": float(self.s.temperature)},
            "stream": False,
        }
        try:
            read_to = (self.s.llm_timeout_ms / 1000.0) if (self.s.llm_timeout_ms and self.s.llm_timeout_ms > 0) else None
            r = self._session.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=(5, read_to))
            r.raise_for_status()
            data = r.json()
            text = data.get("response", "") or ""
            raw: Dict[str, Any] = {"provider": "ollama", "ollama": data, "response_text": text}
            ms = int((time.time() - t0) * 1000)
            score, reason = parse_score_and_reason(text)
            return score, reason, raw, ms
        except ReqTimeout:
            ms = int((time.time() - t0) * 1000)
            return None, None, {"provider": "ollama", "error": "timeout"}, ms

    def judge(self, prompt: str):
        return call_with_retry(
            lambda: self._single_call(prompt),
            attempts=self.s.retry_attempts,
            enabled=self.s.retry_enabled,
            backoff_ms=self.s.retry_backoff_ms,
        )

    def close(self) -> None:
        self._session.close()
