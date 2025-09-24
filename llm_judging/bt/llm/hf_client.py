from __future__ import annotations
import time, logging, requests
from requests.exceptions import Timeout as ReqTimeout
from typing import Any, Dict
from bt.call import call_with_retry
from bt.util.parsing import parse_score_and_reason
from bt.config import Settings

log = logging.getLogger("bt.llm.hf")

class HFEndpointClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self._session = requests.Session()

    @property
    def model_label(self) -> str:
        # show endpoint in audit logs; you may also include s.model if you want
        return f"hf_endpoint:{self.s.hf_endpoint_url or self.s.model}"

    def _extract_text(self, obj: Any) -> str:
        if isinstance(obj, list) and obj:
            item = obj[0]
            if isinstance(item, dict):
                if "generated_text" in item: return item.get("generated_text") or ""
                if "text" in item: return item.get("text") or ""
        if isinstance(obj, dict):
            if "generated_text" in obj: return obj.get("generated_text") or ""
            if "output_text" in obj: return obj.get("output_text") or ""
            if "outputs" in obj and isinstance(obj["outputs"], list) and obj["outputs"]:
                c = obj["outputs"][0]
                if isinstance(c, dict):
                    if "content" in c: return c.get("content") or ""
                    if "generated_text" in c: return c.get("generated_text") or ""
        return ""

    def _single_call(self, prompt: str):
        t0 = time.time()
        headers = {"Accept": "application/json"}
        if self.s.hf_api_token:
            headers["Authorization"] = f"Bearer {self.s.hf_api_token}"
        payload: Dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "temperature": float(self.s.temperature),
                "max_new_tokens": int(self.s.max_new_tokens),
                "return_full_text": False,
            }
        }
        p = payload["parameters"]
        if self.s.top_p is not None: p["top_p"] = float(self.s.top_p)
        if self.s.top_k is not None: p["top_k"] = int(self.s.top_k)
        if self.s.repetition_penalty is not None: p["repetition_penalty"] = float(self.s.repetition_penalty)

        try:
            read_to = (self.s.llm_timeout_ms / 1000.0) if (self.s.llm_timeout_ms and self.s.llm_timeout_ms > 0) else None
            r = self._session.post(self.s.hf_endpoint_url, headers=headers, json=payload, timeout=(5, read_to))
            r.raise_for_status()
            data = r.json()
            text = self._extract_text(data)
            raw: Dict[str, Any] = {"provider": "hf_endpoint", "hf": data, "response_text": text}
            ms = int((time.time() - t0) * 1000)
            score, reason = parse_score_and_reason(text)
            return score, reason, raw, ms
        except ReqTimeout:
            ms = int((time.time() - t0) * 1000)
            return None, None, {"provider": "hf_endpoint", "error": "timeout"}, ms

    def judge(self, prompt: str):
        return call_with_retry(
            lambda: self._single_call(prompt),
            attempts=self.s.retry_attempts,
            enabled=self.s.retry_enabled,
            backoff_ms=self.s.retry_backoff_ms,
        )

    def close(self) -> None:
        self._session.close()
