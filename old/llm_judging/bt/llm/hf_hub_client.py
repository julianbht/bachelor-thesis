from __future__ import annotations
import time, logging
from typing import Dict, Any
from huggingface_hub import InferenceClient
from bt.call import call_with_retry
from bt.util.parsing import parse_score_and_reason
from bt.config import Settings

log = logging.getLogger("bt.llm.hf_hub")

class HFHubClient:
    def __init__(self, s: Settings):
        self.s = s
        if not self.s.hf_api_token:
            raise ValueError("HF API token missing. Set hf_api_token or HUGGINGFACE_API_TOKEN.")
        self.client = InferenceClient(token=self.s.hf_api_token)

    @property
    def model_label(self) -> str:
        # Uses repo name, e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
        return f"hf_hub:{self.s.model}"

    def _single_call(self, prompt: str):
        t0 = time.time()
        try:
            # chat.completions works for most chatty text-gen models
            rsp = self.client.chat.completions.create(
                model=self.s.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.s.max_new_tokens,
                temperature=self.s.temperature,
            )
            text = rsp.choices[0].message["content"]
            ms = int((time.time() - t0) * 1000)
            score, reason = parse_score_and_reason(text)
            raw: Dict[str, Any] = {"provider": "hf_hub", "hf": rsp, "response_text": text}
            return score, reason, raw, ms
        except Exception as e:
            ms = int((time.time() - t0) * 1000)
            log.warning("HF Hub call failed: %s", e)
            return None, None, {"provider": "hf_hub", "error": str(e)}, ms

    def judge(self, prompt: str):
        return call_with_retry(
            lambda: self._single_call(prompt),
            attempts=self.s.retry_attempts,
            enabled=self.s.retry_enabled,
            backoff_ms=self.s.retry_backoff_ms,
        )

    def close(self):  # nothing persistent to close
        pass
