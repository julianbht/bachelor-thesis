# bt/llm.py
from __future__ import annotations
import time
import logging
import requests
from requests.exceptions import Timeout as ReqTimeout
import ollama

from bt.parsing import parse_score_and_reason

log = logging.getLogger("bt.llm")

# Reuse connections
_SESSION = requests.Session()
_OLLAMA_HOST = "http://127.0.0.1:11434" 

def ensure_model_downloaded(model: str, *, retries: int = 3, backoff_ms: int = 500) -> None:
    log.info("Checking model availability: %s", model)
    try:
        ollama.show(model=model)
        log.debug("Model '%s' is already available.", model)
        return
    except Exception:
        log.info("Model '%s' not found locally, attempting to pull…", model)

    last_err = None
    for i in range(1, max(1, retries) + 1):
        try:
            log.debug("Pull attempt %d/%d for '%s'", i, retries, model)
            for _ in ollama.pull(model=model, stream=True):
                pass
            ollama.show(model=model)
            log.info("Successfully pulled model '%s'.", model)
            return
        except Exception as e:
            last_err = e
            log.warning("Pull attempt %d failed: %s", i, e)
            if i < retries:
                log.debug("Retrying in %d ms…", backoff_ms)
                time.sleep(backoff_ms / 1000.0)
    raise RuntimeError(f"Failed to ensure model '{model}' is available: {last_err!r}")

def _single_call_http(
    model: str,
    prompt: str,
    temperature: float,
    *,
    llm_timeout_ms: int | None,
) -> tuple[int | None, str | None, dict, int]:
    """
    One HTTP call to /api/generate with a real socket timeout.
    Returns (score, reason, raw, elapsed_ms)
    """
    t0 = time.time()
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": float(temperature)},
        "stream": False,              # single JSON response
        # "keep_alive": -1,           # optional: keep model pinned in RAM
    }
    try:
        # timeout=(connect_timeout, read_timeout)
        read_to = (llm_timeout_ms / 1000.0) if (llm_timeout_ms and llm_timeout_ms > 0) else None
        r = _SESSION.post(f"{_OLLAMA_HOST}/api/generate", json=payload, timeout=(5, read_to))
        r.raise_for_status()
        data = r.json()
        raw_text = data.get("response", "") or ""
        raw = {"ollama": data, "response_text": raw_text}
        elapsed_ms = int((time.time() - t0) * 1000)
        score, reason = parse_score_and_reason(raw_text)
        return score, reason, raw, elapsed_ms
    except ReqTimeout:
        elapsed_ms = int((time.time() - t0) * 1000)
        return None, None, {"error": "timeout", "elapsed_ms": elapsed_ms}, elapsed_ms

def judge_with_ollama(
    model: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    attempts: int = 1,
    enabled: bool = True,
    backoff_ms: int = 500,
    llm_timeout_ms: int | None = 30000,   # per-attempt timeout (ms)
) -> tuple[int | None, str | None, dict, int]:
    """
    Returns (pred_score or None, reason or None, raw_response_dict, elapsed_ms_total).
    Retries parsing/LLM failures up to `attempts` if `enabled` is True.
    Each attempt is bounded by `llm_timeout_ms` via HTTP-level timeout.
    """
    attempts = max(1, int(attempts))
    total_ms = 0
    last_raw: dict | None = None
    last_reason: str | None = None

    for i in range(1, attempts + 1):
        log.debug("LLM call attempt %d/%d", i, attempts)
        pred, reason, raw, ms = _single_call_http(model, prompt, temperature, llm_timeout_ms=llm_timeout_ms)
        total_ms += ms
        last_raw = raw
        last_reason = reason

        if pred is not None:
            log.debug("LLM call attempt %d/%d succeeded.", i, attempts)
            return pred, reason, raw, total_ms

        # pred=None -> either parse miss or timeout
        if raw and raw.get("error") == "timeout":
            log.warning("LLM call timed out on attempt %d/%d after %d ms", i, attempts, ms)
        else:
            log.warning("LLM call returned no prediction on attempt %d/%d", i, attempts)

        if not enabled or i == attempts:
            break
        log.debug("Retrying in %d ms...", backoff_ms)
        time.sleep(backoff_ms / 1000.0)

    log.warning("LLM call failed after %d attempts; returning None", attempts)
    return None, last_reason, (last_raw or {}), total_ms
