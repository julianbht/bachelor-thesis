# bt/db.py
from __future__ import annotations
import time
import logging
import ollama

from bt.parsing import parse_score_and_reason

log = logging.getLogger("bt.llm")


def ensure_model_downloaded(model: str, *, retries: int = 3, backoff_ms: int = 500) -> None:
    """
    Ensure the Ollama model is present locally. If not, pull it before running.
    """
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


def _single_call(model: str, prompt: str, temperature: float) -> tuple[int | None, str | None, dict, int]:
    t0 = time.time()
    res = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": float(temperature)},
    )
    elapsed_ms = int((time.time() - t0) * 1000)

    raw_text = res.response
    raw_meta = res.model_dump()
    raw_meta.pop("response", None)
    raw = {"ollama": raw_meta, "response_text": raw_text}

    score, reason = parse_score_and_reason(raw_text)
    return score, reason, raw, elapsed_ms


def judge_with_ollama(
    model: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    attempts: int = 1,
    enabled: bool = True,
    backoff_ms: int = 500,
) -> tuple[int | None, str | None, dict, int]:
    """
    Returns (pred_score or None, reason or None, raw_response_dict, elapsed_ms_total).
    Retries parsing/LLM failures up to `attempts` if `enabled` is True.
    """
    attempts = max(1, int(attempts))
    total_ms = 0
    last_raw: dict | None = None
    last_reason: str | None = None

    for i in range(1, attempts + 1):
        log.debug(
            "LLM call attempt %d/%d",
            i, attempts
        )
        try:
            pred, reason, raw, ms = _single_call(model, prompt, temperature)
            total_ms += ms
            last_raw = raw
            last_reason = reason

            if pred is not None:
                log.debug("LLM call attempt %d/%d succeeded.",i, attempts)
                return pred, reason, raw, total_ms
            else:
                log.warning("LLM call returned no prediction on attempt %d/%d",i, attempts)
        except Exception as e:
            last_raw = {"error": repr(e)}
            log.exception("LLM call attempt %d/%d failed", i, attempts)

        # if retries disabled or last attempt, stop here
        if not enabled or i == attempts:
            break

        log.debug("Retrying in %d ms...", backoff_ms)
        time.sleep(backoff_ms / 1000.0)

    log.error(
        "LLM call failed after %d attempt(s) | returning None",
        attempts
    )
    return None, last_reason, (last_raw or {}), total_ms

