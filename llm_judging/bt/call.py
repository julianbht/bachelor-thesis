# bt/retry.py
from __future__ import annotations
import time
import logging
from typing import Callable, Tuple, Any, Dict

log = logging.getLogger("bt.llm.retry")

def call_with_retry(
    fn: Callable[[], Tuple[int | None, str | None, Dict[str, Any], int]],
    attempts: int,
    enabled: bool,
    backoff_ms: int,
) -> Tuple[int | None, str | None, Dict[str, Any], int]:
    """
    Calls `fn()` up to `attempts` times (if `enabled`), accumulating elapsed time.
    Expects `fn()` to return: (pred, reason, raw, ms)
    """
    attempts = max(1, int(attempts))
    total_ms = 0
    last_raw: Dict[str, Any] | None = None
    last_reason: str | None = None

    for i in range(1, attempts + 1):
        log.debug("LLM call attempt %d/%d", i, attempts)

        pred, reason, raw, ms = fn()
        total_ms += (ms or 0)
        last_raw = raw
        last_reason = reason

        if pred is not None:
            log.debug("LLM call attempt %d/%d succeeded.", i, attempts)
            return pred, reason, raw, total_ms

        # pred=None -> either parse miss or timeout (provider should put error='timeout' in raw)
        provider = (raw or {}).get("provider", "unknown")
        if raw and raw.get("error") == "timeout":
            # If the single-call returned its own elapsed ms, prefer that;
            # otherwise log the per-attempt ms we accumulated.
            per_attempt_ms = (raw.get("elapsed_ms") if isinstance(raw.get("elapsed_ms"), int) else ms) or 0
            log.warning(
                "LLM (%s) call timed out on attempt %d/%d after %d ms",
                provider, i, attempts, per_attempt_ms
            )
        else:
            log.warning(
                "LLM (%s) call returned no prediction on attempt %d/%d",
                provider, i, attempts
            )

        if not enabled or i == attempts:
            break

        log.debug("Retrying in %d ms…", backoff_ms)
        time.sleep(max(0, backoff_ms) / 1000.0)

    log.warning("LLM call failed after %d attempts; returning None", attempts)
    return None, last_reason, (last_raw or {}), total_ms
