# llm.py
import json
import time
import ollama

def ensure_model_downloaded(model: str, *, retries: int = 3, backoff_ms: int = 500) -> None:
    """
    Ensure the Ollama model is present locally. If not, pull it before running.
    """
    # Quick existence check
    try:
        ollama.show(model=model)
        return
    except Exception:
        pass  # Not present -> pull

    last_err = None
    for i in range(1, max(1, retries) + 1):
        try:
            # Pull with streaming so large models are fetched progressively
            for _ in ollama.pull(model=model, stream=True):
                pass
            # Verify after pull
            ollama.show(model=model)
            return
        except Exception as e:
            last_err = e
            if i == retries:
                break
            time.sleep(backoff_ms / 1000.0)
    raise RuntimeError(f"Failed to ensure model '{model}' is available: {last_err!r}")


def _single_call(model: str, prompt: str, temperature: float) -> tuple[int | None, str | None, dict, int]:
    t0 = time.time()
    res = ollama.generate(
        model=model,
        prompt=prompt,
        format="json",
        options={"temperature": float(temperature)},
    )
    elapsed_ms = int((time.time() - t0) * 1000)

    raw_text = res.response
    raw_meta = res.model_dump()
    raw_meta.pop("response", None)
    raw = {"ollama": raw_meta, "response_text": raw_text}

    try:
        obj = json.loads(raw_text)
        score = int(obj["score"])
        reason = obj.get("reason")
        if reason is not None and not isinstance(reason, str):
            reason = str(reason)
        if 0 <= score <= 3:
            return score, reason, raw, elapsed_ms
        return None, reason, raw, elapsed_ms
    except Exception:
        return None, None, raw, elapsed_ms


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
        try:
            pred, reason, raw, ms = _single_call(model, prompt, temperature)
            total_ms += ms
            last_raw = raw
            last_reason = reason
            if pred is not None:
                return pred, reason, raw, total_ms
        except Exception as e:
            total_ms += 0
            last_raw = {"error": repr(e)}

        if not enabled or i == attempts:
            break
        time.sleep(backoff_ms / 1000.0)

    return None, last_reason, (last_raw or {}), total_ms
