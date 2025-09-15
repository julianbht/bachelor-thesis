import json
import time
import ollama


import json
import time
import ollama

def _single_call(model: str, prompt: str, temperature: float) -> tuple[int | None, dict, int]:
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
        if 0 <= score <= 3:
            return score, raw, elapsed_ms
        return None, raw, elapsed_ms
    except Exception:
        return None, raw, elapsed_ms


def judge_with_ollama(
    model: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    attempts: int = 1,
    enabled: bool = True,
    backoff_ms: int = 500,
) -> tuple[int | None, dict, int]:
    """
    Returns (pred_score or None, raw_response_dict, elapsed_ms_total).
    Retries parsing/LLM failures up to `attempts` if `enabled` is True.
    """
    attempts = max(1, int(attempts))
    total_ms = 0
    last_raw: dict | None = None
    for i in range(1, attempts + 1):
        try:
            pred, raw, ms = _single_call(model, prompt, temperature)
            total_ms += ms
            last_raw = raw
            if pred is not None:
                return pred, raw, total_ms
        except Exception as e:
            total_ms += 0
            last_raw = {"error": repr(e)}

        if not enabled or i == attempts:
            break
        time.sleep(backoff_ms / 1000.0)

    return None, (last_raw or {}), total_ms

