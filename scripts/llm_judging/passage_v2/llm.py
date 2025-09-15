import json
import time
import ollama


def judge_with_ollama(model: str, prompt: str) -> tuple[int | None, dict, int]:
    """
    Returns (pred_score or None, raw_response_dict, elapsed_ms).
    """
    t0 = time.time()
    res = ollama.generate(
        model=model,
        prompt=prompt,
        format="json",
        options={"temperature": 0.0},
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
