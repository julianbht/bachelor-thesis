from __future__ import annotations
import json
import re
from typing import Optional, Tuple

def extract_json_block(text: str) -> Optional[str]:
    """
    Return the first syntactically valid JSON object found in `text`,
    or None if no valid {...} block is present.
    """
    s = _strip_code_fences(text)
    return _find_first_json_object(s)


def parse_score_and_reason(text: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse (score, reason) from model output with two simple rules:
      1) Valid JSON: keys are case-insensitive; "score" required; "reason" optional.
      2) Fallback: textual pattern `score: [0..3]` (case-insensitive). Reason = None.
    """
    # 1) Strict JSON path (case-insensitive keys)
    block = extract_json_block(text)
    if block is not None:
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                score = _get_ci_key(obj, "score")
                if score is not None:
                    score = _normalize_score(score)
                    if score is not None:
                        reason_val = _get_ci_key(obj, "reason")
                        reason = str(reason_val) if isinstance(reason_val, (str, int, float, bool)) else None
                        return score, reason
        except Exception:
            pass  # fall through to textual fallback

    # 2) Textual fallback: "score: 0..3"
    m = _RE_SCORE_KV.search(text)
    if m:
        return int(m.group(1)), None

    return None, None


# --- Internals ---------------------------------------------------------------

_RE_FENCE = re.compile(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE)
_RE_SCORE_KV = re.compile(r"\bscore\s*[:=]\s*([0-3])\b", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    m = _RE_FENCE.match(text.strip())
    return m.group(1).strip() if m else text


def _get_ci_key(d: dict, name: str):
    name = name.lower()
    for k, v in d.items():
        if isinstance(k, str) and k.lower() == name:
            return v
    return None


def _normalize_score(value) -> Optional[int]:
    try:
        iv = int(value)
        return iv if 0 <= iv <= 3 else None
    except Exception:
        return None


def _find_first_json_object(s: str) -> Optional[str]:
    """
    Scan for the first balanced {...} block and return it if json.loads succeeds.
    Minimal but robust enough for our needs.
    """
    depth = 0
    in_str = False
    esc = False
    start = None

    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue
        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = s[start:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        # keep scanning; maybe a later block is valid
                        start = None
                        continue
    return None
