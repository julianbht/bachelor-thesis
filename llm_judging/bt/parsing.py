import json
import re
from typing import Optional

"""
Module concerned with parsing the LLM response. 
"""

def extract_json_block(text: str) -> Optional[str]:
    """
    Return the first syntactically valid JSON object found in `text`,
    or None if no valid {...} block is present.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        json.loads(candidate)
        return candidate
    except Exception:
        return None
