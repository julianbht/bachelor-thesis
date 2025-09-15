from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import hashlib
import json
import platform
import socket
import sys


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class RunNotes:
    # core
    model: str
    prompt_template: str
    max_text_chars: int

    # data / storage
    data_schema: str
    audit_schema: str

    # runner knobs
    commit_every: Optional[int] = None
    limit: Optional[int] = None
    temperature: float = 0.0

    # free-form
    user_notes: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None  # e.g., {"git_sha": "...", "host_gpu": "RTX 4090"}

    def render(self, include_system: bool = True) -> str:
        """
        Returns a compact, human-readable note string for llm_runs.notes.
        JSON is used so it pastes well and is easy to parse later.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        prompt_info = {
            "sha256": sha256_hex(self.prompt_template),
            "first_line": self.prompt_template.strip().splitlines()[0][:160] if self.prompt_template else "",
            "chars": len(self.prompt_template or ""),
        }

        sys_info = None
        if include_system:
            sys_info = {
                "started_at_utc": now,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
                "hostname": socket.gethostname(),
            }

        payload = {
            "run": {
                "model": self.model,
                "temperature": self.temperature,
                "max_text_chars": self.max_text_chars,
                "limit": self.limit,
                "commit_every": self.commit_every,
            },
            "data": {
                "data_schema": self.data_schema,
                "audit_schema": self.audit_schema,
            },
            "prompt": prompt_info,
            "notes": (self.user_notes or "").strip() or None,
            "extras": self.extras or None,
            "system": sys_info,
        }

        # drop None keys for cleanliness
        def drop_nones(obj):
            if isinstance(obj, dict):
                return {k: drop_nones(v) for k, v in obj.items() if v is not None}
            if isinstance(obj, list):
                return [drop_nones(x) for x in obj]
            return obj

        return json.dumps(drop_nones(payload), ensure_ascii=False, separators=(",", ":"))
