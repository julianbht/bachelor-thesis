# bt/util/helpers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

def validate_range_and_limit(start_qrel: Optional[int], end_qrel: Optional[int], limit_qrels: Optional[int]) -> None:
    if limit_qrels is not None and limit_qrels <= 0:
        raise ValueError("limit_qrels must be positive or None")
    if start_qrel is not None and start_qrel <= 0:
        raise ValueError("start_qrel must be positive (1-based) if provided.")
    if end_qrel is not None and end_qrel <= 0:
        raise ValueError("end_qrel must be positive (1-based) if provided.")
    if (start_qrel is not None and end_qrel is not None) and (start_qrel > end_qrel):
        raise ValueError("start_qrel cannot be greater than end_qrel.")

@dataclass(frozen=True)
class QrelWindow:
    start_1b: int                # effective 1-based start used for counting
    end_1b: int                  # effective 1-based end used for counting
    intended_count: int          # count implied by [start,end] ∩ total
    processed_target: int        # count after applying limit_qrels
    is_subset: bool              # whether processed_target < total_available

def compute_qrel_window(total_available: int,
                        start_qrel: Optional[int],
                        end_qrel: Optional[int],
                        limit_qrels: Optional[int]) -> QrelWindow:
    """
    Convert 1-based inclusive [start_qrel, end_qrel] and an optional limit into a concrete
    window against total_available.
    """
    s = start_qrel if (start_qrel and start_qrel > 0) else 1
    e = end_qrel if (end_qrel and end_qrel > 0) else total_available

    # Clamp to [1, total_available]
    s_eff = max(1, min(s, total_available)) if total_available > 0 else 1
    e_eff = max(1, min(e, total_available)) if total_available > 0 else 1

    intended = max(0, e_eff - s_eff + 1) if total_available > 0 else 0
    processed = min(intended, limit_qrels) if (limit_qrels is not None) else intended
    is_subset = processed < total_available

    return QrelWindow(
        start_1b=s_eff,
        end_1b=e_eff,
        intended_count=intended,
        processed_target=processed,
        is_subset=is_subset,
    )

def log_qrel_banner(logger, cfg, window: QrelWindow, total_available: int) -> None:
    logger.info(
        "QREL window: start=%s end=%s limit=%s → target=%d (intended=%d) of total=%d",
        getattr(cfg, "start_qrel", None),
        getattr(cfg, "end_qrel", None),
        getattr(cfg, "limit_qrels", None),
        window.processed_target,
        window.intended_count,
        total_available,
    )

def ensure_official_guard(official: bool, is_subset: bool) -> None:
    if official and is_subset:
        raise ValueError("Cannot mark run 'official' when processing only a subset of qrels (range and/or limit).")

def choose_prompt_template(reasoning_enabled: bool, tmpl_with_reason, tmpl_plain):
    return tmpl_with_reason if reasoning_enabled else tmpl_plain

def start_run_from_cfg(
    *,
    conn,
    audit_schema: str,
    run_key: str,
    client,
    prompt_template: str,
    cfg,
    git,
):
    from ..db import start_run as _start_run  # local import to avoid cycles

    _start_run(
        conn,
        audit_schema,
        run_key=run_key,
        model=client.model_label,
        prompt_template=prompt_template,
        data_schema=cfg.data_schema,
        audit_schema_name=cfg.audit_schema,
        max_text_chars=(cfg.max_text_chars if cfg.max_text_chars is not None else None),
        commit_every=cfg.commit_every,
        limit_qrels=cfg.limit_qrels,
        temperature=cfg.temperature,
        retry_enabled=cfg.retry_enabled,
        retry_attempts=cfg.retry_attempts,
        retry_backoff_ms=cfg.retry_backoff_ms,
        runner="pipeline.run_once",
        official=cfg.official,
        user_notes=cfg.user_notes,
        git_commit=(git.commit if git else None),
        git_branch=(git.branch if git else None),
        git_dirty=(git.dirty if git else False),
        start_qrel=getattr(cfg, "start_qrel", None),
        end_qrel=getattr(cfg, "end_qrel", None),
    )

def fetch_items_with_window(conn, data_schema: str, start_qrel: Optional[int], end_qrel: Optional[int], limit_qrels: Optional[int]):
    from ..db import fetch_qrels as _fetch_qrels  # local import to avoid cycles
    return _fetch_qrels(
        conn,
        data_schema,
        start=start_qrel,
        end=end_qrel,
        limit=limit_qrels,
    )
