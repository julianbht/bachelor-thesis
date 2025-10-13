# pipeline.py
from __future__ import annotations
import logging
import time

from bt.config import Settings
from bt.util.logging_utils import setup_run_logger
from bt.db import (
    connect, ensure_audit_schema,
    insert_prediction, count_available_qrels, finalize_run,
)
from bt.prompts import PROMPT_TMPL, PROMPT_TMPL_WITH_REASON, build_prompt
from bt.llm.factory import build_llm_client  
from bt.util.git import get_git_info
import json

from bt.util.helpers import (
    validate_range_and_limit,
    compute_qrel_window,
    log_qrel_banner,
    choose_prompt_template,
    start_run_from_cfg,
    fetch_items_with_window,
)

def _hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _truncate(text: str, limit: int | None) -> str:
    if limit is None or len(text) <= limit:
        return text
    return text[:limit]


def run_once(cfg: Settings, *, run_key: str, non_interactive: bool = True) -> None:
    """
    Orchestrates a single run using a provider-agnostic LLM client.
    """
    # ---- Per-run logging FIRST so all subsequent logs (incl. bt.db) show up
    log, log_path = setup_run_logger(run_key)
    root = logging.getLogger("bt")

    root.info("Run settings:\n%s", json.dumps(cfg.__dict__, indent=2, default=str))
    conn = connect()

    # Build the LLM client (Ollama or HF endpoint) from cfg
    client = build_llm_client(cfg)

    try:
        ensure_audit_schema(conn, cfg.audit_schema)

        total_available = count_available_qrels(conn, cfg.data_schema)

        # Range & limit validation + window computation
        validate_range_and_limit(cfg.start_qrel, cfg.end_qrel, cfg.limit_qrels)
        window = compute_qrel_window(
            total_available=total_available,
            start_qrel=cfg.start_qrel,
            end_qrel=cfg.end_qrel,
            limit_qrels=cfg.limit_qrels,
        )

        prompt_template = choose_prompt_template(
            cfg.reasoning_enabled, PROMPT_TMPL_WITH_REASON, PROMPT_TMPL
        )

        git = get_git_info()
        if git:
            log.info("Code version: %s (%s)%s",
                     git.commit, git.branch, " +dirty" if git.dirty else "")

        log_qrel_banner(log, cfg, window, total_available)

        # Persist run metadata (incl. range)
        start_run_from_cfg(
            conn=conn,
            audit_schema=cfg.audit_schema if False else cfg.audit_schema,  
            run_key=run_key,
            client=client,
            prompt_template=prompt_template,
            cfg=cfg,
            git=git,
        )

        # Fetch items with start/end/limit applied
        items = fetch_items_with_window(
            conn, cfg.data_schema, cfg.start_qrel, cfg.end_qrel, cfg.limit_qrels
        )

        n = len(items)
        if n == 0:
            log.warning("No qrels found for the requested window. Finalizing empty run.")
            finalize_run(conn, cfg.audit_schema, run_key)
            log.info("Run %s finished (empty). Detailed log at: %s", run_key, log_path)
            return

        correct = 0
        counted = 0
        t_start = time.time()

        for i, row in enumerate(items, start=1):
            query_text = (row["query_text"] or "").strip()
            doc_text_full = (row["doc_text"] or "").strip()
            doc_text = _truncate(doc_text_full, cfg.max_text_chars)

            prompt = build_prompt(query_text, doc_text, template=prompt_template)

            log.info("Processing item %d/%d | qid=%s doc=%s", i, n, row["query_id"], row["doc_id"])

            try:
                log.debug("=== Prompt: ===\n%s", prompt)
                pred, reason, raw, ms_total = client.judge(prompt)
                log.debug("=== Response: ===\n%s", raw.get("response_text"))

            except Exception:
                log.exception("LLM call failed for qid=%s doc=%s", row["query_id"], row["doc_id"])
                pred, reason, raw, ms_total = None, None, {"error": "exception during LLM call"}, 0

            is_correct = None
            if pred is not None:
                is_correct = (pred == int(row["gold_score"]))
                counted += 1
                if is_correct:
                    correct += 1

            status = "HIT" if is_correct else ("MISS" if pred is not None else "N/A")
            agree_pct = (100.0 * correct / counted) if counted else 0.0
            log.info(
                "Item %d/%d | qid=%s doc=%s | gold=%s → pred=%s | %s | ms=%s | agree-so-far=%d/%d (%.2f%%)",
                i, n, row["query_id"], row["doc_id"], row["gold_score"], pred, status, ms_total,
                correct, counted, agree_pct,
            )

            insert_prediction(conn, cfg.audit_schema, run_key, i, row, pred, reason, is_correct, ms_total, raw)

            if cfg.commit_every and (i % cfg.commit_every == 0):
                conn.commit()
                log.debug("Committed batch at item %d", i)

        conn.commit()

        total_agree = (100.0 * correct / counted) if counted > 0 else 0.0
        total_time = time.time() - t_start
        invalid_pct = finalize_run(conn, cfg.audit_schema, run_key)

        log.info(
            "Done | items=%d | valid_preds=%d | agreement=%.2f%% | invalid_preds=%.2f%% | time=%s",
            n, counted, total_agree, invalid_pct, _hms(total_time)
        )
        log.info("Run %s finished. Detailed log at: %s", run_key, log_path)

    finally:
        # Close client first (releases HTTP sessions), then DB
        try:
            client.close()
        except Exception:
            logging.getLogger("bt").exception("Failed to close LLM client")
        try:
            conn.close()
        except Exception:
            logging.getLogger("bt").exception("Failed to close DB connection")
