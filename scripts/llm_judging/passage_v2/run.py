import time

from config import Pg, Settings
from db import (
    connect,
    ensure_audit_schema,
    start_run,
    fetch_qrels,
    insert_prediction,
    count_available_qrels,
)
from llm import judge_with_ollama
from prompts import PROMPT_TMPL, build_prompt


def hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    pg = Pg()
    cfg = Settings()

    conn = connect(pg)
    try:
        ensure_audit_schema(conn, cfg.audit_schema)

        # Validate partial run vs. official
        total_available = count_available_qrels(conn, cfg.data_schema)
        limit_qrels = cfg.limit_qrels
        if limit_qrels is not None:
            if limit_qrels <= 0:
                raise ValueError("Settings.limit_qrels must be a positive integer or None.")
            if cfg.official and limit_qrels < total_available:
                raise ValueError(
                    f"Illegal state: cannot mark as official when processing only the first "
                    f"{limit_qrels} of {total_available} qrels."
                )

        # Create run record
        run_id = start_run(
            conn,
            cfg.audit_schema,
            model=cfg.model,
            prompt_template=PROMPT_TMPL,
            data_schema=cfg.data_schema,
            audit_schema_name=cfg.audit_schema,
            max_text_chars=cfg.max_text_chars if cfg.max_text_chars is not None else None,
            commit_every=cfg.commit_every,
            limit_qrels=limit_qrels,
            temperature=cfg.temperature,
            retry_enabled=cfg.retry_enabled,
            retry_attempts=cfg.retry_attempts,
            retry_backoff_ms=cfg.retry_backoff_ms,
            runner="passage_v2.run",
            official=cfg.official,
            user_notes=cfg.user_notes,
        )

        items = fetch_qrels(conn, cfg.data_schema, limit=limit_qrels)
        n = len(items)
        if n == 0:
            print("No qrels found (check tables/schemas and/or limit).")
            return

        scope = (
            f"first {n} of {total_available}"
            if (limit_qrels is not None and limit_qrels < total_available)
            else f"all {total_available}"
        )
        print(
            f"Run ID: {run_id} | Model: {cfg.model} | Items: {n} ({scope}) | "
            f"official={cfg.official} | data={cfg.data_schema} audit={cfg.audit_schema} | commit_every={cfg.commit_every}"
        )

        correct = 0
        counted = 0
        t_start = time.time()

        for i, row in enumerate(items, start=1):
            query_text = (row["query_text"] or "").strip()
            doc_text = (row["doc_text"] or "").strip()
            if cfg.max_text_chars is not None:
                doc_text = doc_text[:cfg.max_text_chars]
            prompt = build_prompt(query_text, doc_text)

            pred, raw, ms_total = judge_with_ollama(
                cfg.model,
                prompt,
                temperature=cfg.temperature,
                attempts=cfg.retry_attempts if cfg.retry_enabled else 1,
                enabled=cfg.retry_enabled,
                backoff_ms=cfg.retry_backoff_ms,
            )

            is_correct = None
            if pred is not None:
                is_correct = (pred == int(row["gold_score"]))
                counted += 1
                if is_correct:
                    correct += 1

            agree = (100.0 * correct / counted) if counted > 0 else 0.0
            elapsed = time.time() - t_start
            avg_per = elapsed / i
            eta = avg_per * (n - i)
            print(
                f"[{i:4d}/{n}] qid={row['query_id']} doc={row['doc_id']} "
                f"gold={row['gold_score']} pred={pred} "
                f"| agreement={agree:.2f}% | ETA ~ {hms(eta)}"
            )

            insert_prediction(conn, cfg.audit_schema, run_id, i, row, pred, is_correct, ms_total, raw)

            if cfg.commit_every and (i % cfg.commit_every == 0):
                conn.commit()

        conn.commit()

        total_agree = (100.0 * correct / counted) if counted > 0 else 0.0
        total_time = time.time() - t_start
        print("\nDone.")
        print(f"Run ID: {run_id}")
        print(f"Total items judged: {n} | Agreement (on {counted} valid preds): {total_agree:.2f}% | Time: {hms(total_time)}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
