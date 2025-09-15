import argparse
import time

from config import Pg, Settings
from db import connect, ensure_audit_schema, start_run, fetch_qrels, insert_prediction
from llm import judge_with_ollama
from prompts import PROMPT_TMPL, build_prompt
from notes import RunNotes


def hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_args():
    ap = argparse.ArgumentParser(description="Judge qrels via Ollama and store results in Postgres.")
    ap.add_argument("--limit", type=int, default=None, help="Number of qrels to judge (default: all).")
    ap.add_argument("--notes", type=str, default=None, help="Optional notes to store with this run.")
    ap.add_argument("--model", type=str, default=None, help="Override model (default from settings).")
    ap.add_argument("--commit-every", type=int, default=None, help="Commit every N inserts (default from settings).")
    return ap.parse_args()


def main():
    args = parse_args()
    pg = Pg()
    cfg = Settings()

    model = args.model or cfg.model
    commit_every = args.commit_every or cfg.commit_every

    conn = connect(pg)
    try:
        ensure_audit_schema(conn, cfg.audit_schema)
        note = args.notes or f"Run with limit={args.limit}; strict JSON; Passage text only; NULL on parse fail."
        run_id = start_run(conn, cfg.audit_schema, model, PROMPT_TMPL, notes=note)

        items = fetch_qrels(conn, cfg.data_schema, limit=args.limit)
        n = len(items)
        if n == 0:
            print("No qrels found (check tables/schemas and/or limit).")
            return

        print(f"Run ID: {run_id} | Model: {model} | Items: {n}")
        print(f"Schemas: data={cfg.data_schema} audit={cfg.audit_schema} | commit_every={commit_every}")

        correct = 0
        counted = 0
        t_start = time.time()

        for i, row in enumerate(items, start=1):
            query_text = (row["query_text"] or "").strip()
            doc_text   = (row["doc_text"]   or "").strip()[:cfg.max_text_chars]
            prompt = build_prompt(query_text, doc_text)

            pred, raw, ms_total = judge_with_ollama(model, prompt)

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

            if commit_every and (i % commit_every == 0):
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
