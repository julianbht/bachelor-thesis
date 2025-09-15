import os
import json
import time
import hashlib
import argparse
from dataclasses import dataclass
from datetime import timedelta

import psycopg2
import psycopg2.extras
import ollama


# ---- Config ----
@dataclass(frozen=True)
class Pg:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "bachelor-thesis"
    user: str = "postgres"
    password: str = "123"
    schema: str = "llm1"

PG = Pg()
MODEL = "llama3.2:3b"

PROMPT_TMPL = """You are a relevance judge for document retrieval.
Rate how relevant the DOCUMENT is to the user QUERY on a 0–3 scale:

0 = Irrelevant: The passage has nothing to do with the query.
1 = Related: The passage seems related to the query but does not answer it.
2 = Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.
3 = Perfectly relevant: The passage is dedicated to the query and contains the exact answer.

Return strict JSON ONLY with key: score (0,1,2,3).

QUERY:
{query}

DOCUMENT (title then body):
{title}
{body}
"""

MAX_TITLE_CHARS = 500
MAX_BODY_CHARS = 40000000


# ---- Args ----
def parse_args():
    ap = argparse.ArgumentParser(description="Judge MS MARCO qrels via Ollama and store results in Postgres.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Number of qrels to judge (default: all).")
    ap.add_argument("--notes", type=str, default=None,
                    help="Optional notes to store with this run.")
    return ap.parse_args()


# ---- DB helpers ----
def connect():
    dsn = f"host={PG.host} port={PG.port} dbname={PG.dbname} user={PG.user} password={PG.password}"
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    return conn

def ensure_audit_schema(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {PG.schema}.llm_runs (
            run_id           BIGSERIAL PRIMARY KEY,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            model            TEXT NOT NULL,
            prompt_hash      TEXT NOT NULL,
            notes            TEXT
        );
        """)
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {PG.schema}.llm_predictions (
            run_id           BIGINT NOT NULL REFERENCES {PG.schema}.llm_runs(run_id) ON DELETE CASCADE,
            idx              INTEGER NOT NULL,
            query_id         TEXT NOT NULL,
            doc_id           TEXT NOT NULL,
            gold_score       INTEGER NOT NULL,
            pred_score       INTEGER,
            is_correct       BOOLEAN,
            ms_total         INTEGER NOT NULL,
            raw_response     JSONB NOT NULL,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (run_id, idx)
        );
        """)
    conn.commit()

def start_run(conn, model: str, prompt_template: str, notes: str|None=None) -> int:
    h = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {PG.schema}.llm_runs (model, prompt_hash, notes) VALUES (%s,%s,%s) RETURNING run_id;",
            (model, h, notes)
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id

def fetch_qrels(conn, limit: int|None):
    limit_clause = "LIMIT %s" if limit is not None else ""
    sql = f"""
        SELECT
            qr.query_id,
            q.text AS query_text,
            qr.doc_id,
            COALESCE(d.title,'') AS title,
            COALESCE(d.body,'')  AS body,
            qr.relevance AS gold_score
        FROM {PG.schema}.qrels qr
        JOIN {PG.schema}.queries q ON q.query_id = qr.query_id
        JOIN {PG.schema}.docs d    ON d.doc_id   = qr.doc_id
        ORDER BY qr.query_id, qr.doc_id
        {limit_clause};
    """
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        if limit is not None:
            cur.execute(sql, (limit,))
        else:
            cur.execute(sql)
        rows = cur.fetchall()
        return [dict(r) for r in rows]


# ---- LLM call ----
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


# ---- Utilities ----
def hms(seconds: float) -> str:
    return str(timedelta(seconds=int(max(0, seconds))))


def main():
    args = parse_args()

    conn = connect()
    try:
        ensure_audit_schema(conn)
        note = args.notes or f"Run with limit={args.limit}; no rationale; strict JSON; NULL on parse fail."
        run_id = start_run(conn, MODEL, PROMPT_TMPL, notes=note)

        items = fetch_qrels(conn, limit=args.limit)
        n = len(items)
        if n == 0:
            print("No qrels found (check tables and/or limit).")
            return

        correct = 0
        counted = 0
        t_start = time.time()

        with conn.cursor() as cur:
            for i, row in enumerate(items, start=1):
                qid = row["query_id"]
                did = row["doc_id"]
                gold = int(row["gold_score"])

                query_text = (row["query_text"] or "").strip()
                title = (row["title"] or "").strip()[:MAX_TITLE_CHARS]
                body = (row["body"] or "").strip()[:MAX_BODY_CHARS]

                prompt = PROMPT_TMPL.format(query=query_text, title=title, body=body)

                pred, raw, ms_total = judge_with_ollama(MODEL, prompt)

                is_correct = None
                if pred is not None:
                    is_correct = (pred == gold)
                    counted += 1
                    if is_correct:
                        correct += 1

                agree = (100.0 * correct / counted) if counted > 0 else 0.0
                elapsed = time.time() - t_start
                avg_per = elapsed / i
                eta = avg_per * (n - i)

                print(
                    f"[{i:4d}/{n}] qid={qid} doc={did} gold={gold} pred={pred} "
                    f"| agreement={agree:.2f}% | ETA ~ {hms(eta)}"
                )

                cur.execute(
                    f"""
                    INSERT INTO {PG.schema}.llm_predictions
                    (run_id, idx, query_id, doc_id, gold_score, pred_score, is_correct, ms_total, raw_response)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);
                    """,
                    (
                        run_id, i, qid, did, gold,
                        pred, is_correct, ms_total,
                        json.dumps(raw)
                    )
                )

                if i % 50 == 0:
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
