import json
import hashlib
import psycopg2
import psycopg2.extras

from config import Pg


def connect(pg: Pg):
    dsn = f"host={pg.host} port={pg.port} dbname={pg.dbname} user={pg.user} password={pg.password}"
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    return conn


def ensure_audit_schema(conn, audit_schema: str):
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {audit_schema};")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {audit_schema}.llm_runs (
                run_id           BIGSERIAL PRIMARY KEY,
                created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                model            TEXT NOT NULL,
                prompt_hash      TEXT NOT NULL,
                notes            TEXT
            );
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {audit_schema}.llm_predictions (
                run_id           BIGINT NOT NULL REFERENCES {audit_schema}.llm_runs(run_id) ON DELETE CASCADE,
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


def start_run(conn, audit_schema: str, model: str, prompt_template: str, notes: str | None) -> int:
    h = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {audit_schema}.llm_runs (model, prompt_hash, notes) VALUES (%s,%s,%s) RETURNING run_id;",
            (model, h, notes)
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


def fetch_qrels(conn, data_schema: str, limit: int | None):
    limit_clause = "LIMIT %s" if limit is not None else ""
    sql = f"""
        SELECT
            qr.query_id,
            q.text AS query_text,
            qr.doc_id,
            d.text AS doc_text,
            qr.relevance AS gold_score
        FROM {data_schema}.qrels qr
        JOIN {data_schema}.queries q ON q.query_id = qr.query_id
        JOIN {data_schema}.docs    d ON d.doc_id   = qr.doc_id
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


def insert_prediction(conn, audit_schema: str, run_id: int, idx: int, row, pred, is_correct, ms_total, raw):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {audit_schema}.llm_predictions
            (run_id, idx, query_id, doc_id, gold_score, pred_score, is_correct, ms_total, raw_response)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """,
            (
                run_id, idx, row["query_id"], row["doc_id"], int(row["gold_score"]),
                pred, is_correct, ms_total,
                json.dumps(raw)
            )
        )
