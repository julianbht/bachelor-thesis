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

        # Runs table
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {audit_schema}.llm_runs (
                run_id            BIGSERIAL PRIMARY KEY,
                created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                model             TEXT NOT NULL,
                prompt_hash       TEXT NOT NULL,
                prompt_template   TEXT NOT NULL,

                data_schema       TEXT NOT NULL,
                audit_schema      TEXT NOT NULL,
                max_text_chars    INTEGER,
                commit_every      INTEGER NOT NULL,
                limit_qrels       INTEGER,
                temperature       REAL NOT NULL,

                retry_enabled     BOOLEAN NOT NULL DEFAULT FALSE,
                retry_attempts    INTEGER NOT NULL DEFAULT 1,
                retry_backoff_ms  INTEGER NOT NULL DEFAULT 500,

                runner            TEXT NOT NULL,
                official          BOOLEAN NOT NULL DEFAULT FALSE,
                user_notes        TEXT,

                -- NEW
                finished          BOOLEAN NOT NULL DEFAULT FALSE,
                finished_at       TIMESTAMPTZ,
                invalid_pct       REAL
            );
        """)

        cur.execute(f"CREATE INDEX IF NOT EXISTS llm_runs_created_at_idx ON {audit_schema}.llm_runs(created_at DESC);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS llm_runs_model_idx      ON {audit_schema}.llm_runs(model);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS llm_runs_official_idx   ON {audit_schema}.llm_runs(official);")

        # Predictions table (already has pred_reason from previous change)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {audit_schema}.llm_predictions (
                run_id           BIGINT NOT NULL REFERENCES {audit_schema}.llm_runs(run_id) ON DELETE CASCADE,
                idx              INTEGER NOT NULL,
                query_id         TEXT NOT NULL,
                doc_id           TEXT NOT NULL,
                gold_score       INTEGER NOT NULL,
                pred_score       INTEGER,
                pred_reason      TEXT,
                is_correct       BOOLEAN,
                ms_total         INTEGER NOT NULL,
                raw_response     JSONB NOT NULL,
                created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (run_id, idx)
            );
        """)
        cur.execute(f"ALTER TABLE {audit_schema}.llm_predictions ADD COLUMN IF NOT EXISTS pred_reason TEXT;")

    conn.commit()


def finalize_run(conn, audit_schema: str, run_id: int):
    """
    Computes invalid percentage (pred_score IS NULL) for this run and marks it finished.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
              COUNT(*)::float AS total,
              COUNT(*) FILTER (WHERE pred_score IS NULL)::float AS invalid
            FROM {audit_schema}.llm_predictions
            WHERE run_id = %s;
            """,
            (run_id,)
        )
        total, invalid = cur.fetchone()
        invalid_pct = (invalid / total * 100.0) if total > 0 else 0.0

        cur.execute(
            f"""
            UPDATE {audit_schema}.llm_runs
            SET finished = TRUE,
                finished_at = NOW(),
                invalid_pct = %s
            WHERE run_id = %s;
            """,
            (invalid_pct, run_id)
        )
    conn.commit()
    return invalid_pct

def start_run(
    conn,
    audit_schema: str,
    *,
    model: str,
    prompt_template: str,
    data_schema: str,
    audit_schema_name: str,
    max_text_chars: int | None,
    commit_every: int,
    limit_qrels: int | None,
    temperature: float,
    retry_enabled: bool,
    retry_attempts: int,
    retry_backoff_ms: int,
    runner: str,
    official: bool,
    user_notes: str | None,
) -> int:
    prompt_hash = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {audit_schema}.llm_runs
            (model, prompt_hash, prompt_template,
             data_schema, audit_schema, max_text_chars, commit_every, limit_qrels, temperature,
             retry_enabled, retry_attempts, retry_backoff_ms,
             runner, official, user_notes)
            VALUES
            (%s,%s,%s,
             %s,%s,%s,%s,%s,%s,
             %s,%s,%s,
             %s,%s,%s)
            RETURNING run_id;
            """,
            (
                model, prompt_hash, prompt_template,
                data_schema, audit_schema_name, max_text_chars, commit_every, limit_qrels, float(temperature),
                bool(retry_enabled), int(retry_attempts), int(retry_backoff_ms),
                runner, bool(official), user_notes,
            )
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id

def count_available_qrels(conn, data_schema: str) -> int:
    sql = f"""
        SELECT COUNT(*) AS c
        FROM {data_schema}.qrels qr
        JOIN {data_schema}.queries q ON q.query_id = qr.query_id
        JOIN {data_schema}.docs    d ON d.doc_id   = qr.doc_id;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return int(cur.fetchone()[0])

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

def insert_prediction(conn, audit_schema: str, run_id: int, idx: int, row, pred, pred_reason, is_correct, ms_total, raw):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {audit_schema}.llm_predictions
            (run_id, idx, query_id, doc_id, gold_score, pred_score, pred_reason, is_correct, ms_total, raw_response)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """,
            (
                run_id, idx, row["query_id"], row["doc_id"], int(row["gold_score"]),
                pred, pred_reason, is_correct, ms_total,
                json.dumps(raw)
            )
        )
