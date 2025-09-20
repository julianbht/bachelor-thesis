# bt/db.py
from __future__ import annotations
import json
import hashlib
import logging
import secrets
import psycopg2
import psycopg2.extras

from bt.config import Pg

log = logging.getLogger("bt.db")

# Crockford Base32 (no I, L, O, U, 0, 1)
ALPHABET = "ABCDEFGHJKMNPQRSTVWXYZ23456789"

def gen_run_key(n: int = 12) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(n))


def connect(pg: Pg = Pg()):
    dsn = f"host={pg.host} port={pg.port} dbname={pg.dbname} user={pg.user} password={pg.password}"
    log.info("Connecting to Postgres (host=%s port=%s db=%s user=%s)", pg.host, pg.port, pg.dbname, pg.user)
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    log.debug("Connection established; autocommit=%s", conn.autocommit)
    return conn


def ensure_audit_schema(conn, audit_schema: str):
    """
    Creates/updates audit tables. Safe to call repeatedly.
    """
    log.info("Ensuring audit schema exists: %s", audit_schema)
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {audit_schema};")

        # Runs
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {audit_schema}.llm_runs (
                run_key           TEXT PRIMARY KEY,
                created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                -- metadata about how the run was executed
                model             TEXT NOT NULL,
                prompt_template   TEXT,
                data_schema       TEXT NOT NULL,
                audit_schema_name TEXT NOT NULL,
                max_text_chars    INTEGER,
                commit_every      INTEGER,
                limit_qrels       INTEGER,
                temperature       DOUBLE PRECISION,
                retry_enabled     BOOLEAN,
                retry_attempts    INTEGER,
                retry_backoff_ms  INTEGER,
                runner            TEXT,
                official          BOOLEAN DEFAULT FALSE,
                user_notes        TEXT,

                -- structured git metadata
                git_commit        TEXT,     -- e.g. 'a1b2c3d'
                git_branch        TEXT,     -- e.g. 'main'
                git_dirty         BOOLEAN NOT NULL DEFAULT FALSE,

                -- run results
                total_items       INTEGER DEFAULT 0,
                valid_predictions INTEGER DEFAULT 0,
                agreement_pct     DOUBLE PRECISION,
                invalid_pct       DOUBLE PRECISION,
                finished_at       TIMESTAMPTZ
            );
        """)

        cur.execute(f"ALTER TABLE {audit_schema}.llm_runs ADD COLUMN IF NOT EXISTS finished BOOLEAN NOT NULL DEFAULT FALSE;")
        cur.execute(f"ALTER TABLE {audit_schema}.llm_runs ADD COLUMN IF NOT EXISTS start_qrel INTEGER;")
        cur.execute(f"ALTER TABLE {audit_schema}.llm_runs ADD COLUMN IF NOT EXISTS end_qrel   INTEGER;")


        cur.execute(f"CREATE INDEX IF NOT EXISTS llm_runs_created_at_idx ON {audit_schema}.llm_runs(created_at DESC);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS llm_runs_model_idx      ON {audit_schema}.llm_runs(model);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS llm_runs_official_idx   ON {audit_schema}.llm_runs(official);")

        # Predictions
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {audit_schema}.llm_predictions (
                run_key          TEXT NOT NULL REFERENCES {audit_schema}.llm_runs(run_key) ON DELETE CASCADE,
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
                PRIMARY KEY (run_key, idx)
            );
        """)
    conn.commit()
    log.debug("Audit schema ensured and committed: %s", audit_schema)


def finalize_run(conn, audit_schema: str, run_key: str):
    """
    Computes invalid percentage (pred_score IS NULL) for this run_key and marks it finished.
    """
    log.info("Finalizing run key=%s (computing invalid percentage)…", run_key)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
              COUNT(*)::float AS total,
              COUNT(*) FILTER (WHERE pred_score IS NULL)::float AS invalid
            FROM {audit_schema}.llm_predictions
            WHERE run_key = %s;
            """,
            (run_key,)
        )
        total, invalid = cur.fetchone()
        invalid_pct = (invalid / total * 100.0) if total and total > 0 else 0.0

        cur.execute(
            f"""
            UPDATE {audit_schema}.llm_runs
            SET finished = TRUE,
                finished_at = NOW(),
                invalid_pct = %s
            WHERE run_key = %s;
            """,
            (invalid_pct, run_key)
        )
    conn.commit()
    log.info("Run %s finalized | total=%s invalid=%s (%.2f%%)", run_key, int(total or 0), int(invalid or 0), invalid_pct)
    return invalid_pct


def start_run(
    conn,
    audit_schema: str,
    *,
    run_key: str,
    model: str,
    prompt_template: str | None,
    data_schema: str,
    audit_schema_name: str,
    max_text_chars: int | None,
    commit_every: int | None,
    limit_qrels: int | None,
    temperature: float | None,
    retry_enabled: bool | None,
    retry_attempts: int | None,
    retry_backoff_ms: int | None,
    runner: str,
    official: bool | None,
    user_notes: str | None,
    git_commit: str | None = None,
    git_branch: str | None = None,
    git_dirty: bool = False,
    start_qrel: int | None = None,
    end_qrel: int | None = None,
):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {audit_schema}.llm_runs
            (run_key, model, prompt_template, data_schema, audit_schema_name,
             max_text_chars, commit_every, limit_qrels, temperature,
             retry_enabled, retry_attempts, retry_backoff_ms, runner, official, user_notes,
             git_commit, git_branch, git_dirty,
             start_qrel, end_qrel)
            VALUES
            (%s,%s,%s,%s,%s,
             %s,%s,%s,%s,
             %s,%s,%s,%s,%s,%s,
             %s,%s,%s,
             %s,%s);
            """,
            (
                run_key, model, prompt_template, data_schema, audit_schema_name,
                max_text_chars, commit_every, limit_qrels, temperature,
                retry_enabled, retry_attempts, retry_backoff_ms, runner, official, user_notes,
                git_commit, git_branch, git_dirty,
                start_qrel, end_qrel,
            ),
        )
    conn.commit()
    log.info("Run started: key=%s | model=%s data=%s audit=%s", run_key, model, data_schema, audit_schema)
    return run_key


def count_available_qrels(conn, data_schema: str) -> int:
    sql = f"""
        SELECT COUNT(*) AS c
        FROM {data_schema}.qrels qr
        JOIN {data_schema}.queries q ON q.query_id = qr.query_id
        JOIN {data_schema}.docs    d ON d.doc_id   = qr.doc_id;
    """
    log.debug("Counting available qrels in schema=%s…", data_schema)
    with conn.cursor() as cur:
        cur.execute(sql)
        c = int(cur.fetchone()[0])
        log.info("Available qrels: %d (schema=%s)", c, data_schema)
        return c


def fetch_qrels(conn, data_schema: str, *, start: int | None, end: int | None, limit: int | None):
    """
    Fetch qrels in the default ORDER BY (query_id, doc_id), applying
    an inclusive 1-based [start, end] window and/or a hard limit.
    """
    # Compute OFFSET and LIMIT from start/end
    # start/end are 1-based inclusive; OFFSET is 0-based
    offset = max(0, (start - 1)) if (start and start > 0) else 0

    window_count: int | None = None
    if end and end > 0:
        if start and start > 0:
            window_count = max(0, end - start + 1)
        else:
            window_count = end  # "first end rows"

    # Final LIMIT is the min of window_count and limit if both are given
    if window_count is not None and limit is not None:
        final_limit = min(window_count, limit)
    else:
        final_limit = window_count if window_count is not None else limit

    limit_clause = "LIMIT %s" if final_limit is not None else ""
    offset_clause = "OFFSET %s" if offset else ""

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
        {limit_clause}
        {offset_clause};
    """
    log.info("Fetching qrels (schema=%s, start=%s, end=%s, limit=%s → final_limit=%s, offset=%s)…",
             data_schema, start, end, limit, final_limit, offset)
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        params = []
        if final_limit is not None:
            params.append(final_limit)
        if offset:
            params.append(offset)
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        log.info("Fetched %d qrels.", len(rows))
        return [dict(r) for r in rows]


def insert_prediction(conn, audit_schema: str, run_key: str, idx: int, row, pred, pred_reason, is_correct, ms_total, raw):
    log.debug(
        "Insert prediction | idx=%s qid=%s doc=%s gold=%s pred=%s correct=%s ms=%s",
        idx, row["query_id"], row["doc_id"], int(row["gold_score"]), pred, is_correct, ms_total
    )
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {audit_schema}.llm_predictions
            (run_key, idx, query_id, doc_id, gold_score, pred_score, pred_reason, is_correct, ms_total, raw_response)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """,
            (
                run_key, idx, row["query_id"], row["doc_id"], int(row["gold_score"]),
                pred, pred_reason, is_correct, ms_total, json.dumps(raw)
            )
        )
