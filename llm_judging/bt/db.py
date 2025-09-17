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

def _gen_run_key(n: int = 12) -> str:
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
    Creates fresh tables using run_key (TEXT) as the sole primary key.
    Safe to call after you dropped the old tables.
    """
    log.info("Ensuring audit schema exists: %s", audit_schema)
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {audit_schema};")

        # Runs (PRIMARY KEY = run_key)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {audit_schema}.llm_runs (
                run_key          TEXT PRIMARY KEY,
                created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                model            TEXT NOT NULL,
                prompt_hash      TEXT NOT NULL,
                prompt_template  TEXT NOT NULL,

                data_schema      TEXT NOT NULL,
                audit_schema     TEXT NOT NULL,
                max_text_chars   INTEGER,
                commit_every     INTEGER NOT NULL,
                limit_qrels      INTEGER,
                temperature      REAL NOT NULL,

                retry_enabled    BOOLEAN NOT NULL DEFAULT FALSE,
                retry_attempts   INTEGER NOT NULL DEFAULT 1,
                retry_backoff_ms INTEGER NOT NULL DEFAULT 500,

                runner           TEXT NOT NULL,
                official         BOOLEAN NOT NULL DEFAULT FALSE,
                user_notes       TEXT,

                finished         BOOLEAN NOT NULL DEFAULT FALSE,
                finished_at      TIMESTAMPTZ,
                invalid_pct      REAL,

                -- keep the key format tight (12 chars base32 alphabet)
                CONSTRAINT llm_runs_run_key_chk CHECK (run_key ~ '^[A-Z2-9]{{12}}$')
            );
        """)

        cur.execute(f"CREATE INDEX IF NOT EXISTS llm_runs_created_at_idx ON {audit_schema}.llm_runs(created_at DESC);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS llm_runs_model_idx      ON {audit_schema}.llm_runs(model);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS llm_runs_official_idx   ON {audit_schema}.llm_runs(official);")

        # Predictions (FK references run_key)
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
) -> str:
    """
    Inserts a new run and returns its run_key (12-char Crockford base32).
    """
    prompt_hash = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()

    # allocate a unique run_key (retry only on extremely unlikely collision)
    for _ in range(5):
        run_key = _gen_run_key()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {audit_schema}.llm_runs
                    (run_key,
                     model, prompt_hash, prompt_template,
                     data_schema, audit_schema, max_text_chars, commit_every, limit_qrels, temperature,
                     retry_enabled, retry_attempts, retry_backoff_ms,
                     runner, official, user_notes)
                    VALUES
                    (%s,
                     %s,%s,%s,
                     %s,%s,%s,%s,%s,%s,
                     %s,%s,%s,
                     %s,%s,%s)
                    """,
                    (
                        run_key,
                        model, prompt_hash, prompt_template,
                        data_schema, audit_schema_name, max_text_chars, commit_every, limit_qrels, float(temperature),
                        bool(retry_enabled), int(retry_attempts), int(retry_backoff_ms),
                        runner, bool(official), user_notes,
                    )
                )
            conn.commit()
            log.info(
                "Run started: key=%s | model=%s data=%s audit=%s limit_qrels=%s temp=%.2f official=%s",
                run_key, model, data_schema, audit_schema, limit_qrels, temperature, official
            )
            log.debug("Prompt hash: %s", prompt_hash[:16])
            return run_key
        except Exception as e:
            # unique violation → try again
            if getattr(e, "pgcode", None) == "23505":
                conn.rollback()
                continue
            conn.rollback()
            raise

    raise RuntimeError("Failed to allocate a unique run_key after several attempts")


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
    log.info("Fetching qrels (schema=%s, limit=%s)…", data_schema, limit)
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        if limit is not None:
            cur.execute(sql, (limit,))
        else:
            cur.execute(sql)
        rows = cur.fetchall()
        log.info("Fetched %d qrels.", len(rows))
        return [dict(r) for r in rows]


def insert_prediction(conn, audit_schema: str, run_key: str, idx: int, row, pred, pred_reason, is_correct, ms_total, raw):
    log.debug(
        "Insert prediction | key=%s idx=%s qid=%s doc=%s gold=%s pred=%s correct=%s ms=%s",
        run_key, idx, row["query_id"], row["doc_id"], int(row["gold_score"]), pred, is_correct, ms_total
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
