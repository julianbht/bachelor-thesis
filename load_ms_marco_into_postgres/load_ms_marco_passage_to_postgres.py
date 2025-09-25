from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Tuple, List, Set, Dict
import random

import psycopg2
import psycopg2.extras
import ir_datasets


# ──────────────────────────────────────────────────────────────────────────────
# Config (hardcoded)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Pg:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "bachelor-thesis"
    user: str = "postgres"
    password: str = "123"
    schema: str = "passage"


DATASET_ID = "msmarco-passage/trec-dl-2019/judged"   # queries + qrels
DOC_DATASET_ID = "msmarco-passage"                   # docs via doc_store
PER_CLASS = 250                                      # per relevance class
RELEVANCE_CLASSES = (0, 1, 2, 3)
RANDOM_SEED = 42                                     # deterministic sampling


# ──────────────────────────────────────────────────────────────────────────────
# DB utils
# ──────────────────────────────────────────────────────────────────────────────

def connect(pg: Pg):
    dsn = (
        f"host={pg.host} port={pg.port} dbname={pg.dbname} "
        f"user={pg.user} password={pg.password}"
    )
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    return conn


def ensure_schema_and_tables(conn, schema: str) -> None:
    """
    Create tables to match bt/db.py expectations:
      queries(query_id TEXT, text TEXT)
      docs(doc_id TEXT, text TEXT)
      qrels(query_id TEXT, doc_id TEXT, relevance INT)
    """
    with conn.cursor() as cur:
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS "{schema}".queries (
                query_id TEXT PRIMARY KEY,
                text     TEXT NOT NULL
            );
        """)

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS "{schema}".docs (
                doc_id TEXT PRIMARY KEY,
                text   TEXT NOT NULL
            );
        """)

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS "{schema}".qrels (
                query_id  TEXT NOT NULL REFERENCES "{schema}".queries(query_id) ON DELETE CASCADE,
                doc_id    TEXT NOT NULL REFERENCES "{schema}".docs(doc_id)     ON DELETE CASCADE,
                relevance INT  NOT NULL,
                PRIMARY KEY (query_id, doc_id)
            );
        """)

        cur.execute(f'CREATE INDEX IF NOT EXISTS "qrels_query_id_idx" ON "{schema}".qrels(query_id);')
        cur.execute(f'CREATE INDEX IF NOT EXISTS "qrels_doc_id_idx"   ON "{schema}".qrels(doc_id);')

    conn.commit()


def execute_values_upsert(
    conn,
    schema: str,
    table: str,
    columns: List[str],
    rows: Iterable[Tuple],
    conflict_cols: List[str],
    update_cols: Optional[List[str]] = None,
    page_size: int = 10_000,
) -> None:
    it = iter(rows)
    chunk: List[Tuple] = []
    with conn.cursor() as cur:
        while True:
            chunk.clear()
            try:
                for _ in range(page_size):
                    chunk.append(next(it))
            except StopIteration:
                pass

            if not chunk:
                break

            cols_sql = ", ".join(f'"{c}"' for c in columns)
            conflict_sql = ", ".join(f'"{c}"' for c in conflict_cols)
            if update_cols:
                set_sql = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)
                on_conflict = f'ON CONFLICT ({conflict_sql}) DO UPDATE SET {set_sql}'
            else:
                on_conflict = f'ON CONFLICT ({conflict_sql}) DO NOTHING'
            sql = f'INSERT INTO "{schema}"."{table}" ({cols_sql}) VALUES %s {on_conflict};'
            psycopg2.extras.execute_values(cur, sql, chunk, page_size=page_size)
        conn.commit()


# ──────────────────────────────────────────────────────────────────────────────
# ir_datasets adapters
# ──────────────────────────────────────────────────────────────────────────────

def build_query_map(ds) -> Dict[str, str]:
    """Return {query_id(str): text} for TREC DL 2019 queries (passage)."""
    return {str(q.query_id): q.text for q in ds.queries_iter()}


def balanced_qrels(ds, per_class: int, classes: Iterable[int]) -> List[Tuple[str, str, int]]:
    """
    Return exactly per_class items for each relevance in classes, as (query_id:str, doc_id:str, relevance:int).
    Raises RuntimeError if any class has too few.
    """
    buckets: Dict[int, List[Tuple[str, str, int]]] = {c: [] for c in classes}

    for r in ds.qrels_iter():
        qid = str(r.query_id)
        did = str(r.doc_id)
        rel = int(r.relevance)
        if rel in buckets:
            buckets[rel].append((qid, did, rel))

    shortages = {rel: len(items) for rel, items in buckets.items() if len(items) < per_class}
    if shortages:
        details = ", ".join(f"rel={rel}: have {count}, need {per_class}" for rel, count in shortages.items())
        raise RuntimeError(f"Not enough judged qrels for balanced sampling: {details}")

    rng = random.Random(RANDOM_SEED)
    selected: List[Tuple[str, str, int]] = []
    for rel, items in buckets.items():
        rng.shuffle(items)
        selected.extend(items[:per_class])

    selected.sort(key=lambda t: (t[2], t[0], t[1]))  # stable order
    return selected


def fetch_passages_for_ids_list(doc_ds, doc_ids: List[str]) -> Tuple[List[Tuple[str, str]], Set[str]]:
    """
    Fetch (doc_id, text) for given doc_ids using doc_store ONLY (passage corpus).
    Returns (rows, found_ids).
    """
    store = getattr(doc_ds, "docs_store", None)
    if callable(store):
        store = store()
    if store is None:
        raise RuntimeError("Document dataset has no doc_store. Use 'msmarco-passage'.")

    rows: List[Tuple[str, str]] = []
    found: Set[str] = set()

    for did in doc_ids:
        d = store.get(did)
        if d is None:
            continue
        text = getattr(d, "text", None) or ""
        rows.append((d.doc_id, text))
        found.add(d.doc_id)

    return rows, found


# ──────────────────────────────────────────────────────────────────────────────
# ETL stages
# ──────────────────────────────────────────────────────────────────────────────

def load_queries(conn, schema: str, it: Iterable[Tuple[str, str]]) -> None:
    execute_values_upsert(conn, schema, "queries", ["query_id", "text"], it,
                          conflict_cols=["query_id"], update_cols=["text"])


def load_docs(conn, schema: str, it: Iterable[Tuple[str, str]]) -> None:
    execute_values_upsert(conn, schema, "docs", ["doc_id", "text"], it,
                          conflict_cols=["doc_id"], update_cols=["text"])


def load_qrels(conn, schema: str, it: Iterable[Tuple[str, str, int]]) -> None:
    execute_values_upsert(conn, schema, "qrels", ["query_id", "doc_id", "relevance"], it,
                          conflict_cols=["query_id", "doc_id"], update_cols=["relevance"])


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Load datasets
    ds = ir_datasets.load(DATASET_ID)     # queries + qrels (judged)
    doc_ds = ir_datasets.load(DOC_DATASET_ID)  # passages with doc_store

    # 2) Balanced sample of qrels: 250 per class {0,1,2,3}
    selected_qrels = balanced_qrels(ds, PER_CLASS, RELEVANCE_CLASSES)
    assert len(selected_qrels) == PER_CLASS * len(RELEVANCE_CLASSES)  # 1000

    # 3) Collect referenced IDs
    selected_qids: Set[str] = set()
    selected_docids_ordered: List[str] = []
    for qid, did, rel in selected_qrels:
        selected_qids.add(qid)
        selected_docids_ordered.append(did)

    # Dedup doc_ids preserving order
    seen: Set[str] = set()
    unique_docids: List[str] = []
    for did in selected_docids_ordered:
        if did not in seen:
            seen.add(did)
            unique_docids.append(did)

    # 4) Build query map and slice to selected qids
    qmap = build_query_map(ds)
    missing_q = [qid for qid in selected_qids if qid not in qmap]
    if missing_q:
        raise RuntimeError(f"Missing queries for qids: {missing_q[:10]} (+{max(0, len(missing_q)-10)} more)")

    selected_queries = ((qid, qmap[qid]) for qid in sorted(selected_qids))

    # 5) Fetch passages from doc_store and verify all exist
    doc_rows, found_ids = fetch_passages_for_ids_list(doc_ds, unique_docids)
    missing_docs = [d for d in unique_docids if d not in found_ids]
    if missing_docs:
        raise RuntimeError(f"Missing {len(missing_docs)} passages in doc_store; first few: {missing_docs[:10]}")

    # 6) Load into Postgres (order: queries → docs → qrels)
    pg = Pg()
    conn = connect(pg)
    ensure_schema_and_tables(conn, pg.schema)

    print(f"Loading {len(selected_qids)} queries…")
    load_queries(conn, pg.schema, selected_queries)
    print("✓ Queries loaded.")

    print(f"Loading {len(doc_rows)} docs…")
    load_docs(conn, pg.schema, iter(doc_rows))
    print("✓ Docs loaded.")

    print("Loading 1,000 balanced qrels (250 per relevance 0/1/2/3)…")
    load_qrels(conn, pg.schema, iter(selected_qrels))
    print("✓ Qrels loaded.")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
