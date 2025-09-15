"""
This script extracts a minimal subset of MS MARCO Document v2 TREC-DL (2019–2022) and loads it into PostgreSQL.
Only queries that appear in the qrels and only documents referenced by those qrels are included. 
Unjudged queries/documents are omitted.

Note 1 : Accessing docs may trigger large downloads if the MS MARCO Document v2 corpus is not already cached.
Running this script for the first time took about 5 hours on my laptop (35 GB download).
Note 2 : Python 3.11 needed. Using ir_datasets with Python 3.13 did not work. 

- Creates tables and inserts in dependency order: datasets → queries → docs → qrels.
- Queries are de-duplicated across years by query_id (first occurrence wins).
- Documents are fetched by id from the shared docstore; only qrels-referenced docs are inserted.
"""

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple, List, Dict, Set
import psycopg2
import psycopg2.extras
import ir_datasets

# ---------------- Config ----------------
@dataclass(frozen=True)
class Pg:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "bachelor-thesis"
    user: str = "postgres"
    password: str = "123"
    schema: str = "public"

PG = Pg()

DATASETS: List[Tuple[str, str]] = [
    ("trec-dl-2019", "msmarco-document-v2/trec-dl-2019/judged"),
    ("trec-dl-2020", "msmarco-document-v2/trec-dl-2020/judged"),
    ("trec-dl-2021", "msmarco-document-v2/trec-dl-2021/judged"),
    ("trec-dl-2022", "msmarco-document-v2/trec-dl-2022/judged"),
]

BATCH = 2000  # small, readable batches
# ---------------------------------------


# =============== DB helpers ===============
def connect():
    dsn = f"host={PG.host} port={PG.port} dbname={PG.dbname} user={PG.user} password={PG.password}"
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    return conn

def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {PG.schema}.datasets (
          dataset_id  text PRIMARY KEY,
          dataset_key text NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS {PG.schema}.queries (
          query_id   text PRIMARY KEY,
          "text"     text NOT NULL,
          dataset_id text NOT NULL REFERENCES {PG.schema}.datasets(dataset_id)
            ON UPDATE CASCADE ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS {PG.schema}.docs (
          doc_id text PRIMARY KEY,
          url    text,
          title  text,
          body   text
        );
        CREATE TABLE IF NOT EXISTS {PG.schema}.qrels (
          query_id  text NOT NULL REFERENCES {PG.schema}.queries(query_id)
            ON UPDATE CASCADE ON DELETE CASCADE,
          doc_id    text NOT NULL REFERENCES {PG.schema}.docs(doc_id)
            ON UPDATE CASCADE ON DELETE CASCADE,
          relevance smallint NOT NULL,
          iteration text,
          PRIMARY KEY (query_id, doc_id)
        );
        """)
    conn.commit()

def exec_batch(conn, sql: str, rows: Iterable[Tuple], page: int = BATCH) -> None:
    """
    Simple, readable chunked inserts. No ON CONFLICT. Let errors raise.
    """
    buf: List[Tuple] = []
    with conn.cursor() as cur:
        for r in rows:
            buf.append(r)
            if len(buf) >= page:
                psycopg2.extras.execute_batch(cur, sql, buf, page_size=len(buf))
                conn.commit()
                buf.clear()
        if buf:
            psycopg2.extras.execute_batch(cur, sql, buf, page_size=len(buf))
            conn.commit()


# =============== Dataset I/O ===============
def load_ir_dataset(ds_key: str):
    return ir_datasets.load(ds_key)

def iter_datasets() -> Iterator[Tuple[str, str]]:
    for ds_id, ds_key in DATASETS:
        yield ds_id, ds_key

def iter_queries(ds) -> Iterator[Tuple[str, str]]:
    for q in ds.queries_iter():
        qid = _nt(getattr(q, "query_id", ""))
        text = _nt(getattr(q, "text", ""))
        if qid:
            yield (qid, text)

def iter_qrels(ds) -> Iterator[Tuple[str, str, int, str]]:
    for qr in ds.qrels_iter():
        qid = _nt(getattr(qr, "query_id", ""))
        did = _nt(getattr(qr, "doc_id", ""))
        rel = int(getattr(qr, "relevance", 0) or 0)
        it  = _nt(getattr(qr, "iteration", ""))
        if qid and did:
            yield (qid, did, rel, it)

def docs_store(ds):
    getter = getattr(ds, "docs_store", None)
    store = getter() if callable(getter) else None
    if store is None:
        raise RuntimeError("Dataset has no docstore; cannot random-access by doc_id.")
    return store

def iter_docs_by_ids(store, doc_ids: Iterable[str]) -> Iterator[Tuple[str, str, str, str]]:
    for d in store.get_many_iter(doc_ids):
        yield (
            _nt(getattr(d, "doc_id", "")),
            _nt(getattr(d, "url", "")),
            _nt(getattr(d, "title", "")),
            _nt(getattr(d, "body", "")),
        )

def _nt(v) -> str:
    """Normalize to str (decode bytes), then strip NULs that Postgres can't store."""
    if v is None:
        s = ""
    elif isinstance(v, bytes):
        s = v.decode("utf-8", "ignore")
    else:
        s = str(v)
    return s.replace("\x00", "")


# =============== Loaders (strict) ===============
def insert_datasets(conn) -> None:
    sql = f'INSERT INTO {PG.schema}.datasets (dataset_id, dataset_key) VALUES (%s, %s);'
    exec_batch(conn, sql, iter_datasets())

def insert_queries(conn) -> Set[str]:
    """
    Insert unique queries across all judged splits.
    Returns the set of query_ids inserted.
    """
    sql = f'INSERT INTO {PG.schema}.queries (query_id, "text", dataset_id) VALUES (%s, %s, %s);'
    seen: Set[str] = set()
    rows: List[Tuple[str, str, str]] = []

    for ds_id, ds_key in DATASETS:
        ds = load_ir_dataset(ds_key)
        for qid, text in iter_queries(ds):
            if qid not in seen:
                seen.add(qid)
                rows.append((qid, text, ds_id))
                if len(rows) >= BATCH:
                    exec_batch(conn, sql, rows); rows.clear()
    if rows:
        exec_batch(conn, sql, rows)
    return seen

def collect_qrels_and_doc_ids() -> Tuple[List[Tuple[str, str, int, str]], List[str]]:
    """
    Iterate all judged splits once; collect all qrels rows and the unique doc_ids they reference.
    """
    qrels_rows: List[Tuple[str, str, int, str]] = []
    doc_ids: Dict[str, None] = {}
    for _dsid, ds_key in DATASETS:
        ds = load_ir_dataset(ds_key)
        for qid, did, rel, it in iter_qrels(ds):
            qrels_rows.append((qid, did, rel, it))
            doc_ids[did] = None
    return qrels_rows, list(doc_ids.keys())

def insert_docs(conn, doc_ids: List[str]) -> None:
    """
    Fetch all referenced docs from the shared docstore and insert.
    """
    # Any judged split has the same doc corpus; use the first.
    _, first_key = DATASETS[0]
    store = docs_store(load_ir_dataset(first_key))
    sql = f'INSERT INTO {PG.schema}.docs (doc_id, url, title, body) VALUES (%s, %s, %s, %s);'

    batch: List[Tuple[str, str, str, str]] = []
    for row in iter_docs_by_ids(store, doc_ids):
        batch.append(row)
        if len(batch) >= BATCH:
            exec_batch(conn, sql, batch); batch.clear()
    if batch:
        exec_batch(conn, sql, batch)

def insert_qrels(conn, qrels_rows: List[Tuple[str, str, int, str]]) -> None:
    """
    Insert qrels last. FKs will be checked strictly and FAIL if a query_id/doc_id is missing.
    """
    sql = f'INSERT INTO {PG.schema}.qrels (query_id, doc_id, relevance, iteration) VALUES (%s, %s, %s, %s);'
    exec_batch(conn, sql, qrels_rows, page=BATCH)


# =============== Orchestration ===============
def main():
    conn = connect()
    try:
        ensure_schema(conn)

        print("1) insert datasets …")
        insert_datasets(conn)

        print("2) insert queries …")
        seen_queries = insert_queries(conn)
        print(f"   unique queries: {len(seen_queries):,}")

        print("3) collect qrels + doc_ids …")
        qrels_rows, doc_ids = collect_qrels_and_doc_ids()
        print(f"   qrels rows: {len(qrels_rows):,}")
        print(f"   unique doc_ids: {len(doc_ids):,}")

        print("4) insert docs … (this may take a while)")
        insert_docs(conn, doc_ids)

        print("5) insert qrels …")
        insert_qrels(conn, qrels_rows)

        # Analyze (optional but nice)
        with conn.cursor() as cur:
            for t in ("datasets","queries","docs","qrels"):
                cur.execute(f"ANALYZE {PG.schema}.{t};")
        conn.commit()
        print(" Done.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
