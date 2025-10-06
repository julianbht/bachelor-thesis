"""
Load a simple, balanced 1,000-qrel subset of MS MARCO Passage v2 (TREC-DL 2021/2022, judged)
into PostgreSQL: datasets, queries, docs, qrels.

Balance: 250 qrels for each relevance label 0, 1, 2, 3.

This is intentionally simple:
- Inserts one row at a time.
- No batching, no ON CONFLICT, no retries.
- Raises on problems (e.g., not enough qrels per label, missing docs).
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, DefaultDict
from collections import defaultdict
import random
import psycopg2
import ir_datasets

# ---------------- Config ----------------
@dataclass(frozen=True)
class Pg:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "bachelor-thesis"
    user: str = "postgres"
    password: str = "123"
    schema: str = "passagev2"

PG = Pg()

DATASETS: List[Tuple[str, str]] = [
    ("trec-dl-2021", "msmarco-passage-v2/trec-dl-2021/judged"),
    ("trec-dl-2022", "msmarco-passage-v2/trec-dl-2022/judged"),
]

LABELS = [0, 1, 2, 3]
TARGET_PER_LABEL = 250
TOTAL_TARGET = TARGET_PER_LABEL * len(LABELS)  # 1000
RANDOM_SEED = 42
# ---------------------------------------


def _nt(v) -> str:
    """Normalize to str and strip NULs (Postgres can't store them)."""
    if v is None:
        s = ""
    elif isinstance(v, bytes):
        s = v.decode("utf-8", "ignore")
    else:
        s = str(v)
    return s.replace("\x00", "")


def ensure_schema(conn) -> None:
    cur = conn.cursor()
    print("Creating tables if not exist …")
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
      text   text
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
    cur.close()


def main():
    random.seed(RANDOM_SEED)

    print("Connecting to Postgres …")
    dsn = f"host={PG.host} port={PG.port} dbname={PG.dbname} user={PG.user} password={PG.password}"
    conn = psycopg2.connect(dsn)
    conn.autocommit = False

    try:
        ensure_schema(conn)

        print("Loading judged splits via ir_datasets …")
        qrels_by_label: DefaultDict[int, List[Tuple[str,str,int,str]]] = defaultdict(list)
        queries_all: Dict[str, Tuple[str, str]] = {}  # qid -> (text, dataset_id)

        for dataset_id, ds_key in DATASETS:
            print(f"  Reading {ds_key} …")
            ds = ir_datasets.load(ds_key)

            # collect queries (first occurrence wins)
            for q in ds.queries_iter():
                qid = _nt(getattr(q, "query_id", "") or getattr(q, "qid", ""))
                text = _nt(getattr(q, "text", ""))
                if qid and qid not in queries_all:
                    queries_all[qid] = (text, dataset_id)

            # collect qrels by label
            for qr in ds.qrels_iter():
                qid = _nt(getattr(qr, "query_id", ""))
                did = _nt(getattr(qr, "doc_id", ""))
                rel = int(getattr(qr, "relevance", 0) or 0)
                it  = _nt(getattr(qr, "iteration", ""))
                if qid and did and rel in LABELS:
                    qrels_by_label[rel].append((qid, did, rel, it))

        print("Counts per label (available):")
        for lbl in LABELS:
            print(f"  label {lbl}: {len(qrels_by_label[lbl])} qrels")

        # sample 250 per label (deterministic)
        print(f"Sampling {TARGET_PER_LABEL} per label (total {TOTAL_TARGET}) …")
        selected_qrels: List[Tuple[str,str,int,str]] = []
        for lbl in LABELS:
            pool = qrels_by_label[lbl]
            picked = random.sample(pool, TARGET_PER_LABEL)
            selected_qrels.extend(picked)
        random.shuffle(selected_qrels)

        used_qids: Set[str] = {qid for (qid, *_rest) in selected_qrels}
        used_dids: Set[str] = {did for (_qid, did, *_rest) in selected_qrels}
        print(f"Selected qrels: {len(selected_qrels)}")
        print(f"  distinct queries: {len(used_qids)}")
        print(f"  distinct docs:    {len(used_dids)}")

        # fetch docs from msmarco-passage-v2 corpus
        print("Fetching passage texts for selected docs … (this can take time on first run)")
        corpus = ir_datasets.load("msmarco-passage-v2")
        store_getter = getattr(corpus, "docs_store", None)
        if not callable(store_getter):
            raise RuntimeError("msmarco-passage-v2 has no docs_store()")
        store = store_getter()

        fetched_docs: Dict[str, str] = {}
        for d in store.get_many_iter(used_dids):
            did = _nt(getattr(d, "doc_id", ""))
            text = _nt(getattr(d, "text", ""))
            if did:
                fetched_docs[did] = text

        if len(fetched_docs) != len(used_dids):
            missing = len(used_dids) - len(fetched_docs)
            raise RuntimeError(f"Missing {missing} docs from corpus fetch; aborting.")

        # insert everything (1-by-1)
        cur = conn.cursor()

        print("Inserting datasets …")
        for dataset_id, ds_key in DATASETS:
            cur.execute(
                f'INSERT INTO {PG.schema}.datasets (dataset_id, dataset_key) VALUES (%s, %s);',
                (dataset_id, ds_key)
            )

        print(f"Inserting {len(used_qids)} queries …")
        for qid in used_qids:
            if qid not in queries_all:
                raise RuntimeError(f"Selected query {qid} not found in queries_all")
            qtext, dsid = queries_all[qid]
            cur.execute(
                f'INSERT INTO {PG.schema}.queries (query_id, "text", dataset_id) VALUES (%s, %s, %s);',
                (qid, qtext, dsid)
            )

        print(f"Inserting {len(used_dids)} docs …")
        for did in used_dids:
            cur.execute(
                f'INSERT INTO {PG.schema}.docs (doc_id, text) VALUES (%s, %s);',
                (did, fetched_docs[did])
            )

        print(f"Inserting {len(selected_qrels)} qrels …")
        for (qid, did, rel, it) in selected_qrels:
            cur.execute(
                f'INSERT INTO {PG.schema}.qrels (query_id, doc_id, relevance, iteration) VALUES (%s, %s, %s, %s);',
                (qid, did, rel, it)
            )

        conn.commit()
        cur.close()

        # Summary
        print("Done.")
        print("Summary:")
        print("  datasets inserted: 2")
        print(f"  queries inserted:  {len(used_qids)}")
        print(f"  docs inserted:     {len(used_dids)}")
        print(f"  qrels inserted:    {len(selected_qrels)}")
        by_lbl = defaultdict(int)
        for _qid, _did, r, _it in selected_qrels:
            by_lbl[r] += 1
        print("  qrels by label:", ", ".join(f"{l}={by_lbl.get(l,0)}" for l in LABELS))

    finally:
        conn.close()


if __name__ == "__main__":
    main()
