from __future__ import annotations

import os
import io
import csv
import tarfile
import gzip
import shutil
import pathlib
import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Tuple, List, Set, Dict

import requests
import psycopg2
import psycopg2.extras


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

# Data sources (official)
QUERIES_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz"
QRELS_PASS_URL = "https://trec.nist.gov/data/deep/2019qrels-pass.txt"
COLLECTION_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"  # contains collection.tsv

# Local cache
BASE = pathlib.Path(__file__).resolve().parent
CACHE = BASE / ".msmarco_cache"
CACHE.mkdir(exist_ok=True)

QUERIES_GZ = CACHE / "msmarco-test2019-queries.tsv.gz"
QRELS_TXT = CACHE / "2019qrels-pass.txt"
COLLECTION_TAR = CACHE / "collection.tar.gz"

# Balanced sampling
PER_CLASS = 250
RELEVANCE_CLASSES = (0, 1, 2, 3)
RANDOM_SEED = 42  # deterministic sampling


# ──────────────────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────────────────

def download(url: str, dest: pathlib.Path) -> None:
    """Stream download to a temp file then atomic rename (project-local, avoids %TEMP%)."""
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


# ──────────────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────────────

def parse_queries_tsv_gz(path: pathlib.Path) -> Dict[str, str]:
    """Return {query_id: text} from gzipped TSV with columns: qid<TAB>query."""
    out: Dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                qid, text = row[0].strip(), row[1]
                if qid and qid.isdigit():
                    out[qid] = text
    return out


def parse_qrels_file(path: pathlib.Path) -> List[Tuple[str, str, int]]:
    """
    TREC qrels format (passage): qid 0 docid relevance
    Returns list of (query_id, doc_id, relevance).
    """
    items: List[Tuple[str, str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split()
            if len(p) < 4:
                continue
            qid, docid, rel = p[0], p[2], p[3]
            if qid.isdigit() and docid and rel.lstrip("-").isdigit():
                items.append((qid, docid, int(rel)))
    return items


def balanced_sample_qrels(all_qrels: List[Tuple[str, str, int]],
                          per_class: int,
                          classes: Iterable[int]) -> List[Tuple[str, str, int]]:
    buckets: Dict[int, List[Tuple[str, str, int]]] = {c: [] for c in classes}
    for qid, did, rel in all_qrels:
        if rel in buckets:
            buckets[rel].append((qid, did, rel))

    shortages = {rel: len(lst) for rel, lst in buckets.items() if len(lst) < per_class}
    if shortages:
        details = ", ".join(f"rel={rel}: have {cnt}, need {per_class}" for rel, cnt in shortages.items())
        raise RuntimeError(f"Not enough judged qrels for balanced sampling: {details}")

    rng = random.Random(RANDOM_SEED)
    out: List[Tuple[str, str, int]] = []
    for rel, lst in buckets.items():
        rng.shuffle(lst)
        out.extend(lst[:per_class])

    out.sort(key=lambda t: (t[2], t[0], t[1]))
    return out


def iter_collection_tsv_from_tar_gz(tar_path: pathlib.Path) -> Iterator[Tuple[str, str]]:
    """
    Yield (doc_id, text) rows from collection.tsv contained in collection.tar.gz.
    Streams without extracting to disk.
    """
    # Open streaming; there is usually a single member named 'collection.tsv'
    with tarfile.open(tar_path, mode="r:gz") as tf:
        member = None
        for m in tf.getmembers():
            if m.name.endswith("collection.tsv"):
                member = m
                break
        if member is None:
            # fallback: pick first *.tsv inside
            for m in tf.getmembers():
                if m.name.lower().endswith(".tsv"):
                    member = m
                    break
        if member is None:
            raise RuntimeError("Could not find collection.tsv inside collection.tar.gz")

        fobj = tf.extractfile(member)
        if fobj is None:
            raise RuntimeError("Failed to open collection.tsv from tar")

        # Each line: doc_id<TAB>text
        with io.TextIOWrapper(fobj, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                doc_id = row[0].strip()
                text = row[1] if len(row) > 1 else ""
                if doc_id:
                    yield (doc_id, text)


def collect_docs_subset_from_collection(tar_path: pathlib.Path,
                                        needed_ids: Set[str]) -> List[Tuple[str, str]]:
    """
    Scan collection.tsv in the tarball and return [(doc_id, text)] for needed_ids only.
    Stops early once all are found.
    """
    found: Dict[str, str] = {}
    total_needed = len(needed_ids)
    for doc_id, text in iter_collection_tsv_from_tar_gz(tar_path):
        if doc_id in needed_ids and doc_id not in found:
            found[doc_id] = text
            if len(found) == total_needed:
                break

    missing = needed_ids - set(found.keys())
    if missing:
        raise RuntimeError(f"Missing {len(missing)} documents from collection; first few: {sorted(list(missing))[:10]}")
    # Keep insertion order stable matching the first-seen in needed_ids
    return [(did, found[did]) for did in needed_ids]


# ──────────────────────────────────────────────────────────────────────────────
# DB utils (match bt/db.py)
# ──────────────────────────────────────────────────────────────────────────────

def connect(pg: Pg):
    dsn = f"host={pg.host} port={pg.port} dbname={pg.dbname} user={pg.user} password={pg.password}"
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    return conn


def ensure_schema_and_tables(conn, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')

        # queries(query_id TEXT, text TEXT)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS "{schema}".queries (
                query_id TEXT PRIMARY KEY,
                text     TEXT NOT NULL
            );
        """)

        # docs(doc_id TEXT, text TEXT)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS "{schema}".docs (
                doc_id TEXT PRIMARY KEY,
                text   TEXT NOT NULL
            );
        """)

        # qrels(query_id TEXT, doc_id TEXT, relevance INT)
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
    # 0) Download sources into project-local cache
    print("Downloading queries and qrels…")
    download(QUERIES_URL, QUERIES_GZ)
    download(QRELS_PASS_URL, QRELS_TXT)
    if not COLLECTION_TAR.exists():
        print("Downloading MS MARCO passage collection (large, one-time)…")
        download(COLLECTION_URL, COLLECTION_TAR)

    # 1) Parse queries & qrels
    qmap = parse_queries_tsv_gz(QUERIES_GZ)
    all_qrels = parse_qrels_file(QRELS_TXT)

    # 2) Balanced selection: 250 per relevance class
    selected_qrels = balanced_sample_qrels(all_qrels, PER_CLASS, RELEVANCE_CLASSES)
    assert len(selected_qrels) == PER_CLASS * len(RELEVANCE_CLASSES)

    # 3) Collect IDs
    selected_qids: Set[str] = {qid for qid, _, _ in selected_qrels}
    docids_in_order: List[str] = [did for _, did, _ in selected_qrels]
    needed_docids: Set[str] = set(docids_in_order)

    # Verify queries available
    missing_q = [qid for qid in selected_qids if qid not in qmap]
    if missing_q:
        raise RuntimeError(f"Missing queries for qids: {missing_q[:10]} (+{max(0, len(missing_q)-10)} more)")

    selected_queries = ((qid, qmap[qid]) for qid in sorted(selected_qids))

    # 4) Collect document texts from collection.tar.gz (only the needed ones)
    print(f"Scanning collection for {len(needed_docids)} unique passages…")
    doc_rows = collect_docs_subset_from_collection(COLLECTION_TAR, needed_docids)
    print("✓ Passages collected.")

    # 5) Load into Postgres in FK-safe order
    pg = Pg()
    conn = connect(pg)
    ensure_schema_and_tables(conn, pg.schema)

    print(f"Loading {len(selected_qids)} queries…")
    load_queries(conn, pg.schema, selected_queries)
    print("✓ Queries loaded.")

    print(f"Loading {len(doc_rows)} docs…")
    load_docs(conn, pg.schema, doc_rows)
    print("✓ Docs loaded.")

    print("Loading 1,000 balanced qrels (250 per relevance 0/1/2/3)…")
    load_qrels(conn, pg.schema, selected_qrels)
    print("✓ Qrels loaded.")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
