"""
Export a subset of the MS MARCO Document v2 TREC-DL (2019–2022) judged data to Parquet.
The subset contains the qrels and their related queries and docs.

This script loads the following datasets via `ir_datasets`:
  - msmarco-document-v2/trec-dl-2019/judged
  - msmarco-document-v2/trec-dl-2020/judged
  - msmarco-document-v2/trec-dl-2021/judged
  - msmarco-document-v2/trec-dl-2022/judged

It writes four Parquet files under OUT_DIR:
  1) datasets.parquet  (dataset_id, dataset_key)
  2) queries.parquet   (query_id, text, dataset_id)
  3) qrels.parquet     (query_id, doc_id, relevance, iteration)
  4) docs.parquet      (doc_id, url, title, body)

Documents are fetched via the dataset's docstore (random-access by doc_id), in batches, with NO truncation.
Note 1 : Accessing docs may trigger large downloads if the MS MARCO Document v2 corpus is not already cached.
         First runs can take many hours and ~35 GB download.
Note 2 : Python 3.11 recommended. Using ir_datasets with Python 3.13 does not work.
"""

import os
from collections import OrderedDict

import ir_datasets

import pyarrow as pa
import pyarrow.parquet as pq

# ---- CONFIG ----
DATASETS = [
    ("trec-dl-2019", "msmarco-document-v2/trec-dl-2019/judged"),
    ("trec-dl-2020", "msmarco-document-v2/trec-dl-2020/judged"),
    ("trec-dl-2021", "msmarco-document-v2/trec-dl-2021/judged"),
    ("trec-dl-2022", "msmarco-document-v2/trec-dl-2022/judged"),
]
OUT_DIR = "docv2_trec_dl_parquet_subset"
DOC_BATCH_SIZE = 10_000
PARQUET_COMPRESSION = "zstd"  # good tradeoff (snappy is fine too)
# ---------------

def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def nt_get(obj, name, default=""):
    v = getattr(obj, name, default)
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="ignore")
    return v

def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    ensure_outdir()

    p_datasets = os.path.join(OUT_DIR, "datasets.parquet")
    p_queries  = os.path.join(OUT_DIR, "queries.parquet")
    p_qrels    = os.path.join(OUT_DIR, "qrels.parquet")
    p_docs     = os.path.join(OUT_DIR, "docs.parquet")

    # Fixed schemas (exact shapes you specified)
    sc_datasets = pa.schema([
        pa.field("dataset_id", pa.string()),
        pa.field("dataset_key", pa.string()),
    ])
    sc_queries = pa.schema([
        pa.field("query_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("dataset_id", pa.string()),
    ])
    sc_qrels = pa.schema([
        pa.field("query_id", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("relevance", pa.int32()),
        pa.field("iteration", pa.string()),
    ])
    sc_docs = pa.schema([
        pa.field("doc_id", pa.string()),
        pa.field("url", pa.string()),
        pa.field("title", pa.string()),
        pa.field("body", pa.string()),
    ])

    datasets_rows = []
    queries_rows  = []
    qrels_rows    = []

    # Track seen queries (so each query appears once) and docs to fetch
    seen_queries = set()
    needed_doc_ids = OrderedDict()
    ds_for_store = None

    print("Collecting datasets, queries (with dataset_id), qrels; recording doc_ids…")
    for i, (ds_id, ds_key) in enumerate(DATASETS, start=1):
        print(f"[{i}/{len(DATASETS)}] {ds_key}")
        datasets_rows.append({"dataset_id": ds_id, "dataset_key": ds_key})

        ds = ir_datasets.load(ds_key)
        if ds_for_store is None:
            ds_for_store = ds  # all these share the same doc-v2 corpus

        # QUERIES: GenericQuery(query_id, text) + dataset_id (1-to-N)
        for q in ds.queries_iter():
            qid = nt_get(q, "query_id")
            if qid not in seen_queries:
                queries_rows.append({
                    "query_id": qid,
                    "text": nt_get(q, "text"),
                    "dataset_id": ds_id
                })
                seen_queries.add(qid)

        # QRELS: TrecQrel(query_id, doc_id, relevance, iteration) — keep ALL labels incl. 0
        for qr in ds.qrels_iter():
            qrels_rows.append({
                "query_id":  nt_get(qr, "query_id"),
                "doc_id":    nt_get(qr, "doc_id"),
                "relevance": int(getattr(qr, "relevance", 0)),
                "iteration": nt_get(qr, "iteration"),
            })
            needed_doc_ids[qrels_rows[-1]["doc_id"]] = None

    # ---- Write datasets/queries/qrels to Parquet (single write each) ----
    print("Writing datasets.parquet, queries.parquet, qrels.parquet …")
    pq.write_table(pa.Table.from_pylist(datasets_rows, schema=sc_datasets),
                   p_datasets, compression=PARQUET_COMPRESSION)
    pq.write_table(pa.Table.from_pylist(queries_rows, schema=sc_queries),
                   p_queries, compression=PARQUET_COMPRESSION)
    pq.write_table(pa.Table.from_pylist(qrels_rows, schema=sc_qrels),
                   p_qrels, compression=PARQUET_COMPRESSION)

    print(f"Unique doc_ids to fetch for docs.parquet: {len(needed_doc_ids):,}")

    # ---- DOCS via docstore (MsMarcoDocument: doc_id, url, title, body) ----
    if ds_for_store is None:
        raise RuntimeError("No datasets loaded; nothing to export.")

    getter = getattr(ds_for_store, "docs_store", None)
    store = getter() if callable(getter) else None
    if store is None:
        raise RuntimeError("Dataset has no docstore; cannot random-access by doc_id.")

    print("Fetching & writing docs to Parquet in streaming mode (this can take a while)…")
    written = 0
    total = len(needed_doc_ids)

    # Stream docs using a ParquetWriter (append row groups per batch)
    writer = pq.ParquetWriter(p_docs, schema=sc_docs, compression=PARQUET_COMPRESSION)
    try:
        for bi, chunk in enumerate(batched(needed_doc_ids.keys(), DOC_BATCH_SIZE), start=1):
            rows = []
            wrote = 0
            for d in store.get_many_iter(chunk):
                rows.append({
                    "doc_id": nt_get(d, "doc_id"),
                    "url":    nt_get(d, "url"),
                    "title":  nt_get(d, "title"),
                    "body":   nt_get(d, "body"),
                })
                wrote += 1
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=sc_docs))
            written += wrote
            print(f"  batch {bi}: wrote {wrote} docs | total {written}/{total}")
    finally:
        writer.close()

    print("\nAll done. Parquet files:")
    print("  ", p_datasets)
    print("  ", p_queries)
    print("  ", p_qrels)
    print("  ", p_docs)

if __name__ == "__main__":
    main()
