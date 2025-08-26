"""
Note: I was unable to load the generated csv into postgres. That's why I abondened this script and loaded directly into postgres instead.

Export a subset of the MS MARCO Document v2 TREC-DL (2019–2022) judged data to CSVs.
The subset contains the qrels and its related queries and docs.

This script loads the following datasets via `ir_datasets`:
  - msmarco-document-v2/trec-dl-2019/judged
  - msmarco-document-v2/trec-dl-2020/judged
  - msmarco-document-v2/trec-dl-2021/judged
  - msmarco-document-v2/trec-dl-2022/judged

It writes four CSV files under OUT_DIR:
  1) datasets.csv 
  2) queries.csv  
  3) qrels.csv    
  4) docs.csv 

Documents are fetched via the dataset's docstore (random-access by doc_id), in batches, with NO truncation.
Note 1 : Accessing docs may trigger large downloads if the MS MARCO Document v2 corpus is not already cached.
Running this script for the first time took about 5 hours on my laptop (35 GB download).
Note 2 : Python 3.11 needed. Using ir_datasets with Python 3.13 did not work. 
"""

import os
import csv
from collections import OrderedDict

import ir_datasets

# ---- CONFIG ----
DATASETS = [
    ("trec-dl-2019", "msmarco-document-v2/trec-dl-2019/judged"),
    ("trec-dl-2020", "msmarco-document-v2/trec-dl-2020/judged"),
    ("trec-dl-2021", "msmarco-document-v2/trec-dl-2021/judged"),
    ("trec-dl-2022", "msmarco-document-v2/trec-dl-2022/judged"),
]
OUT_DIR = "docv2_trec_dl_csv_subset"
DOC_BATCH_SIZE = 10_000
# ---------------

def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def open_writer(path, header):
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    return f, w

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

    p_datasets = os.path.join(OUT_DIR, "datasets.csv")
    p_queries  = os.path.join(OUT_DIR, "queries.csv")
    p_qrels    = os.path.join(OUT_DIR, "qrels.csv")
    p_docs     = os.path.join(OUT_DIR, "docs.csv")

    # Fixed headers
    H_DATASETS = ["dataset_id", "dataset_key"]
    H_QUERIES  = ["query_id", "text", "dataset_id"]
    H_QRELS    = ["query_id", "doc_id", "relevance", "iteration"]
    H_DOCS     = ["doc_id", "url", "title", "body"]

    f_ds, w_ds = open_writer(p_datasets, H_DATASETS)
    f_q,  w_q  = open_writer(p_queries,  H_QUERIES)
    f_r,  w_r  = open_writer(p_qrels,    H_QRELS)

    # Track seen queries (so each query appears once) and docs to fetch
    seen_queries = set()
    needed_doc_ids = OrderedDict()
    ds_for_store = None

    print("Writing datasets, queries (with dataset_id), qrels; collecting doc_ids…")
    for i, (ds_id, ds_key) in enumerate(DATASETS, start=1):
        print(f"[{i}/{len(DATASETS)}] {ds_key}")
        w_ds.writerow({"dataset_id": ds_id, "dataset_key": ds_key})

        ds = ir_datasets.load(ds_key)
        if ds_for_store is None:
            ds_for_store = ds  # all these share the same doc-v2 corpus

        # QUERIES: GenericQuery(query_id, text) + dataset_id (1-to-N)
        for q in ds.queries_iter():
            qid = nt_get(q, "query_id")
            if qid not in seen_queries:
                w_q.writerow({"query_id": qid, "text": nt_get(q, "text"), "dataset_id": ds_id})
                seen_queries.add(qid)
            else:
                # if a duplicate ever appears
                pass

        # QRELS: TrecQrel(query_id, doc_id, relevance, iteration) — keep ALL labels incl. 0
        for qr in ds.qrels_iter():
            row = {
                "query_id":  nt_get(qr, "query_id"),
                "doc_id":    nt_get(qr, "doc_id"),
                "relevance": int(getattr(qr, "relevance", 0)),
                "iteration": nt_get(qr, "iteration"),
            }
            w_r.writerow(row)
            needed_doc_ids[row["doc_id"]] = None

    f_ds.close()
    f_q.close()
    f_r.close()

    print(f"Unique doc_ids to fetch for docs.csv: {len(needed_doc_ids):,}")

    # ---- DOCS via docstore (MsMarcoDocument: doc_id, url, title, body) ----
    if ds_for_store is None:
        raise RuntimeError("No datasets loaded; nothing to export.")

    getter = getattr(ds_for_store, "docs_store", None)
    store = getter() if callable(getter) else None
    if store is None:
        raise RuntimeError("Dataset has no docstore; cannot random-access by doc_id.")

    f_d, w_d = open_writer(p_docs, H_DOCS)
    print("Fetching & writing docs (this can take a while)…")
    written = 0
    total = len(needed_doc_ids)
    for bi, chunk in enumerate(batched(needed_doc_ids.keys(), DOC_BATCH_SIZE), start=1):
        wrote = 0
        for d in store.get_many_iter(chunk):
            w_d.writerow({
                "doc_id": nt_get(d, "doc_id"),
                "url":    nt_get(d, "url"),
                "title":  nt_get(d, "title"),
                "body":   nt_get(d, "body"),
            })
            wrote += 1
        written += wrote
        print(f"  batch {bi}: wrote {wrote} docs | total {written}/{total}")
    f_d.close()

    print("\nAll done. CSVs written:")
    print("  ", p_datasets)
    print("  ", p_queries)
    print("  ", p_qrels)
    print("  ", p_docs)

if __name__ == "__main__":
    main()
