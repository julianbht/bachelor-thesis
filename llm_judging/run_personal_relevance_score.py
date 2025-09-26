#!/usr/bin/env python3
"""
run_personal_relevance_score.py — label into msmarco.qrels.personal_score

Behavior:
  • Ensures qrels.personal_score exists
  • Pulls only rows where personal_score IS NULL
  • Randomizes question order and passage order within each question
  • Commits after every label
"""

from __future__ import annotations
import os, textwrap, random
from typing import List, Dict
from bt.config import Pg
from bt.db import connect

SCHEMA = "passagev2"
TABLE  = "qrels"
LABELS = ["0", "1", "2", "3"]

def wrap_block(title: str, content: str, width: int = 100) -> str:
    wrapped = "\n".join(textwrap.fill(line, width=width) for line in content.splitlines())
    bar = "-" * min(width, max(len(title) + 4, 20))
    return f"\n{title}\n{bar}\n{wrapped}\n"

def prompt(msg: str) -> str:
    try:
        return input(msg)
    except EOFError:
        return "q"

def fetch_unlabeled(conn) -> List[dict]:
    """Return all qrels where personal_score is NULL, joined with queries/docs."""
    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            qrels.query_id,
            qrels.doc_id,
            q.text AS query_text,
            d.text AS doc_text
        FROM {SCHEMA}.{TABLE} AS qrels
        JOIN {SCHEMA}.queries AS q ON q.query_id = qrels.query_id
        JOIN {SCHEMA}.docs    AS d ON d.doc_id   = qrels.doc_id
        WHERE qrels.personal_score IS NULL
    """)
    cols = [desc[0] for desc in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]

def build_randomized_sequence(rows: List[dict]) -> List[dict]:
    by_qid: Dict[str, List[dict]] = {}
    for r in rows:
        by_qid.setdefault(r["query_id"], []).append(r)
    for qid in by_qid:
        random.shuffle(by_qid[qid])      # shuffle passages within question
    qids = list(by_qid.keys())
    random.shuffle(qids)                  # shuffle question order
    out: List[dict] = []
    for qid in qids:
        out.extend(by_qid[qid])
    return out

def main():
    conn = connect(Pg())
    cur = conn.cursor()

    # Ensure personal_score column exists
    cur.execute(f"""
        ALTER TABLE {SCHEMA}.{TABLE}
        ADD COLUMN IF NOT EXISTS personal_score smallint;
    """)
    conn.commit()

    rows = fetch_unlabeled(conn)
    if not rows:
        print("All qrels already have personal_score. Nothing to do.")
        return

    rows = build_randomized_sequence(rows)
    n = len(rows)

    i = 0
    history: List[int] = []
    while 0 <= i < n:
        row = rows[i]
        qid, did = row["query_id"], row["doc_id"]
        query = (row["query_text"] or "").strip()
        doc   = (row["doc_text"]  or "").strip()

        os.system("cls" if os.name == "nt" else "clear")
        print(f"[{i+1}/{n}] qid={qid}  doc={did}")
        print(wrap_block("QUERY", query))
        print(wrap_block("PASSAGE", doc))

        ans = prompt(f"Label {LABELS} | s(skip) b(back) q(quit): ").strip().lower()

        if ans == "q":
            conn.commit()
            print("Progress saved. Bye!")
            break

        if ans == "b":
            if history:
                i = history.pop()
            continue

        if ans == "s":
            history.append(i); i += 1
            continue

        if ans not in LABELS:
            continue

        pred = int(ans)
        cur.execute(
            f"UPDATE {SCHEMA}.{TABLE} SET personal_score = %s WHERE query_id = %s AND doc_id = %s",
            (pred, qid, did)
        )
        conn.commit()  # commit after every entry
        history.append(i); i += 1

    if i >= n:
        conn.commit()
        print("Finished labeling all remaining qrels.")
    conn.close()

if __name__ == "__main__":
    main()
