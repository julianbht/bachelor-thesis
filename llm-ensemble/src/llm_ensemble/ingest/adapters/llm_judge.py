from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, Tuple

from llm_ensemble.ingest.domain.models import Query, Document, Relevance, JudgingExample    

@dataclass(frozen=True)
class LlmJudgePaths:
    base_dir: Path

    @property
    def queries(self) -> Path:
        return self.base_dir / "llm4eval_query_2024.txt"

    @property
    def documents(self) -> Path:
        return self.base_dir / "llm4eval_document_2024.jsonl"

    @property
    def qrels(self) -> Path:
        return self.base_dir / "llm4eval_test_qrel_2024.txt"


def read_queries(path: Path) -> Dict[str, Query]:
    """Read TSV of (query_id, query_text) into a dict."""
    out: Dict[str, Query] = {}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            # Expect exactly two columns, split once to be robust to tabs in text
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid query line {i}: {line!r}")
            qid, qtext = parts[0].strip(), parts[1].strip()
            out[qid] = Query(query_id=qid, query_text=qtext)
    return out


def read_documents(path: Path) -> Dict[str, Document]:
    out: Dict[str, Document] = {}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {i}: {e}") from e
            docid = obj.get("docid")
            doc = obj.get("doc")
            if not (isinstance(docid, str) and isinstance(doc, str)):
                raise ValueError(f"Missing docid/doc at line {i}")
            out[docid] = Document(docid=docid, doc=doc)
    return out


def read_qrels(path: Path) -> Iterable[Relevance]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Invalid qrel line {i}: {line!r}")
            qid, rel, docid = parts
            try:
                rel_i = int(rel)
            except ValueError:
                raise ValueError(f"Invalid relevance at line {i}: {rel!r}")
            yield Relevance(query_id=qid, docid=docid, relevance=rel_i)


def iter_examples(base_dir: Path) -> Generator[JudgingExample, None, None]:
    paths = LlmJudgePaths(base_dir)
    queries = read_queries(paths.queries)
    docs = read_documents(paths.documents)
    missing_q, missing_d = 0, 0
    for r in read_qrels(paths.qrels):
        q = queries.get(r.query_id)
        d = docs.get(r.docid)
        if q is None:
            missing_q += 1
            continue
        if d is None:
            missing_d += 1
            continue
        yield JudgingExample.from_parts("llm-judge-2024", q, d, r)
    if missing_q or missing_d:
        # Emit a warning summary; the CLI will surface this via logging
        print(
            f"[llm-judge] Skipped {missing_q} qrels with missing query and {missing_d} with missing doc",
            flush=True,
        )