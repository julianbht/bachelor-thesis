import io
import json
from pathlib import Path
from adapters.llm_judge import iter_examples

def _write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.write_text(content, encoding="utf-8")
    return p

def test_iter_examples(tmp_path: Path):
    _write(tmp_path, "llm4eval_query_2024.txt", "q1\tWhat is AI?\n")
    _write(tmp_path, "llm4eval_document_2024.jsonl", json.dumps({"docid": "d1", "doc": "AI is..."})+"\n")
    _write(tmp_path, "llm4eval_test_qrel_2024.txt", "q1 1 d1\n")

    items = list(iter_examples(tmp_path))
    assert len(items) == 1
    ex = items[0]
    assert ex.query_id == "q1"
    assert ex.docid == "d1"
    assert ex.gold_relevance == 1