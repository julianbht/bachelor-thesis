from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field

class Query(BaseModel):
    query_id: str = Field(..., description="Unique query ID")
    query_text: str = Field(..., description="The natural language query")

class Document(BaseModel):
    docid: str = Field(..., description="Document ID")
    doc: str = Field(..., description="Document text")

class Relevance(BaseModel):
    query_id: str
    docid: str
    relevance: int = Field(..., description="Relevance label from dataset")

class JudgingExample(BaseModel):
    """Normalized training/eval unit for the ensemble judge.

    This joins a query + document + gold label into one record so downstream
    components (prompt builders, inference adapters) can operate on a single item.
    """

    dataset: Literal["llm-judge-2024"]
    query_id: str
    query_text: str
    docid: str
    doc: str
    gold_relevance: int

    @classmethod
    def from_parts(
        cls, dataset: str, q: Query, d: Document, r: Relevance
    ) -> "JudgingExample":
        assert q.query_id == r.query_id
        assert d.docid == r.docid
        return cls(
            dataset=dataset, query_id=q.query_id, query_text=q.query_text,
            docid=d.docid, doc=d.doc, gold_relevance=r.relevance,
        )
