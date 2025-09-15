import json
import psycopg2
import psycopg2.extras
import ollama
from dataclasses import dataclass

# ---- Config ----
@dataclass(frozen=True)
class Pg:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "bachelor-thesis"
    user: str = "postgres"
    password: str = "123"
    schema: str = "public"

PG = Pg()

MODEL = "llama3.2:3b"
PROMPT_TMPL = """You are a relevance judge for ad-hoc document retrieval.
Rate how relevant the DOCUMENT is to the user QUERY on a 0–3 scale:

0 = Not relevant (off-topic).
1 = Partially relevant (some overlap; not directly answering).
2 = Relevant (on-topic and useful; may miss specifics).
3 = Highly relevant (direct, comprehensive, and focused on the query).

Return strict JSON ONLY with keys: score (int 0-3), rationale (string <= 2 sentences).

QUERY:
{query}

DOCUMENT (title then body):
{title}
{body}
"""

def connect():
    dsn = f"host={PG.host} port={PG.port} dbname={PG.dbname} user={PG.user} password={PG.password}"
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    return conn

def fetch_one_pair(conn):
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(f"""
        SELECT q.query_id, q.text AS query_text,
               d.doc_id, COALESCE(d.title,'') AS title, COALESCE(d.body,'') AS body,
               qr.relevance AS gold_score
        FROM {PG.schema}.qrels qr
        JOIN {PG.schema}.queries q ON q.query_id = qr.query_id
        JOIN {PG.schema}.docs d ON d.doc_id = qr.doc_id
        ORDER BY random()
        LIMIT 1;
        """)
        row = cur.fetchone()
        return dict(row)

def judge_with_ollama(model, prompt):
    resp = ollama.generate(
        model=model,
        prompt=prompt,
        format="json",   # forces strict JSON output
        options={"temperature": 0}
    )
    txt = resp["response"]
    obj = json.loads(txt)
    return {"score": int(obj["score"]), "rationale": obj["rationale"]}

def main():
    conn = connect()
    try:
        pair = fetch_one_pair(conn)
        prompt = PROMPT_TMPL.format(
            query=pair["query_text"],
            title=pair["title"][:500],
            body=pair["body"][:4000]
        )
        result = judge_with_ollama(MODEL, prompt)
        print({
            "query_id": pair["query_id"],
            "doc_id": pair["doc_id"],
            "gold_score": int(pair["gold"]),
            "llm_score": result["score"],
            "rationale": result["rationale"]
        })
    finally:
        conn.close()

if __name__ == "__main__":
    main()
