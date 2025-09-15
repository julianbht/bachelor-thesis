PROMPT_TMPL = """You are a relevance judge for document retrieval.
Rate how relevant the DOCUMENT is to the user QUERY on a 0–3 scale:

0 = Irrelevant: The passage has nothing to do with the query.
1 = Related: The passage seems related to the query but does not answer it.
2 = Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.
3 = Perfectly relevant: The passage is dedicated to the query and contains the exact answer.

Return strict JSON ONLY with key: score (0,1,2,3).

QUERY:
{query}

DOCUMENT (passage text):
{text}
"""


def build_prompt(query: str, text: str, template: str = PROMPT_TMPL) -> str:
    return template.format(query=query, text=text)
