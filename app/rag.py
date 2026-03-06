from typing import List, Dict

from app.llm_client import ask_llm
from app.retriever import retrieve_top_k


RAG_SYSTEM_PROMPT = """
You are Aurora Beauty Studio's customer support assistant.

Use ONLY the provided context to answer.
If the context does not contain the answer, say you don't know and suggest contacting the studio.

Be concise, friendly, and clear.
"""


def build_context(chunks: List[Dict]) -> str:
    """
    Convert retrieved chunks into a single context string for the LLM.
    """
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c['source_file']} | distance={c['distance']:.3f}]\n{c['text']}")
    return "\n\n".join(parts)


def answer_with_rag(user_question: str, k: int = 3) -> str:
    """
    Retrieve top-k relevant chunks and ask the LLM to answer using that context.
    """
    chunks = retrieve_top_k(user_question, k=k)
    context = build_context(chunks)

    prompt = f"""
{RAG_SYSTEM_PROMPT}

CONTEXT:
{context}

USER QUESTION:
{user_question}

Return a helpful answer.
"""

    return ask_llm(prompt)


if __name__ == "__main__":
    q = "Do you offer gel manicure and how much does it cost?"
    print(answer_with_rag(q))