import json
from pathlib import Path
from typing import List, Dict

from app.embeddings_client import embed_texts
from app.vector_utils import cosine_similarity


INDEX_FILE = Path("data/kb/index.json")


def load_index() -> List[Dict]:
    """
    Load vector index from disk.
    """
    if not INDEX_FILE.exists():
        raise FileNotFoundError("KB index not found. Run kb_indexer first.")

    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_top_k(query: str, k: int = 3) -> List[Dict]:
    """
    Given a query, return top-k most similar chunks.
    """
    index_data = load_index()

    # 1) Generate embedding for the query
    query_vector = embed_texts([query])[0]

    # 2) Compute similarity against all chunks
    scored_chunks = []

    for entry in index_data:
        score = cosine_similarity(query_vector, entry["embedding"])
        scored_chunks.append({
            "score": score,
            "text": entry["text"],
            "source_file": entry["source_file"],
        })

    # 3) Sort by similarity descending
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    # 4) Return top-k
    return scored_chunks[:k]


if __name__ == "__main__":
    results = retrieve_top_k("How much is gel manicure?")
    for r in results:
        print("\nScore:", r["score"])
        print("Source:", r["source_file"])
        print("Text:", r["text"])