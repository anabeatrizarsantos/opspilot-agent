import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np

from app.embeddings_client import embed_texts


FAISS_INDEX_FILE = Path("data/kb/faiss.index")
METADATA_FILE = Path("data/kb/metadata.json")


def load_index() -> faiss.Index:
    """
    Load FAISS index from disk.
    """
    if not FAISS_INDEX_FILE.exists():
        raise FileNotFoundError("FAISS index not found. Run kb_indexer first.")

    return faiss.read_index(str(FAISS_INDEX_FILE))


def load_metadata() -> List[Dict]:
    """
    Load metadata from disk.
    """
    if not METADATA_FILE.exists():
        raise FileNotFoundError("Metadata file not found. Run kb_indexer first.")

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_top_k(query: str, k: int = 3) -> List[Dict]:
    """
    Given a query, return top-k most similar chunks using FAISS.
    """
    index = load_index()
    metadata = load_metadata()

    # 1) Generate embedding for the query
    query_vector = embed_texts([query])[0]

    # 2) Convert query vector to numpy matrix with shape (1, dimension)
    query_vector_np = np.array([query_vector], dtype="float32")

    # 3) Search nearest vectors in FAISS
    distances, indices = index.search(query_vector_np, k)

    # 4) Build result list
    results: List[Dict] = []

    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        entry = metadata[idx]
        results.append(
            {
                "distance": float(distance),
                "text": entry["text"],
                "source_file": entry["source_file"],
            }
        )

    return results


if __name__ == "__main__":
    results = retrieve_top_k("How much is gel manicure?")
    for r in results:
        print("\ndistance:", r["distance"])
        print("Source:", r["source_file"])
        print("Text:", r["text"])