import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from app.embeddings_client import embed_texts


KB_FOLDER = Path("kb")
DATA_FOLDER = Path("data/kb")
FAISS_INDEX_FILE = DATA_FOLDER / "faiss.index"
METADATA_FILE = DATA_FOLDER / "metadata.json"


def chunk_text(text: str, max_words: int = 180, overlap_words: int = 40) -> List[str]:
    """
    Split text into overlapping chunks.

    Why this exists:
    - RAG works better when we retrieve small, focused passages (chunks)
    - overlap helps preserve context across chunk boundaries
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end == len(words):
            break

        start = max(0, end - overlap_words)

    return chunks


def load_kb_files() -> List[Dict[str, str]]:
    """
    Load all .txt files from kb/ folder.
    Returns a list of dicts: {filename, content}.
    """
    files: List[Dict[str, str]] = []

    for file_path in KB_FOLDER.rglob("*.txt"):
        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            rel_path = file_path.relative_to(KB_FOLDER).as_posix()
            files.append({"filename": rel_path, "content": content})

    return files


def build_index() -> None:
    """
    Build a FAISS vector index from kb/*.txt using chunking.

    Output:
    - faiss.index -> stores only embeddings/vectors
    - metadata.json -> stores text and metadata for each chunk
    """
    kb_files = load_kb_files()

    # 1) Create metadata entries (without embeddings)
    metadata: List[Dict] = []

    for f in kb_files:
        chunks = chunk_text(f["content"], max_words=180, overlap_words=40)

        for i, ch in enumerate(chunks):
            metadata.append(
                {
                    "id": len(metadata),
                    "source_file": f["filename"],
                    "chunk_index": i,
                    "text": ch,
                }
            )

    if not metadata:
        raise ValueError("No KB content found. Add .txt files under kb/.")

    # 2) Generate embeddings in batch
    texts = [entry["text"] for entry in metadata]
    vectors = embed_texts(texts)

    # 3) Convert embeddings to numpy matrix
    vectors_np = np.array(vectors, dtype="float32")

    # 4) Create FAISS index
    dimension = vectors_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_np)

    # 5) Save index and metadata
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(FAISS_INDEX_FILE))

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"FAISS index created: {FAISS_INDEX_FILE}")
    print(f"Metadata file created: {METADATA_FILE}")
    print(f"Indexed chunks: {len(metadata)}")


if __name__ == "__main__":
    build_index()