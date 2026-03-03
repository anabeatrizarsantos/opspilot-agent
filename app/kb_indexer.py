import json
from pathlib import Path
from typing import Dict, List

from app.embeddings_client import embed_texts


KB_FOLDER = Path("kb")
INDEX_FILE = Path("data/kb/index.json")


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
    Build a vector index from kb/*.txt using chunking.
    Output format: a JSON array with entries:
      { id, source_file, chunk_index, text, embedding }
    """
    kb_files = load_kb_files()

    # 1) Create chunks
    entries: List[Dict] = []
    for f in kb_files:
        chunks = chunk_text(f["content"], max_words=180, overlap_words=40)
        for i, ch in enumerate(chunks):
            entries.append(
                {
                    "id": f"{f['filename']}::chunk{i}",
                    "source_file": f["filename"],
                    "chunk_index": i,
                    "text": ch,
                }
            )

    if not entries:
        raise ValueError("No KB content found. Add .txt files under kb/.")

    # 2) Generate embeddings (batch)
    texts = [e["text"] for e in entries]
    vectors = embed_texts(texts)

    # 3) Attach embeddings
    for e, v in zip(entries, vectors):
        e["embedding"] = v

    # 4) Save index
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"KB index created: {INDEX_FILE} (chunks: {len(entries)})")


if __name__ == "__main__":
    build_index()