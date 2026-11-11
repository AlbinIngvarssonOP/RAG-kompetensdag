"""Open and update the document index in Chroma."""

from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError


DATABASE_PATH = Path(__file__).resolve().parent.parent / "storage" / "vector_db"
COLLECTION_NAME = "wikipedia"
SAVE_BATCH_SIZE = 32


def open_collection(directory=DATABASE_PATH):
    """Open an existing index. Without an index, the chat can run without RAG."""
    if not (Path(directory) / "chroma.sqlite3").is_file():
        return None
    database = chromadb.PersistentClient(path=str(directory))
    try:
        collection = database.get_collection(name=COLLECTION_NAME)
    except NotFoundError:
        return None
    return collection if collection.count() else None


def save_chunks(chunks, directory=DATABASE_PATH):
    """Save text, vectors and sources. Remove chunks that are no longer present."""
    database = chromadb.PersistentClient(path=str(directory))
    collection = database.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"},
    )
    for start in range(0, len(chunks), SAVE_BATCH_SIZE):
        batch = chunks[start : start + SAVE_BATCH_SIZE]
        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[
                {"title": chunk["title"], "source": chunk["source"]}
                for chunk in batch
            ],
        )
    current_ids = {chunk["chunk_id"] for chunk in chunks}
    stale_ids = [
        identifier for identifier in collection.get()["ids"]
        if identifier not in current_ids
    ]
    if stale_ids:
        collection.delete(ids=stale_ids)
    return collection
