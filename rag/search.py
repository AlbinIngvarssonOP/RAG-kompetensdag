"""Find relevant text chunks and format them as context for the chat."""

from .embeddings import embed_texts


DEFAULT_TOP_K = 4


def retrieve(client, model, question, collection, top_k=DEFAULT_TOP_K):
    """Turn the question into a vector and let Chroma find the closest text chunks."""
    if top_k < 1:
        return []
    count = collection.count()
    if count == 0:
        return []
    query_embeddings = embed_texts(client, model, [question])
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=min(top_k, count),
        include=["documents", "metadatas"],
    )
    chunks = []
    for chunk_id, text, metadata in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0],
    ):
        chunks.append({
            "chunk_id": chunk_id,
            "text": text,
            "title": metadata["title"],
            "source": metadata["source"],
        })
    return chunks


def format_context(chunks):
    """Combine the search results' text and sources into context for the chat response."""
    excerpts = []
    for chunk in chunks:
        source = f" ({chunk['source']})" if chunk["source"] else ""
        excerpts.append(f"[{chunk['title']}{source}]\n{chunk['text']}")
    return "\n\n---\n\n".join(excerpts)
