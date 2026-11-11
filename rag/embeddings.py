"""Turn document chunks and search queries into vectors with Ollama."""


OLLAMA_EMBED_MODEL = "bge-m3"
EMBED_BATCH_SIZE = 32


def embed_texts(client, model, texts):
    """Create one embedding per text, a few texts at a time."""
    embeddings = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        response = client.embed(model=model, input=batch)
        if len(response.embeddings) != len(batch):
            raise ValueError("Ollama did not return an embedding for every text.")
        embeddings.extend(response.embeddings)
    return embeddings


def embed_chunks(client, model, chunks):
    """Use Ollama to turn each text chunk into a vector of numbers."""
    embeddings = embed_texts(client, model, [chunk["text"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks
