"""Build the RAG index: documents → text chunks → embeddings → Chroma."""

from chromadb.errors import ChromaError
from httpx import HTTPError, TimeoutException
from ollama import RequestError, ResponseError

from llm import create_client
from rag.chunking import DOCUMENTS_PATH, load_chunks
from rag.embeddings import OLLAMA_EMBED_MODEL, embed_chunks
from rag.vector_store import DATABASE_PATH, save_chunks


def main():
    chunks = load_chunks()
    if not chunks:
        print(
            f"No Markdown documents found in {DOCUMENTS_PATH}. "
            "Add your own files or run 'python wikipedia/fetch_wikipedia.py'."
        )
        return 1

    client, _ = create_client()
    try:
        embed_chunks(client, OLLAMA_EMBED_MODEL, chunks)
        collection = save_chunks(chunks)
        print(f"Saved {collection.count()} chunks with {OLLAMA_EMBED_MODEL} in {DATABASE_PATH}.")
    except ResponseError as error:
        if error.status_code == 404:
            print(f"Embedding model not found. Run 'ollama pull {OLLAMA_EMBED_MODEL}'.")
        else:
            print(f"Ollama error ({error.status_code}): {error.error}")
        return 1
    except (ChromaError, ConnectionError, TimeoutException, HTTPError, RequestError, ValueError) as error:
        print(f"Could not index Markdown documents ({type(error).__name__}): {error}")
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
