"""A small CLI RAG chat to experiment with and build on."""

from chromadb.errors import ChromaError
from httpx import HTTPError, TimeoutException
from ollama import RequestError, ResponseError

from llm import create_client, send_message
from rag.embeddings import OLLAMA_EMBED_MODEL
from rag.search import format_context, retrieve
from rag.vector_store import open_collection

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer in the user's language. "
    "When document excerpts are provided, use them as your factual basis and cite "
    "supporting sources using their titles in brackets. If they do not contain the "
    "answer, say that you could not find it in the documents. "
    "Without document excerpts, answer normally."
)


def main():
    client, model = create_client()
    embedding_model = OLLAMA_EMBED_MODEL
    history = []
    print(
        f"\nRAG workshop — Ollama ({model}).\n"
        "/clear resets the chat; exit quits."
    )
    try:
        while True:
            question = input("\nYou: ").strip()
            if question.lower() in {"exit", "/exit"}:
                break
            if not question:
                continue
            if question == "/clear":
                history.clear()
                print("Conversation cleared.")
                continue

            chunks = []
            context = ""
            try:
                collection = open_collection()
                if collection is not None:
                    chunks = retrieve(client, embedding_model, question, collection)
                    context = format_context(chunks)
            except (ChromaError, ConnectionError, TimeoutException, HTTPError, RequestError, ResponseError, ValueError) as error:
                print(f"Document search unavailable ({type(error).__name__}): {error}. Continuing without document context.")

            print(
                f"RAG: document excerpts retrieved: {len(chunks)}."
                if context else "Without RAG: no document context. Run 'python index_documents.py' to index documents."
            )
            user_message = {"role": "user", "content": question}
            system_content = SYSTEM_PROMPT
            if context:
                system_content = f"{SYSTEM_PROMPT}\n\nDocument excerpts:\n{context}"
            messages = [
                {
                    "role": "system",
                    "content": system_content,
                },
                *history,
                user_message,
            ]
            print("Ollama is generating a response… This may take a moment.", flush=True)
            try:
                answer = send_message(client, model, messages)
            except ConnectionError:
                print("Cannot connect to Ollama. Start the Ollama app or run 'ollama serve'; check OLLAMA_HOST in llm.py.")
                continue
            except TimeoutException:
                print("Ollama timed out. Try a smaller model or increase OLLAMA_TIMEOUT in llm.py.")
                continue
            except ResponseError as error:
                if error.status_code == 404:
                    print(f"Model not found. Run 'ollama pull {model}' on your Ollama server.")
                else:
                    print(f"Ollama error ({error.status_code}): {error.error}")
                continue
            except (HTTPError, RequestError, ValueError) as error:
                print(f"Request failed ({type(error).__name__}): {error}")
                continue
            history.extend([user_message, {"role": "assistant", "content": answer}])
            history = history[-6:]  # Keep the three most recent question/answer pairs.
            print(f"\nAssistant: {answer}")
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
