# RAG Workshop
Here's a simple RAG chat to try. Build on it or try your own approach.

## Getting started

Install **Python 3.11+** and [Ollama](https://ollama.com/download).
We recommend these Ollama models to get started (about 3.7 GB in total),
but you're welcome to choose others:

Before pulling the models, start Ollama with `ollama serve` in another terminal.

```bash
ollama pull qwen3:4b
ollama pull bge-m3
```

If you'd rather not download an Ollama model for chat responses, you can build on
this with Copilot SDK; see below. The example still uses Ollama for
embeddings with `bge-m3`.

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python chat.py
```

## What's here?

[chat.py](chat.py) is the terminal chat, and [llm.py](llm.py) handles calls to
Ollama. Most of the things to tinker with are in [rag/](rag/):
[chunking.py](rag/chunking.py) splits documents into text chunks,
[embeddings.py](rag/embeddings.py) turns the chunks and the question into vectors (embeddings),
[vector_store.py](rag/vector_store.py) saves the index in Chroma, and
[search.py](rag/search.py) picks up to four chunks that most closely match the question.
The chat sends those chunks to the model as context for its answer.

### Try the chat with and without documents

Run these commands from the repository root, with your virtual environment activated and Ollama running.

1. **Try the chat without document context.** Before building an index, start the chat:
   ```bash
   python chat.py
   ```
   Ask a question about material you plan to add. Try something specific from your own or up-to-date material that the model probably doesn't already know. Save the question and note the answer so you can compare later.

   Type `exit` to quit before continuing.

2. **Add documents.** Choose one or both options:
   - Add your own Markdown files to [wikipedia/documents/](wikipedia/documents/).
   - Fetch Wikipedia articles by adding their URLs to [wikipedia/wikipedia_urls.txt](wikipedia/wikipedia_urls.txt), then running:
     ```bash
     python wikipedia/fetch_wikipedia.py
     ```
     The downloaded articles are saved in `wikipedia/documents/`.

3. **Build the index.**
   ```bash
   python index_documents.py
   ```
   The index is stored in `storage/vector_db/`. Run this command again after adding or changing documents.

4. **Ask the same question again.**
   ```bash
   python chat.py
   ```
   The chat now uses relevant passages from your documents as context. Compare the answer with the first one: does it include information it previously missed, or correct something it got wrong?

Without an index, the chat answers without document context. If you have already built an index, the chat will use it, so the first step is only a “without documents” test before your first indexing.

In the chat, `/clear` clears the conversation, and `exit` quits.

Ollama needs to be running with the default settings, which run the models locally. Chat model and timeout settings are in [llm.py](llm.py); the embedding model is in [rag/embeddings.py](rag/embeddings.py).

## A few ideas to try

- **Show the search results:** Add a view of the retrieved text chunks and their ranking. That makes it easier to see what's sent to the model.
- **Combine search methods:** Try combining vector search with text search, such as BM25, and compare the results.
- **Try agentic RAG:** Let Copilot search your vector index — see below for one way to get started.

It can be interesting to compare both the search results and the answer, for example by phrasing the same question in different ways.

## Possible next step: Copilot SDK

If you'd like to try agentic RAG, you can use
[Copilot SDK](https://github.com/github/copilot-sdk/tree/main/python)
to let your own chat use Copilot for responses instead of Ollama.
You can then give Copilot a search tool so it can decide when to look
something up in your documents. Ollama can still handle the embeddings.

One way to get started:

- Get a basic SDK example running. Its [getting started guide](https://github.com/github/copilot-sdk/blob/main/docs/getting-started.md) covers the runtime and Copilot sign-in.
- Connect your chat to a Copilot session instead of calling Ollama for responses.
- Use the existing index and search code in [rag/](rag/) as a starting point for a tool that takes a search query and returns relevant text chunks.
- Make your document search tool the only tool available to the SDK session. Otherwise, Copilot may use other tools to read the document files directly instead of searching your index, making it harder to see what RAG adds.

  Once you have defined a custom tool named `search_documents`, configure the session like this in your SDK example:
  ```python
  session = await client.create_session(
      model="auto",
      tools=[search_documents],
      available_tools=["custom:search_documents"],
  )
  ```
  Replace `search_documents` with your own tool's name in both places. `tools` registers your tool; `available_tools` limits the session to that tool, excluding built-in tools and any MCP tools. `custom:` means it is a tool you defined yourself.

- Give the session access to the tool and try asking about your documents. Compare with the current chat, which searches before every response.

Questions and retrieved document excerpts are then sent to Copilot, so use approved material.
