"""Read Markdown documents and split them into text chunks with a title and source."""

import re
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter


DOCUMENTS_PATH = Path(__file__).resolve().parent.parent / "wikipedia" / "documents"
# SentenceSplitter measures size and overlap in tokens, not characters.
CHUNK_SIZE = 256
CHUNK_OVERLAP = 32


def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def load_chunks(directory=DOCUMENTS_PATH, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split each Markdown document into text chunks and keep its source details."""
    chunks = []
    for path in sorted(Path(directory).glob("*.md")):
        document = path.read_text(encoding="utf-8")
        title_match = re.search(r"^#\s+(.+)$", document, re.MULTILINE)
        source_match = re.search(r"^Source:\s*(\S+)", document, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem
        source = source_match.group(1) if source_match else str(path)
        texts = chunk_text(document, chunk_size, chunk_overlap)
        for number, text in enumerate(texts, start=1):
            chunks.append({
                "chunk_id": f"{path.name}:{number}",
                "text": text,
                "title": title,
                "source": source,
            })
    return chunks
