"""Text chunking with overlap for RAG indexing."""

import re


def url_to_slug(url: str) -> str:
    """Convert URL to a short slug for chunk IDs."""
    slug = url.replace("https://", "").replace("http://", "")
    slug = re.sub(r"[^a-zA-Z0-9]", "_", slug)
    return slug.strip("_")


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return re.split(r"(?<=[.!?])\s+", text)


def chunk_text(
    text: str,
    title: str,
    chunk_size: int = 800,
    overlap: int = 200,
    min_chunk_size: int = 100,
) -> list[str]:
    """Split text into overlapping chunks, respecting paragraph/sentence boundaries."""
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        # If adding this paragraph exceeds chunk_size, finalize current chunk
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current)
            # Start new chunk with overlap from end of previous
            if len(current) > overlap:
                # Find a sentence boundary near the overlap point
                overlap_text = current[-(overlap):]
                sentences = split_into_sentences(overlap_text)
                if len(sentences) > 1:
                    current = " ".join(sentences[1:]) + "\n\n" + para
                else:
                    current = overlap_text + "\n\n" + para
            else:
                current = current + "\n\n" + para
        else:
            current = (current + "\n\n" + para).strip() if current else para

        # If a single paragraph is very long, force split it
        while len(current) > chunk_size * 1.5:
            # Split at sentence boundary near chunk_size
            sentences = split_into_sentences(current)
            chunk = ""
            remaining_sentences = []
            for i, sent in enumerate(sentences):
                if len(chunk) + len(sent) + 1 > chunk_size and chunk:
                    remaining_sentences = sentences[i:]
                    break
                chunk = (chunk + " " + sent).strip() if chunk else sent
            else:
                # Couldn't split by sentences, hard split
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap :]
                continue

            chunks.append(chunk)
            current = " ".join(remaining_sentences)

    if current:
        chunks.append(current)

    # Prepend title and filter short chunks
    result = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) >= min_chunk_size:
            result.append(f"{title}\n\n{chunk}")

    return result


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk all extracted documents. Returns list of chunk dicts with metadata."""
    all_chunks = []

    for doc in documents:
        text = doc["text"]
        title = doc["title"]
        url = doc["url"]
        category = doc["category"]
        slug = url_to_slug(url)

        chunks = chunk_text(text, title)

        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "id": f"{slug}_{i}",
                    "text": chunk,
                    "metadata": {
                        "url": url,
                        "title": title,
                        "category": category,
                        "chunk_index": i,
                    },
                }
            )

    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks
