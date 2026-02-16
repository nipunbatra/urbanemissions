"""Embedding generation and ChromaDB upsert."""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "urbanemissions"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 100


def get_chroma_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def embed_and_store(chunks: list[dict]) -> int:
    """Embed all chunks and upsert into ChromaDB. Returns number of chunks stored."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    collection = get_chroma_collection()

    print(f"Embedding and upserting {len(chunks)} chunks...")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]

        ids = [c["id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        done = min(i + BATCH_SIZE, len(chunks))
        print(f"  Upserted: {done}/{len(chunks)}")

    total = collection.count()
    print(f"Collection now has {total} chunks")
    return total
