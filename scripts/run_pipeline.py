"""One-command pipeline: scrape -> extract -> chunk -> embed."""

import time


def main():
    start = time.time()

    # Stage 1: Crawl
    print("=" * 60)
    print("STAGE 1: Crawling urbanemissions.info")
    print("=" * 60)
    from scraper.crawl import run as crawl_run

    crawl_run()

    # Stage 2: Extract
    print("\n" + "=" * 60)
    print("STAGE 2: Extracting text from HTML")
    print("=" * 60)
    from scraper.extract import extract_all

    documents = extract_all()

    # Stage 3: Chunk
    print("\n" + "=" * 60)
    print("STAGE 3: Chunking documents")
    print("=" * 60)
    from indexer.chunker import chunk_documents

    chunks = chunk_documents(documents)

    # Stage 4: Embed + Store
    print("\n" + "=" * 60)
    print("STAGE 4: Embedding and storing in ChromaDB")
    print("=" * 60)
    from indexer.embed import embed_and_store

    total = embed_and_store(chunks)

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"DONE. {total} chunks indexed in {elapsed:.1f}s")
    print("=" * 60)
    print("\nRun the server with:")
    print("  uv run python -m backend.app")


if __name__ == "__main__":
    main()
