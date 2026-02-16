# urbanemissions-rag

RAG-based chat interface for [urbanemissions.info](https://urbanemissions.info) -- an air pollution knowledge platform focused on India.

Scrapes the site, chunks and embeds content into ChromaDB, and serves a chat UI backed by retrieval-augmented generation (Gemini 2.5 Flash + all-MiniLM-L6-v2 embeddings).

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file with your Google AI API key:

```
GOOGLE_API_KEY=your-key-here
```

## Usage

### Full pipeline (scrape + index + serve)

```bash
uv run python -m scripts.run_pipeline
```

This runs four stages:
1. **Crawl** -- fetches all pages from the urbanemissions.info sitemap
2. **Extract** -- parses HTML to plain text
3. **Chunk** -- splits documents into overlapping chunks
4. **Embed** -- encodes chunks with sentence-transformers and stores in ChromaDB

### Start the server (if already indexed)

```bash
uv run python -m backend.app
```

The app runs at `http://localhost:8000`. The frontend is served at `/`.

### API

- `POST /api/chat` -- send a question, get a RAG-grounded answer with sources
- `GET /api/health` -- check server status and chunk count

## Project structure

```
scraper/        Sitemap crawler and HTML text extractor
indexer/        Document chunker and ChromaDB embedding
backend/        FastAPI server and RAG pipeline
frontend/       Chat UI (single HTML file)
scripts/        Pipeline runner
data/           Scraped HTML and extracted JSON
chroma_db/      Vector store
```
