"""FastAPI app serving the RAG chat interface."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

from backend.models import ChatRequest, ChatResponse
from backend.rag import RAGPipeline

rag: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    print("Initializing RAG pipeline...")
    rag = RAGPipeline()
    print(f"RAG ready. {rag.chunk_count()} chunks indexed.")
    yield
    print("Shutting down.")


app = FastAPI(title="UrbanEmissions RAG", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path("frontend/index.html")
    return FileResponse(html_path, media_type="text/html")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in request.chat_history]
    answer, sources = rag.query(request.question, history)
    return ChatResponse(answer=answer, sources=sources)


@app.get("/api/health")
async def health():
    count = rag.chunk_count() if rag else 0
    return {"status": "ok", "chunks": count}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
