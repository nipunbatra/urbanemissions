"""RAG pipeline: retrieve from ChromaDB + generate with Gemini."""

import os
from collections import defaultdict

from google import genai
from sentence_transformers import SentenceTransformer

from backend.models import Source
from indexer.embed import get_chroma_collection

SYSTEM_PROMPT = """You are an expert assistant for urbanemissions.info, a comprehensive air pollution knowledge platform focused on India.

Answer the user's question using ONLY the provided context passages. Follow these rules strictly:
1. Base your answer exclusively on the provided context. Do not use outside knowledge.
2. Cite your sources by mentioning the page title in your answer (e.g., "According to [Page Title]...").
3. If the context does not contain enough information to answer the question, say "I don't have enough information from urbanemissions.info to answer this question fully."
4. Be specific and factual. Include numbers, dates, and details when available in the context.
5. If multiple sources discuss the topic, synthesize the information coherently.
6. Keep answers concise but thorough."""


class RAGPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = get_chroma_collection()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment")
        self.client = genai.Client(api_key=api_key)

    def retrieve(self, query: str, top_k: int = 6) -> list[dict]:
        """Embed query, search ChromaDB, deduplicate by URL."""
        query_embedding = self.embedder.encode(query).tolist()

        # Fetch more than needed so we can deduplicate
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 3, 20),
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            return []

        # Deduplicate: max 2 chunks per page
        url_counts: dict[str, int] = defaultdict(int)
        deduped = []

        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            url = metadata["url"]

            if url_counts[url] >= 2:
                continue
            url_counts[url] += 1

            deduped.append(
                {
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": metadata,
                    "distance": results["distances"][0][i],
                }
            )

            if len(deduped) >= top_k:
                break

        return deduped

    def generate(
        self,
        question: str,
        contexts: list[dict],
        chat_history: list[dict] | None = None,
    ) -> str:
        """Construct prompt with context and history, call Gemini."""
        # Build context block
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            meta = ctx["metadata"]
            context_parts.append(
                f"[Source {i}: {meta['title']}]\n"
                f"URL: {meta['url']}\n"
                f"Category: {meta['category']}\n"
                f"{ctx['text']}\n"
            )
        context_block = "\n---\n".join(context_parts)

        # Build conversation history
        contents = []
        if chat_history:
            for msg in chat_history[-6:]:  # Keep last 6 messages for context
                role = "user" if msg.get("role") == "user" else "model"
                contents.append(
                    genai.types.Content(
                        role=role,
                        parts=[genai.types.Part(text=msg["content"])],
                    )
                )

        # Add current question with context
        user_message = (
            f"Context from urbanemissions.info:\n\n{context_block}\n\n"
            f"Question: {question}"
        )
        contents.append(
            genai.types.Content(
                role="user",
                parts=[genai.types.Part(text=user_message)],
            )
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=4096,
            ),
        )

        return response.text

    def query(
        self, question: str, chat_history: list[dict] | None = None
    ) -> tuple[str, list[Source]]:
        """Full RAG pipeline: retrieve + generate. Returns (answer, sources)."""
        contexts = self.retrieve(question)

        if not contexts:
            return (
                "I couldn't find any relevant information from urbanemissions.info for this question.",
                [],
            )

        answer = self.generate(question, contexts, chat_history)

        sources = []
        seen_urls: dict[str, int] = {}
        for ctx in contexts:
            url = ctx["metadata"]["url"]
            title = ctx["metadata"]["title"]
            # Strip prepended title from chunk text to get raw quote
            raw = ctx["text"]
            if raw.startswith(title + "\n\n"):
                raw = raw[len(title) + 2:]
            quote = raw.strip()

            if url in seen_urls:
                # Append quote to existing source
                idx = seen_urls[url]
                sources[idx].quotes.append(quote)
            else:
                seen_urls[url] = len(sources)
                snippet = raw[:200].strip()
                if len(raw) > 200:
                    snippet += "..."
                sources.append(
                    Source(
                        url=url,
                        title=title,
                        category=ctx["metadata"]["category"],
                        snippet=snippet,
                        quotes=[quote],
                    )
                )

        return answer, sources

    def chunk_count(self) -> int:
        """Return total chunks in the collection."""
        return self.collection.count()
