"""Pydantic schemas for the API."""

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class Source(BaseModel):
    url: str
    title: str
    category: str
    snippet: str
    quotes: list[str] = []


class ChatRequest(BaseModel):
    question: str
    chat_history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
