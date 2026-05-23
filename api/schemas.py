"""Pydantic schemas for the semantic paper search API."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request payload for semantic paper search."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    store: str = Field(default="faiss")


class SearchResult(BaseModel):
    """A single semantic search result."""

    id: str
    title: str
    category: str
    year: int | None = None
    score: float


class SearchResponse(BaseModel):
    """Response payload for semantic paper search."""

    query: str
    results: List[SearchResult]
    latency_ms: float


class QARequest(BaseModel):
    """Request payload for research paper question answering."""

    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class QAResponse(BaseModel):
    """Response payload for research paper question answering."""

    question: str
    answer: str
    sources: List[str]
    latency_ms: float
