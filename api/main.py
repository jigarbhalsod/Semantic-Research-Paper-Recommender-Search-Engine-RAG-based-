"""FastAPI application entry point for semantic paper search and QA."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import QARequest, QAResponse, SearchRequest, SearchResponse
from rag.rag_pipeline import RagPipeline
from retrieval import chroma_store, faiss_store


LOGGER = logging.getLogger(__name__)
APP_STATE: dict[str, Any] = {}


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load retrieval and RAG resources once for the app lifetime."""
    configure_logging()
    LOGGER.info("Loading FAISS store and RAG pipeline.")
    faiss_store.load_store()
    APP_STATE["rag_pipeline"] = RagPipeline(top_k=5)
    APP_STATE["model_name"] = faiss_store.DEFAULT_MODEL_NAME
    yield


app = FastAPI(
    title="Semantic Research Paper Search API",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def elapsed_ms(start_time: float) -> float:
    """Return elapsed milliseconds from a perf-counter start time."""
    return round((time.perf_counter() - start_time) * 1000, 2)


@app.get("/health")
def health() -> dict[str, Any]:
    """Return service health and model metadata."""
    return {
        "status": "ok",
        "model": APP_STATE.get("model_name", faiss_store.DEFAULT_MODEL_NAME),
        "vector_store": "faiss",
    }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Run semantic paper search against FAISS or ChromaDB."""
    start_time = time.perf_counter()
    store = request.store.lower()
    try:
        if store == "faiss":
            results = faiss_store.search(request.query, top_k=request.top_k)
        elif store == "chroma":
            results = chroma_store.search(request.query, top_k=request.top_k)
        else:
            raise HTTPException(status_code=400, detail="store must be 'faiss' or 'chroma'.")
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Search failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SearchResponse(query=request.query, results=results, latency_ms=elapsed_ms(start_time))


@app.post("/qa", response_model=QAResponse)
def qa(request: QARequest) -> QAResponse:
    """Answer a research paper question using the RAG pipeline."""
    start_time = time.perf_counter()
    try:
        pipeline = APP_STATE.get("rag_pipeline") or RagPipeline(top_k=request.top_k)
        response = pipeline.answer(request.question, top_k=request.top_k)
    except Exception as exc:
        LOGGER.exception("Question answering failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QAResponse(
        question=request.question,
        answer=response["answer"],
        sources=response["sources"],
        latency_ms=elapsed_ms(start_time),
    )
