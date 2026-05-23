"""Tests for the FastAPI search and QA endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_health_returns_200() -> None:
    """The health endpoint should report service status."""
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_search_valid_payload_returns_results() -> None:
    """The search endpoint should return semantic results."""
    with TestClient(app) as client:
        response = client.post(
            "/search",
            json={"query": "kernel methods", "top_k": 2, "store": "faiss"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "kernel methods"
    assert len(payload["results"]) == 2


def test_qa_valid_question_returns_answer_and_sources() -> None:
    """The QA endpoint should return an answer and source titles."""
    with TestClient(app) as client:
        response = client.post(
            "/qa",
            json={"question": "Which papers discuss learning rates?", "top_k": 2},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["sources"]


def test_search_missing_query_returns_422() -> None:
    """The search endpoint should reject payloads without a query."""
    with TestClient(app) as client:
        response = client.post("/search", json={"top_k": 2, "store": "faiss"})
    assert response.status_code == 422
