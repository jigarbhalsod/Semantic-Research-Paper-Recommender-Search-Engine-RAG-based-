"""Tests for semantic retrieval backends."""

from __future__ import annotations

from retrieval import faiss_store


REQUIRED_KEYS = {"id", "title", "category", "year", "score"}


def test_faiss_search_returns_top_k_results() -> None:
    """FAISS search should return exactly the requested number of results."""
    results = faiss_store.search("kernel methods for additive models", top_k=3)
    assert len(results) == 3


def test_faiss_results_have_required_keys() -> None:
    """Each FAISS result should expose the API result fields."""
    results = faiss_store.search("charged pseudoscalar meson decay", top_k=3)
    assert all(REQUIRED_KEYS.issubset(result.keys()) for result in results)


def test_faiss_scores_are_between_zero_and_one() -> None:
    """Cosine similarity scores should be clamped into the 0..1 range."""
    results = faiss_store.search("laser beam turbulent atmosphere", top_k=5)
    assert all(0.0 <= float(result["score"]) <= 1.0 for result in results)


def test_faiss_search_is_deterministic_for_same_query() -> None:
    """Running the same query twice should return the same ordered IDs."""
    first = faiss_store.search("nucleon spin crisis", top_k=4)
    second = faiss_store.search("nucleon spin crisis", top_k=4)
    assert [result["id"] for result in first] == [result["id"] for result in second]
