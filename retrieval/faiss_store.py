"""Build and query a FAISS index for semantic paper search."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings" / "paper_embeddings.npy"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "embeddings" / "metadata.json"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "faiss_index" / "papers.index"
_MODEL: SentenceTransformer | None = None
_INDEX: faiss.Index | None = None
_METADATA: list[dict[str, Any]] | None = None


def configure_logging() -> None:
    """Configure console logging for command-line execution."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_metadata(metadata_path: Path = DEFAULT_METADATA_PATH) -> list[dict[str, Any]]:
    """Load paper metadata from JSON."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if not isinstance(metadata, list):
        raise ValueError("Expected metadata JSON to contain a list.")
    return metadata


def load_embeddings(embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH) -> np.ndarray:
    """Load paper embeddings from a NumPy array file."""
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    return np.load(embeddings_path).astype(np.float32)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Return L2-normalized vectors for cosine search via inner product."""
    normalized = np.ascontiguousarray(vectors.astype(np.float32))
    faiss.normalize_L2(normalized)
    return normalized


def build_index(
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> faiss.Index:
    """Build and save a FAISS IndexFlatIP from stored paper embeddings."""
    embeddings = normalize_vectors(load_embeddings(embeddings_path))
    if embeddings.ndim != 2:
        raise ValueError("Expected embeddings to be a two-dimensional array.")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    LOGGER.info("Saved FAISS index with %s vectors to %s", index.ntotal, index_path)
    return index


def load_index(index_path: Path = DEFAULT_INDEX_PATH) -> faiss.Index:
    """Load a persisted FAISS index, building it first when missing."""
    if not index_path.exists():
        LOGGER.info("FAISS index not found at %s; building it now.", index_path)
        return build_index(index_path=index_path)
    return faiss.read_index(str(index_path))


def get_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Return a cached SentenceTransformer model."""
    global _MODEL
    if _MODEL is None:
        LOGGER.info("Loading SentenceTransformer model %s", model_name)
        _MODEL = SentenceTransformer(model_name)
    return _MODEL


def load_store(
    index_path: Path = DEFAULT_INDEX_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> tuple[faiss.Index, list[dict[str, Any]]]:
    """Load and cache the FAISS index plus paper metadata."""
    global _INDEX, _METADATA
    if _INDEX is None:
        _INDEX = load_index(index_path=index_path)
    if _METADATA is None:
        _METADATA = load_metadata(metadata_path=metadata_path)
    if _INDEX.ntotal != len(_METADATA):
        raise ValueError(
            f"Index/vector count mismatch: {_INDEX.ntotal} index rows vs {len(_METADATA)} metadata rows."
        )
    return _INDEX, _METADATA


def search(query_text: str, top_k: int = 5, model_name: str = DEFAULT_MODEL_NAME) -> list[dict[str, Any]]:
    """Search the FAISS index and return top matching paper metadata with scores."""
    if not query_text.strip():
        raise ValueError("query_text must not be empty.")
    index, metadata = load_store()
    model = get_model(model_name=model_name)

    query_vector = model.encode([query_text], convert_to_numpy=True).astype(np.float32)
    query_vector = normalize_vectors(query_vector)
    limit = min(top_k, len(metadata))
    scores, indices = index.search(query_vector, limit)

    results: list[dict[str, Any]] = []
    for score, index_id in zip(scores[0], indices[0]):
        if index_id < 0:
            continue
        paper = metadata[int(index_id)]
        results.append(
            {
                "id": paper["id"],
                "title": paper["title"],
                "category": paper.get("category", "unknown"),
                "year": paper.get("year"),
                "score": float(max(0.0, min(1.0, score))),
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for FAISS index operations."""
    parser = argparse.ArgumentParser(description="Build and query the FAISS paper index.")
    parser.add_argument("--build", action="store_true", help="Build the index before test queries.")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Run FAISS index smoke tests from the command line."""
    configure_logging()
    args = parse_args()
    if args.build:
        build_index()

    queries = [
        "kernel methods for additive models",
        "high dimensional statistics learning rates",
        "semiparametric regression classification",
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        for result in search(query, top_k=args.top_k):
            print(f"- {result['score']:.3f} | {result['title']} ({result['category']}, {result['year']})")


if __name__ == "__main__":
    main()
