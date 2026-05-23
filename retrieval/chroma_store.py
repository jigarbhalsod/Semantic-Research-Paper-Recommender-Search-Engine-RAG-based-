"""Build and query a persistent ChromaDB store for semantic paper search."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings" / "paper_embeddings.npy"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "embeddings" / "metadata.json"
DEFAULT_DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "research_papers"
_MODEL: SentenceTransformer | None = None
_CLIENT: chromadb.PersistentClient | None = None
_COLLECTION: Any | None = None


def configure_logging() -> None:
    """Configure console logging for command-line execution."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Return a cached SentenceTransformer model."""
    global _MODEL
    if _MODEL is None:
        LOGGER.info("Loading SentenceTransformer model %s", model_name)
        _MODEL = SentenceTransformer(model_name)
    return _MODEL


def get_client(db_path: Path = DEFAULT_DB_PATH) -> chromadb.PersistentClient:
    """Return a cached ChromaDB persistent client."""
    global _CLIENT
    if _CLIENT is None:
        db_path.mkdir(parents=True, exist_ok=True)
        _CLIENT = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False),
        )
    return _CLIENT


def get_collection(db_path: Path = DEFAULT_DB_PATH, collection_name: str = COLLECTION_NAME) -> Any:
    """Return the cached ChromaDB collection, creating it if needed."""
    global _COLLECTION
    if _COLLECTION is None:
        _COLLECTION = get_client(db_path=db_path).get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _COLLECTION


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


def build_store(
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    batch_size: int = 500,
) -> None:
    """Populate ChromaDB with paper embeddings and metadata."""
    embeddings = load_embeddings(embeddings_path)
    metadata = load_metadata(metadata_path)
    if embeddings.shape[0] != len(metadata):
        raise ValueError("Embeddings and metadata row counts do not match.")

    collection = get_collection()
    if collection.count() > 0:
        LOGGER.info("Collection already contains %s papers; skipping add.", collection.count())
        return

    for start_index in range(0, len(metadata), batch_size):
        end_index = min(start_index + batch_size, len(metadata))
        batch_metadata = metadata[start_index:end_index]
        collection.add(
            ids=[str(item["id"]) for item in batch_metadata],
            embeddings=embeddings[start_index:end_index].tolist(),
            metadatas=[
                {
                    "title": item["title"],
                    "category": item.get("category", "unknown"),
                    "year": item.get("year") if item.get("year") is not None else "",
                }
                for item in batch_metadata
            ],
        )
        LOGGER.info("Added papers %s-%s to ChromaDB", start_index + 1, end_index)


def search(query_text: str, top_k: int = 5, model_name: str = DEFAULT_MODEL_NAME) -> list[dict[str, Any]]:
    """Query ChromaDB and return top matching paper metadata with scores."""
    if not query_text.strip():
        raise ValueError("query_text must not be empty.")
    collection = get_collection()
    if collection.count() == 0:
        build_store()

    model = get_model(model_name=model_name)
    query_embedding = model.encode([query_text], convert_to_numpy=True).astype(np.float32)[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    output: list[dict[str, Any]] = []
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for paper_id, metadata, distance in zip(ids, metadatas, distances):
        score = 1.0 - float(distance)
        year = metadata.get("year")
        output.append(
            {
                "id": paper_id,
                "title": metadata.get("title", ""),
                "category": metadata.get("category", "unknown"),
                "year": int(year) if isinstance(year, str) and year.isdigit() else year or None,
                "score": float(max(0.0, min(1.0, score))),
            }
        )
    return output


def delete_collection(collection_name: str = COLLECTION_NAME) -> None:
    """Delete the ChromaDB collection if it exists."""
    global _COLLECTION
    client = get_client()
    try:
        client.delete_collection(collection_name)
    except ValueError:
        LOGGER.info("Collection %s does not exist.", collection_name)
    _COLLECTION = None


def get_count() -> int:
    """Return the number of papers currently stored in ChromaDB."""
    return int(get_collection().count())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ChromaDB store operations."""
    parser = argparse.ArgumentParser(description="Build and query the ChromaDB paper store.")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Run ChromaDB store smoke tests from the command line."""
    configure_logging()
    args = parse_args()
    if args.rebuild:
        delete_collection()
    build_store()
    print(f"ChromaDB paper count: {get_count()}")

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
