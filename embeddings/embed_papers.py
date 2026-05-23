"""Generate dense semantic embeddings for cleaned research paper records."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking import wandb_logger


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "papers.json"
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings" / "paper_embeddings.npy"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "embeddings" / "metadata.json"


def configure_logging() -> None:
    """Configure console logging for command-line execution."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_papers(data_path: Path) -> list[dict[str, Any]]:
    """Load cleaned paper records from a JSON file."""
    if not data_path.exists():
        raise FileNotFoundError(f"Paper data file not found: {data_path}")
    with data_path.open("r", encoding="utf-8") as data_file:
        papers = json.load(data_file)
    if not isinstance(papers, list):
        raise ValueError("Expected papers JSON to contain a list of records.")
    return papers


def extract_metadata(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract compact metadata saved alongside embeddings."""
    return [
        {
            "id": str(paper["id"]),
            "title": str(paper["title"]),
            "category": str(paper.get("category", "unknown")),
            "year": paper.get("year"),
        }
        for paper in papers
    ]


def encode_batches(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> tuple[np.ndarray, list[float]]:
    """Embed texts in batches and return the matrix plus per-batch timings."""
    batches: list[np.ndarray] = []
    batch_times: list[float] = []

    for start_index in range(0, len(texts), batch_size):
        batch = texts[start_index : start_index + batch_size]
        start_time = time.perf_counter()
        batch_embeddings = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - start_time
        batch_times.append(elapsed)
        batches.append(batch_embeddings.astype(np.float32))
        LOGGER.info(
            "Embedded batch %s-%s in %.2fs",
            start_index + 1,
            start_index + len(batch),
            elapsed,
        )
        wandb_logger.log_metrics(
            {
                "batch_index": len(batch_times),
                "batch_size": len(batch),
                "batch_time_seconds": elapsed,
            }
        )

    if not batches:
        raise ValueError("No texts were available to embed.")
    return np.vstack(batches), batch_times


def save_embeddings(
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    embeddings_path: Path,
    metadata_path: Path,
) -> None:
    """Persist embedding matrix and paper metadata to disk."""
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)


def generate_embeddings(
    data_path: Path = DEFAULT_DATA_PATH,
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Generate and save paper embeddings while logging experiment metrics."""
    papers = load_papers(data_path)
    texts = [str(paper["text"]) for paper in papers]
    metadata = extract_metadata(papers)

    LOGGER.info("Loading SentenceTransformer model %s", model_name)
    model = SentenceTransformer(model_name)
    embedding_dimension = model.get_sentence_embedding_dimension()
    wandb_logger.init_run(
        {
            "model_name": model_name,
            "embedding_dimension": embedding_dimension,
            "number_of_papers": len(papers),
            "batch_size": batch_size,
        }
    )

    tracemalloc.start()
    total_start = time.perf_counter()
    try:
        embeddings, batch_times = encode_batches(model=model, texts=texts, batch_size=batch_size)
        total_time = time.perf_counter() - total_start
        _, peak_memory = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    metrics = {
        "total_embedding_time_seconds": total_time,
        "average_batch_time_seconds": float(np.mean(batch_times)),
        "peak_memory_mb": round(peak_memory / (1024 * 1024), 2),
        "embedding_rows": embeddings.shape[0],
        "embedding_dimension": embeddings.shape[1],
    }
    wandb_logger.log_metrics(metrics)
    wandb_logger.finish()

    save_embeddings(
        embeddings=embeddings,
        metadata=metadata,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )
    LOGGER.info("Saved embeddings to %s and metadata to %s", embeddings_path, metadata_path)
    print(f"Embedding matrix shape: {embeddings.shape}")
    return embeddings


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for embedding generation."""
    parser = argparse.ArgumentParser(description="Generate paper embeddings.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--embeddings-path", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    """Run embedding generation from the command line."""
    configure_logging()
    args = parse_args()
    generate_embeddings(
        data_path=args.data_path,
        embeddings_path=args.embeddings_path,
        metadata_path=args.metadata_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
