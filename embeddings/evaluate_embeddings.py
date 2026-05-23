"""Evaluate semantic embedding quality with simple hand-written retrieval checks."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval import faiss_store
from tracking import wandb_logger


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationPair:
    """A query and keyword expected to appear in a relevant paper title."""

    query: str
    expected_title_keyword: str


EVALUATION_PAIRS: tuple[EvaluationPair, ...] = (
    EvaluationPair("semiparametric additive model learning rates", "additive"),
    EvaluationPair("regularized kernel methods in high dimensions", "additive"),
    EvaluationPair("charged pseudoscalar meson leptonic decay", "leptonic"),
    EvaluationPair("particle physics pseudoscalar decay processes", "pseudoscalar"),
    EvaluationPair("nonlinear nonequilibrium dynamical transport properties", "transport"),
    EvaluationPair("dynamical systems far from equilibrium", "dynamical"),
    EvaluationPair("laser beam propagation through turbulent atmosphere", "laser"),
    EvaluationPair("remote sensing through atmospheric turbulence", "remote"),
    EvaluationPair("nucleon spin crisis european muon collaboration", "nucleon"),
    EvaluationPair("EMC spin structure measurement", "spin"),
)


def configure_logging() -> None:
    """Configure console logging for command-line execution."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def contains_expected_title(results: list[dict[str, object]], expected_keyword: str) -> bool:
    """Return whether any retrieved title contains the expected keyword."""
    expected = expected_keyword.lower()
    return any(expected in str(result.get("title", "")).lower() for result in results)


def evaluate(top_k: int = 5) -> dict[str, float]:
    """Evaluate FAISS retrieval and return Precision@K and Hit Rate@K metrics."""
    rows: list[tuple[str, str, bool]] = []
    hits = 0

    for pair in EVALUATION_PAIRS:
        results = faiss_store.search(pair.query, top_k=top_k)
        hit = contains_expected_title(results, pair.expected_title_keyword)
        hits += int(hit)
        rows.append((pair.query, pair.expected_title_keyword, hit))

    total = len(EVALUATION_PAIRS)
    metrics = {
        f"precision_at_{top_k}": hits / (total * top_k),
        f"hit_rate_at_{top_k}": hits / total,
        "evaluation_pairs": float(total),
    }

    wandb_logger.init_run(
        {
            "evaluation_name": "hand_written_query_title_keyword_pairs",
            "top_k": top_k,
            "number_of_pairs": total,
        }
    )
    wandb_logger.log_metrics(metrics)
    wandb_logger.finish()

    print("Query | Expected keyword | Hit@5")
    print("--- | --- | ---")
    for query, keyword, hit in rows:
        print(f"{query} | {keyword} | {'yes' if hit else 'no'}")
    print(f"\nPrecision@{top_k}: {metrics[f'precision_at_{top_k}']:.3f}")
    print(f"Hit Rate@{top_k}: {metrics[f'hit_rate_at_{top_k}']:.3f}")
    return metrics


def main() -> None:
    """Run the embedding evaluation report."""
    configure_logging()
    evaluate(top_k=5)


if __name__ == "__main__":
    main()
