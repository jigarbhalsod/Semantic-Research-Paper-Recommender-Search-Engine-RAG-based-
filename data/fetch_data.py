"""Load, clean, summarize, and export research paper metadata.

The ingestion path tries Hugging Face datasets first and falls back to a local
CSV file with the expected columns when remote data is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


LOGGER = logging.getLogger(__name__)
DATASET_CANDIDATES: tuple[str, ...] = ("arxiv_access", "ccdv/arxiv-summarization")
DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "papers.csv"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "papers.json"
REQUIRED_OUTPUT_KEYS = ("id", "title", "abstract", "text", "category", "year")


def configure_logging() -> None:
    """Configure console logging for command-line execution."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def clean_text(value: Any) -> str:
    """Normalize whitespace for a text value."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def first_present(record: dict[str, Any], keys: Iterable[str]) -> Any:
    """Return the first non-empty value from a record for the given keys."""
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return ""


def parse_year(value: Any) -> int | None:
    """Extract a four-digit year from an arbitrary metadata value."""
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else None


def derive_title_from_text(value: Any, fallback_id: int) -> str:
    """Create a compact fallback title from article text when none is provided."""
    text = clean_text(value)
    if not text:
        return f"Untitled paper {fallback_id}"

    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    words = sentence.split()
    return " ".join(words[:18]).rstrip(" .") or f"Untitled paper {fallback_id}"


def normalize_record(record: dict[str, Any], fallback_id: int) -> dict[str, Any]:
    """Map dataset-specific fields to the project paper schema."""
    paper_id = clean_text(first_present(record, ("id", "paper_id", "arxiv_id")))
    title = clean_text(first_present(record, ("title", "paper_title")))
    abstract = clean_text(first_present(record, ("abstract", "summary")))
    category = clean_text(first_present(record, ("category", "categories", "primary_category")))
    year = parse_year(first_present(record, ("year", "published", "updated", "update_date")))

    if not paper_id:
        paper_id = f"paper-{fallback_id}"
    if not title:
        title = derive_title_from_text(record.get("article"), fallback_id=fallback_id)
    if not category:
        category = "unknown"

    return {
        "id": paper_id,
        "title": title,
        "abstract": abstract,
        "text": f"{title}. {abstract}".strip(),
        "category": category,
        "year": year,
    }


def load_huggingface_records(limit: int | None = None) -> list[dict[str, Any]]:
    """Load records from the first available Hugging Face arXiv dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is not installed") from exc

    last_error: Exception | None = None
    for dataset_name in DATASET_CANDIDATES:
        try:
            LOGGER.info("Loading Hugging Face dataset %s", dataset_name)
            dataset = load_dataset(dataset_name, split="train")
            if limit is not None:
                dataset = dataset.select(range(min(limit, len(dataset))))
            return [dict(row) for row in dataset]
        except Exception as exc:  # pragma: no cover - depends on network/service state.
            LOGGER.warning("Could not load %s: %s", dataset_name, exc)
            last_error = exc

    raise RuntimeError("No Hugging Face arXiv dataset could be loaded") from last_error


def load_csv_records(csv_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load paper records from a local CSV fallback file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Local CSV fallback not found: {csv_path}")

    LOGGER.info("Loading local CSV fallback %s", csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        records = [dict(row) for row in reader]

    return records[:limit] if limit is not None else records


def load_records(csv_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load paper records from Hugging Face, falling back to a local CSV."""
    try:
        return load_huggingface_records(limit=limit)
    except Exception as exc:
        LOGGER.warning("Hugging Face loading failed; falling back to CSV: %s", exc)
        return load_csv_records(csv_path=csv_path, limit=limit)


def clean_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clean, deduplicate, and normalize raw records into the output schema."""
    cleaned: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, record in enumerate(records, start=1):
        normalized = normalize_record(record, fallback_id=index)
        if not normalized["abstract"]:
            continue
        if normalized["id"] in seen_ids:
            continue

        seen_ids.add(normalized["id"])
        cleaned.append({key: normalized[key] for key in REQUIRED_OUTPUT_KEYS})

    return cleaned


def save_records(records: list[dict[str, Any]], output_path: Path) -> None:
    """Save cleaned paper records as a JSON list of dictionaries."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, ensure_ascii=False, indent=2)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build summary statistics for cleaned paper records."""
    abstract_lengths = [len(record["abstract"].split()) for record in records]
    categories = Counter(record["category"] for record in records)
    return {
        "total_papers": len(records),
        "average_abstract_length": round(mean(abstract_lengths), 2) if abstract_lengths else 0,
        "category_distribution": dict(categories.most_common(10)),
    }


def fetch_and_export(
    csv_path: Path = DEFAULT_CSV_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    limit: int | None = None,
) -> dict[str, Any]:
    """Load raw papers, clean them, export JSON, and return summary stats."""
    raw_records = load_records(csv_path=csv_path, limit=limit)
    cleaned_records = clean_records(raw_records)
    save_records(cleaned_records, output_path=output_path)
    summary = summarize_records(cleaned_records)
    LOGGER.info("Saved %s cleaned papers to %s", len(cleaned_records), output_path)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ingestion script."""
    parser = argparse.ArgumentParser(description="Fetch and clean arXiv paper metadata.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the ingestion script from the command line."""
    configure_logging()
    args = parse_args()
    summary = fetch_and_export(
        csv_path=args.csv_path,
        output_path=args.output_path,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
