"""Simple LangChain tool-based agent for paper search and question answering."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag import rag_pipeline
from retrieval import faiss_store


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure console logging for command-line execution."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@tool
def search_papers(query: str) -> str:
    """Search for semantically relevant research papers."""
    results = faiss_store.search(query, top_k=5)
    return "\n".join(
        f"{item['score']:.3f}: {item['title']} ({item.get('category', 'unknown')}, {item.get('year')})"
        for item in results
    )


@tool
def answer_question(question: str) -> str:
    """Answer a question using retrieved research paper context."""
    response = rag_pipeline.answer_question(question, top_k=5)
    sources = "; ".join(response["sources"])
    return f"{response['answer']}\nSources: {sources}"


TOOLS = (search_papers, answer_question)


def is_question(text: str) -> bool:
    """Return whether text looks like a question rather than a search query."""
    normalized = text.strip().lower()
    question_starts = ("what", "why", "how", "which", "when", "where", "who", "explain", "summarize")
    return normalized.endswith("?") or normalized.startswith(question_starts)


def run_agent(user_input: str) -> dict[str, Any]:
    """Route input to the appropriate LangChain tool and return the result."""
    if not user_input.strip():
        raise ValueError("user_input must not be empty.")

    selected_tool = answer_question if is_question(user_input) else search_papers
    output = selected_tool.invoke(user_input)
    return {
        "tool": selected_tool.name,
        "input": user_input,
        "output": output,
    }


def main() -> None:
    """Run a command-line agent smoke test."""
    configure_logging()
    for query in ("kernel additive models", "Which papers discuss high dimensional learning rates?"):
        response = run_agent(query)
        print(f"\nTool: {response['tool']}")
        print(response["output"])


if __name__ == "__main__":
    main()
