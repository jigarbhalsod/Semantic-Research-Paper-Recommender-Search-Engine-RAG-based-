"""LangChain RAG pipeline for grounded research paper question answering."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval import faiss_store


LOGGER = logging.getLogger(__name__)
DEFAULT_LLM_REPO_ID = "google/flan-t5-base"


def configure_logging() -> None:
    """Configure console logging for command-line execution."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def retrieve_papers(question: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve relevant papers from the FAISS store."""
    return faiss_store.search(question, top_k=top_k)


def format_context(results: list[dict[str, Any]]) -> str:
    """Format retrieved papers as compact RAG context."""
    context_lines: list[str] = []
    for index, result in enumerate(results, start=1):
        context_lines.append(
            f"Source {index}: {result['title']} "
            f"(category: {result.get('category', 'unknown')}, year: {result.get('year')})."
        )
    return "\n".join(context_lines)


def create_llm(repo_id: str = DEFAULT_LLM_REPO_ID) -> Any:
    """Create a Hugging Face Endpoint LLM when an HF token is available."""
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        return None
    try:
        from langchain_community.llms import HuggingFaceEndpoint

        return HuggingFaceEndpoint(
            repo_id=repo_id,
            huggingfacehub_api_token=token,
            max_new_tokens=256,
            temperature=0.1,
        )
    except Exception as exc:  # pragma: no cover - depends on external service state.
        LOGGER.warning("Could not initialize Hugging Face LLM; using stub fallback: %s", exc)
        return None


def stub_completion(inputs: dict[str, str]) -> str:
    """Return a deterministic grounded answer when no external LLM is configured."""
    question = inputs["question"]
    context = inputs["context"]
    source_titles = [
        line.split(": ", 1)[1].split(" (category:", 1)[0]
        for line in context.splitlines()
        if line.startswith("Source ")
    ]
    if not source_titles:
        return "I could not find relevant papers in the local index."
    title_list = "; ".join(source_titles[:3])
    return (
        f"Based on the retrieved papers, the best grounded response to '{question}' "
        f"should focus on: {title_list}. Configure HF_TOKEN to enable generated LLM answers."
    )


def build_chain(repo_id: str = DEFAULT_LLM_REPO_ID) -> Any:
    """Build a LangChain runnable that maps question/context to an answer."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer using only the provided research paper context. "
                "If the context is insufficient, say what is missing. Include source titles.",
            ),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )
    llm = create_llm(repo_id=repo_id)
    if llm is None:
        return RunnableLambda(stub_completion)
    return prompt | llm | StrOutputParser()


class RagPipeline:
    """A small LangChain-backed RAG pipeline over the local FAISS store."""

    def __init__(self, top_k: int = 5, repo_id: str = DEFAULT_LLM_REPO_ID) -> None:
        """Initialize the RAG pipeline."""
        self.top_k = top_k
        self.chain = build_chain(repo_id=repo_id)

    def answer(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """Answer a question with retrieved source titles."""
        if not question.strip():
            raise ValueError("question must not be empty.")
        limit = top_k or self.top_k
        results = retrieve_papers(question, top_k=limit)
        context = format_context(results)
        answer = self.chain.invoke({"question": question, "context": context})
        return {
            "question": question,
            "answer": str(answer),
            "sources": [str(result["title"]) for result in results],
            "results": results,
        }


def answer_question(question: str, top_k: int = 5) -> dict[str, Any]:
    """Answer a question using a short-lived RAG pipeline instance."""
    return RagPipeline(top_k=top_k).answer(question, top_k=top_k)


def main() -> None:
    """Run a command-line RAG smoke test."""
    configure_logging()
    response = answer_question("Which papers discuss high dimensional learning rates?", top_k=3)
    print(response["answer"])
    print("Sources:")
    for source in response["sources"]:
        print(f"- {source}")


if __name__ == "__main__":
    main()
