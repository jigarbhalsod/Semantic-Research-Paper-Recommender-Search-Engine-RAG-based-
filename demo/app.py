"""Gradio demo for semantic research paper search and question answering."""

from __future__ import annotations

import os
from typing import Any

import gradio as gr
import httpx


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def search_papers(query: str, top_k: int) -> list[list[Any]]:
    """Call the FastAPI search endpoint and return rows for a Gradio table."""
    if not query.strip():
        return []
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{API_BASE_URL}/search",
            json={"query": query, "top_k": int(top_k), "store": "faiss"},
        )
        response.raise_for_status()
    payload = response.json()
    return [
        [
            result["title"],
            result["category"],
            result["year"],
            round(float(result["score"]), 4),
        ]
        for result in payload["results"]
    ]


def answer_question(question: str, top_k: int) -> tuple[str, str]:
    """Call the FastAPI QA endpoint and return answer plus source titles."""
    if not question.strip():
        return "", ""
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{API_BASE_URL}/qa",
            json={"question": question, "top_k": int(top_k)},
        )
        response.raise_for_status()
    payload = response.json()
    sources = "\n".join(f"- {source}" for source in payload["sources"])
    return payload["answer"], sources


with gr.Blocks(title="Semantic Paper Search") as demo:
    gr.Markdown(
        "# Semantic Research Paper Search\n"
        "Search research papers semantically and ask grounded questions over retrieved papers."
    )

    with gr.Tab("Semantic Search"):
        search_query = gr.Textbox(label="Query", placeholder="kernel methods for additive models")
        search_top_k = gr.Slider(label="Top K", minimum=1, maximum=10, step=1, value=5)
        search_button = gr.Button("Search", variant="primary")
        search_output = gr.Dataframe(
            headers=["Title", "Category", "Year", "Score"],
            datatype=["str", "str", "str", "number"],
            row_count=5,
            col_count=(4, "fixed"),
        )
        gr.Examples(
            examples=[
                ["semiparametric additive model learning rates", 5],
                ["laser beam propagation through turbulent atmosphere", 5],
                ["nucleon spin crisis european muon collaboration", 5],
            ],
            inputs=[search_query, search_top_k],
        )
        search_button.click(search_papers, inputs=[search_query, search_top_k], outputs=search_output)

    with gr.Tab("Ask a Question"):
        qa_question = gr.Textbox(
            label="Question",
            placeholder="Which papers discuss high dimensional learning rates?",
        )
        qa_top_k = gr.Slider(label="Top K", minimum=1, maximum=10, step=1, value=5)
        qa_button = gr.Button("Ask", variant="primary")
        qa_answer = gr.Textbox(label="Answer", lines=6)
        qa_sources = gr.Textbox(label="Sources", lines=6)
        gr.Examples(
            examples=[
                ["Which papers discuss high dimensional learning rates?", 5],
                ["What papers are about turbulent atmospheres?", 5],
            ],
            inputs=[qa_question, qa_top_k],
        )
        qa_button.click(answer_question, inputs=[qa_question, qa_top_k], outputs=[qa_answer, qa_sources])


if __name__ == "__main__":
    demo.launch()
