# Semantic Research Paper Recommender and Search Engine

A production-oriented semantic search and recommendation system for research papers. It ingests paper metadata, generates transformer embeddings, stores vectors in FAISS and ChromaDB, serves semantic search and RAG question answering through FastAPI, exposes a Gradio demo, and logs embedding/evaluation runs to Weights & Biases.

## Architecture

```text
Raw arXiv / CSV data
        |
        v
data/fetch_data.py
        |
        v
data/papers.json
        |
        v
embeddings/embed_papers.py -----> tracking/wandb_logger.py
        |
        v
embeddings/paper_embeddings.npy + embeddings/metadata.json
        |
        +------> retrieval/faiss_store.py ----+
        |                                      |
        +------> retrieval/chroma_store.py     |
                                               v
                                      rag/rag_pipeline.py
                                               |
                                               v
api/main.py <-------------------------- rag/agent.py
        |
        v
demo/app.py
```

## Setup

```bash
git clone https://github.com/jigarbhalsod/Semantic-Research-Paper-Recommender-Search-Engine-RAG-based-.git
cd Semantic-Research-Paper-Recommender-Search-Engine-RAG-based-
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
```

Set optional environment variables in `.env`:

```text
WANDB_API_KEY=
HF_TOKEN=
OPENAI_API_KEY=
```

The project runs without these keys. W&B uses offline mode when `WANDB_API_KEY` is missing, and the RAG pipeline uses a deterministic stub response when `HF_TOKEN` is missing.

## Run Phase Scripts

Fetch and clean data:

```bash
python data\fetch_data.py --limit 1000
```

Generate embeddings:

```bash
python embeddings\embed_papers.py
```

Build and test FAISS retrieval:

```bash
python retrieval\faiss_store.py --build
```

Build and test ChromaDB retrieval:

```bash
python retrieval\chroma_store.py --rebuild
```

Evaluate embeddings:

```bash
python embeddings\evaluate_embeddings.py
```

Run the RAG pipeline:

```bash
python rag\rag_pipeline.py
```

Run the tool router:

```bash
python rag\agent.py
```

## API

Start the FastAPI backend:

```bash
uvicorn api.main:app --reload
```

Endpoints:

```text
GET  /health
POST /search
POST /qa
```

Example search request:

```bash
curl -X POST http://localhost:8000/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"kernel methods for additive models\",\"top_k\":5,\"store\":\"faiss\"}"
```

Example QA request:

```bash
curl -X POST http://localhost:8000/qa ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Which papers discuss high dimensional learning rates?\",\"top_k\":5}"
```

## Gradio Demo

Start the API first, then run:

```bash
python demo\app.py
```

The demo expects the API at `http://localhost:8000` by default. Override it with:

```bash
set API_BASE_URL=https://your-api-host.example.com
python demo\app.py
```

## Hugging Face Spaces Deployment

The `demo/` directory is ready for a Gradio Space:

```text
demo/
+-- app.py
+-- README.md
+-- requirements.txt
```

Create a new Hugging Face Space with the Gradio SDK, upload the `demo/` contents, and set `API_BASE_URL` in the Space secrets or variables to point at a deployed FastAPI backend.

## W&B Tracking

Project name: `semantic-paper-search`

Dashboard placeholder: `https://wandb.ai/<entity>/semantic-paper-search`

Embedding generation logs model name, embedding dimension, number of papers, batch size, batch timing, total embedding time, and peak memory. Evaluation logs Precision@5 and Hit Rate@5.

## Testing

```bash
python -m pytest tests\
```

CI runs tests with coverage on pushes to `main` and pull requests.

## Repository Layout

```text
data/         Data ingestion and cleaned JSON export
embeddings/   Embedding generation and evaluation
retrieval/    FAISS and ChromaDB vector stores
rag/          LangChain RAG pipeline and tool router
api/          FastAPI backend
demo/         Gradio interface for Hugging Face Spaces
tracking/     W&B logging helpers
tests/        Retrieval and API tests
```

## Limitations

- The committed dataset is a small smoke-test corpus; run ingestion with a larger limit for real retrieval quality.
- The fallback Hugging Face dataset lacks explicit title/category/year fields, so missing titles are derived from article text and categories default to `unknown`.
- The default RAG answer path is a stub unless `HF_TOKEN` is configured.
- FAISS and ChromaDB indexes are generated local artifacts and are intentionally ignored by Git.

## Future Improvements

- Add a larger curated benchmark with labeled relevance judgments.
- Add hybrid retrieval with BM25 plus dense vectors.
- Add reranking with a cross-encoder.
- Deploy the FastAPI backend behind authentication and rate limits.
- Add async batch ingestion jobs and scheduled W&B evaluation reports.
