# Agentic RAG Engine

Docling loads and chunks documents, OpenAI embeds them into Chroma, and LangGraph runs agentic hybrid retrieval with Corrective RAG, NeMo Guardrails, and RAGAS scoring plus conversation memory. The engine is exposed as a FastAPI service.

## Pipeline

1. **Ingest** — `DoclingLoader` parses PDFs and other files. `HybridChunker` splits them with the OpenAI tokenizer (`cl100k_base`), aligned to `text-embedding-3-small`.
2. **Index** — Chunks are embedded with `text-embedding-3-small` and stored in a persistent Chroma collection.
3. **Retrieve** — `BM25Retriever` (keyword) and Chroma dense search (semantic) are fused with weighted reciprocal rank fusion, then `FlashrankRerank` rescores the candidates and keeps `RETRIEVE_K`.
4. **Guardrails** — NeMo Guardrails run an input self-check (jailbreak, abuse, policy) before the agent, then an output self-check plus fact-check against retrieved chunks before the user sees an answer.
5. **Evaluate** — RAGAS scores faithfulness and answer relevancy on the live `(question, contexts, answer)` triple. No labeled gold set is required. DeepEval was not used: it is built around pytest `LLMTestCase`s, while this service already has a LangGraph turn that RAGAS can score in-process.
6. **Runtime** — A LangGraph agent decides whether to retrieve. Reranked chunks are graded (CRAG). Irrelevant hits trigger query rewrite and a second search. If retrieval still fails and `TAVILY_API_KEY` is set, the graph falls back to web search. Checkpointers keep multi-turn context per `thread_id`.

```
prepare → input_guard ─┬─ (blocked) → END
                       └─ agent ─┬─ output_guard → evaluate → END
                                 └─ retrieve → grade_documents ─┬─ generate → output_guard → evaluate
                                                                ├─ rewrite_query → retrieve
                                                                └─ web_search → generate
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

## Run the API

```bash
python main.py
```

Or:

```bash
uvicorn main:app --reload --port 8000
```

Open docs at http://127.0.0.1:8000/docs

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service status and vector count |
| `POST` | `/ingest` | Index local paths or URLs |
| `POST` | `/ingest/upload` | Upload files (multipart) and index them |
| `POST` | `/chat` | Ask a question (`thread_id` keeps conversation memory) |
| `GET` | `/chat/{thread_id}` | Return that thread's message history |

### Ingest from a path or URL

```bash
curl -X POST http://127.0.0.1:8000/ingest ^
  -H "Content-Type: application/json" ^
  -d "{\"paths\": [\"C:\\\\docs\\\\policy.pdf\"]}"
```

### Upload a file

In `/docs`, open `POST /ingest/upload`, click **Choose File**, and pick a local document.

```bash
curl -X POST http://127.0.0.1:8000/ingest/upload ^
  -F "file=@policy.pdf"
```

### Chat

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What does the policy say about remote work?\", \"thread_id\": \"session-1\"}"
```

Reuse the same `thread_id` for follow-up questions. Set `CHECKPOINT_PATH` in `.env` to persist that state in SQLite across restarts.

The chat response includes `guardrails` (`input`/`output`: `pass` or `blocked`) and `ragas` (`faithfulness`, `answer_relevancy` in `[0, 1]`). Set `GUARDRAILS_ENABLED=false` or `RAGAS_ENABLED=false` to skip those steps. Set `RAGAS_MIN_FAITHFULNESS` (for example `0.5`) to replace weakly grounded answers with a refusal.
