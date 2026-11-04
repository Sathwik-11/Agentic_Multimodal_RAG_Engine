# Agentic Multimodal RAG

A production-style retrieval service: documents are parsed with Docling, chunked with a structure-aware HybridChunker, embedded into Chroma, retrieved with BM25 + dense search, reranked with FlashRank, then answered by a LangGraph Corrective RAG agent. NeMo Guardrails bound the model to policy and retrieved evidence. RAGAS scores every grounded turn.

The FastAPI surface is the contract. Ingest a corpus, chat with a `thread_id`, and inspect sources, guardrail decisions, and evaluation scores on the same response.

## Design decisions

### Docling for loading

Docling is used as the document loader because it is a capable, free parser for the formats this engine actually indexes: PDF, DOCX, PPTX, HTML, Markdown, images, spreadsheets, and more. Layout is not thrown away. Each chunk keeps Docling provenance (`dl_meta`), including page numbers and bounding boxes (`l,t,r,b,page`) so a UI can highlight the cited region on the original page instead of returning a naked text snippet.

### HybridChunker for splitting

`HybridChunker` is structure-aware. It walks heading → subheading → paragraph, so a split still knows which section it came from. The OpenAI tokenizer (`cl100k_base`) is aligned with `text-embedding-3-small`, and `CHUNK_MAX_TOKENS` is the hard cap. The chunker stops at the nearest sentence boundary rather than mid-clause. Because the heading path is retained on every fragment, adjacent chunks do not need a sliding-window overlap to keep context.

### text-embedding-3-small

Embeddings use OpenAI `text-embedding-3-small`: 1536 dimensions at a lower price than the large model. For a persistent Chroma index that is queried on every turn, that dimensionality-per-dollar tradeoff is the point. The same tokenizer used at chunk time is used at embed time, so token budgets and vector quality stay aligned.

### Hybrid retrieval

A dense retriever only is weak on exact identifiers, error codes, and proper names. BM25 only is weak on paraphrase. This engine runs both: `BM25Retriever` for lexical match, Chroma cosine search for semantic match, fused with weighted reciprocal rank fusion (`EnsembleRetriever`). Candidate pools are oversized (`RETRIEVE_CANDIDATE_K`) so the reranker has something to work with.

### FlashRank reranker

Hybrid fusion improves recall, not order. `FlashrankRerank` (`ms-marco-MiniLM-L-12-v2` by default) scores each `(query, chunk)` pair with a local ONNX cross-encoder and keeps `RETRIEVE_K`. That is the set the generator actually sees, which is the highest-leverage quality step after retrieval.

### LangGraph as Corrective RAG — how context is handled

LangGraph is the control plane, not a wrapper around a single prompt. Context is split into two lifetimes:

**Durable conversation context.** Each `thread_id` owns a LangGraph checkpointer: in-process `MemorySaver` by default, or SQLite at `CHECKPOINT_PATH` so threads survive restarts. The transcript lives in `messages` (`add_messages`). Only the last five user/assistant exchanges stay verbatim. Older turns are folded by the LLM into `conversation_summary` and dropped from `messages`, so the prompt stays bounded without losing the thread. `GET /chat/{thread_id}` returns that summary plus the five recent exchanges.

**Per-turn working context.** `prepare` clears search query, retrieved docs, CRAG grade, retry count, guardrail flags, and RAGAS scores. Those fields are not mixed into long-term memory.

How that context is used:

1. **Input rail** — the raw user text is checked before any retrieval.
2. **Agent** — sees the conversation summary plus the last five exchanges. Follow-ups such as “what about the exception?” are rewritten into a standalone `retrieve_knowledge` query. Greetings that need no corpus skip retrieval.
3. **Retrieve** — hybrid + FlashRank fetch chunks. Headers include filename, page, and heading path so the generator can cite.
4. **CRAG grade** — an LLM judges which chunks are actually relevant. Irrelevant context is dropped rather than stuffed into the prompt.
5. **Rewrite** — if the grade is weak, the search query is rewritten using the conversation and the failure reason, then retrieval runs again (capped by `MAX_RETRIES`).
6. **Optional web fallback** — if local retrieval still fails and `TAVILY_API_KEY` is set, Tavily results are merged as extra context.
7. **Generate** — a chat LLM (`CHAT_MODEL`, default `gpt-4o-mini`) writes the answer from the graded chunks, the running summary, and the last five exchanges. It is instructed to stay inside that context and cite `[filename, p.N]`.
8. **Output rail + RAGAS** — the answer is checked against policy and retrieved evidence; faithfulness and answer relevancy are attached to the response.

The graph never relies on “the model will remember.” Conversation state is explicit, retrieval context is graded, and retries are bounded.

```
prepare → input_guard ─┬─ blocked → END
                       └─ agent ─┬─ output_guard → evaluate → END
                                 └─ retrieve → grade ─┬─ generate → output_guard → evaluate
                                                      ├─ rewrite_query → retrieve
                                                      └─ web_search → generate
```

### Guardrails and evaluation

NeMo Guardrails run an input self-check (jailbreak, abuse, policy) and an output self-check plus fact-check against `$relevant_chunks`. The fact rail is there so the model does not invent policy, leak a prompt, or wander off the retrieved evidence.

RAGAS then scores the live `(question, retrieved_contexts, answer)` triple: **faithfulness** (claims supported by context) and **answer relevancy** (the answer addresses the question). No labeled gold set is required, which is what you want on a live API. Set `RAGAS_MIN_FAITHFULNESS` if a weakly grounded answer should be replaced with a refusal.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

```bash
python main.py
```

API docs: http://127.0.0.1:7000/docs

## Run with Docker

The image runs as a non-root user. Chroma, SQLite checkpoints, and model caches live on a volume. Secrets stay in `.env` and are not baked into the image.

```bash
copy .env.example .env
docker compose up --build
```

API docs: http://127.0.0.1:8000/docs

```bash
docker build -t agentic-rag .
docker run --rm -p 8000:8000 --env-file .env -v rag-data:/data agentic-rag
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service status, vector count, guardrail/RAGAS flags |
| `POST` | `/ingest` | Index local paths or URLs |
| `POST` | `/ingest/upload` | Upload a file and index it |
| `POST` | `/chat` | Ask a question (`thread_id` is conversation memory) |
| `GET` | `/chat/{thread_id}` | Message history for that thread |

### Ingest

```bash
curl -X POST http://127.0.0.1:8000/ingest ^
  -H "Content-Type: application/json" ^
  -d "{\"paths\": [\"/app/documents/policy.pdf\"]}"
```

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

Reuse the same `thread_id` for follow-ups. The response includes:

- `sources` — chunk text, filename, page, heading path, bounding box
- `guardrails` — `input` / `output`: `pass` or `blocked`
- `ragas` — `faithfulness` and `answer_relevancy` in `[0, 1]`

## Configuration

| Variable | Role |
| --- | --- |
| `OPENAI_API_KEY` | Chat and embeddings |
| `CHAT_MODEL` | Default `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Default `text-embedding-3-small` |
| `CHUNK_MAX_TOKENS` | HybridChunker cap (sentence-aligned) |
| `RETRIEVE_CANDIDATE_K` | BM25 + dense pool before rerank |
| `RETRIEVE_K` | Chunks kept after FlashRank |
| `BM25_WEIGHT` / `DENSE_WEIGHT` | Hybrid fusion |
| `FLASHRANK_MODEL` | Local ONNX reranker |
| `MAX_RETRIES` | CRAG rewrite cap |
| `MAX_HISTORY_EXCHANGES` | Last N Q&A pairs kept verbatim (default 5); older turns are summarized |
| `CHECKPOINT_PATH` | SQLite thread state (`messages` + `conversation_summary` per `thread_id`) |
| `TAVILY_API_KEY` | Optional web fallback |
| `GUARDRAILS_ENABLED` | NeMo input/output/fact rails |
| `RAGAS_ENABLED` | Per-turn evaluation |
| `RAGAS_MIN_FAITHFULNESS` | `0` = score only; `>0` refuses weak answers |
