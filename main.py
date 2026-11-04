"""FastAPI app for the agentic RAG engine."""

from __future__ import annotations

import asyncio
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.config import ROOT_DIR, settings
from rag.graph import run_turn, thread_history
from rag.ingest import SUPPORTED_SUFFIXES, ingest
from rag.vectorstore import collection_count

logger = logging.getLogger(__name__)
UPLOAD_DIR = ROOT_DIR / "uploads"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Agentic RAG Engine",
    description="Docling ingest, hybrid retrieval, NeMo Guardrails, RAGAS, LangGraph CRAG",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    paths: list[str] = Field(
        default_factory=list,
        description="Local file/directory paths or HTTP(S) URLs",
    )


class IngestResponse(BaseModel):
    indexed_chunks: int
    collection_size: int
    sources: list[str]


class ChatRequest(BaseModel):
    question: str
    thread_id: str = "default"


class Source(BaseModel):
    content: str
    filename: str | None = None
    source: str | None = None
    page: int | None = None
    headings: str | None = None
    bbox: str | None = None


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    search_query: str = ""
    retrieval_grade: str = ""
    used_web: bool = False
    sources: list[Source] = Field(default_factory=list)
    guardrails: dict = Field(default_factory=dict)
    ragas: dict = Field(default_factory=dict)


class HistoryResponse(BaseModel):
    thread_id: str
    summary: str = ""
    messages: list[dict]


class HealthResponse(BaseModel):
    status: str
    vectors: int
    chat_model: str
    embedding_model: str
    guardrails: bool
    ragas: bool


def _save_upload(file: UploadFile) -> str:
    original = Path(file.filename or "upload")
    suffix = original.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_SUFFIXES)}",
        )
    dest = UPLOAD_DIR / f"{uuid4().hex}_{original.name}"
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return str(dest)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        vectors=collection_count(),
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        guardrails=settings.guardrails_enabled,
        ragas=settings.ragas_enabled,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_paths(body: IngestRequest) -> IngestResponse:
    if not body.paths:
        raise HTTPException(status_code=400, detail="Provide at least one path or URL.")
    try:
        indexed = await asyncio.to_thread(ingest, body.paths)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IngestResponse(
        indexed_chunks=indexed,
        collection_size=collection_count(),
        sources=body.paths,
    )


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_uploads(
    file: Annotated[
        UploadFile,
        File(description="Document to index (PDF, DOCX, PPTX, HTML, images, ...)"),
    ],
) -> IngestResponse:
    saved = _save_upload(file)
    try:
        indexed = await asyncio.to_thread(ingest, [saved])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Upload ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IngestResponse(
        indexed_chunks=indexed,
        collection_size=collection_count(),
        sources=[saved],
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    try:
        result = await asyncio.to_thread(run_turn, body.question, thread_id=body.thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(**result)


@app.get("/chat/{thread_id}", response_model=HistoryResponse)
def chat_history(thread_id: str) -> HistoryResponse:
    payload = thread_history(thread_id)
    return HistoryResponse(thread_id=thread_id, **payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7000, reload=True)
