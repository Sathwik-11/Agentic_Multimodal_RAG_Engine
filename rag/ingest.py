"""Load documents with Docling, chunk with HybridChunker, embed into Chroma."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import tiktoken
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from langchain_core.documents import Document
from langchain_docling.loader import DoclingLoader, ExportType

from rag.config import settings
from rag.retriever import reset_retriever
from rag.vectorstore import engine_lock, get_vectorstore

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".html",
    ".htm",
    ".md",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".xlsx",
    ".csv",
    ".adoc",
    ".asciidoc",
}


def _openai_tokenizer() -> OpenAITokenizer:
    try:
        encoding = tiktoken.encoding_for_model(settings.embedding_model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return OpenAITokenizer(
        tokenizer=encoding,
        max_tokens=settings.chunk_max_tokens,
    )


def build_chunker() -> HybridChunker:
    return HybridChunker(
        tokenizer=_openai_tokenizer(),
        merge_peers=True,
    )


def _flatten_metadata(doc: Document) -> Document:
    """Chroma only accepts scalar metadata values."""
    meta = dict(doc.metadata or {})
    flat: dict[str, str | int | float | bool] = {}

    source = meta.get("source")
    if source is not None:
        flat["source"] = str(source)

    dl_meta = meta.get("dl_meta") or {}
    if isinstance(dl_meta, str):
        try:
            dl_meta = json.loads(dl_meta)
        except json.JSONDecodeError:
            dl_meta = {}
    if not isinstance(dl_meta, dict):
        dl_meta = {}

    headings = dl_meta.get("headings") or []
    if headings:
        flat["headings"] = " > ".join(str(h) for h in headings)

    origin = dl_meta.get("origin") or {}
    if isinstance(origin, dict) and origin.get("filename"):
        flat["filename"] = str(origin["filename"])

    pages: list[int] = []
    for item in dl_meta.get("doc_items") or []:
        if not isinstance(item, dict):
            continue
        for prov in item.get("prov") or []:
            if isinstance(prov, dict) and "page_no" in prov:
                pages.append(int(prov["page_no"]))
    if pages:
        unique_pages = sorted(set(pages))
        flat["page"] = unique_pages[0]
        flat["pages"] = ",".join(str(p) for p in unique_pages)

    if "filename" not in flat and source:
        parsed = urlparse(str(source))
        name = Path(parsed.path).name or str(source)
        flat["filename"] = name

    return Document(page_content=doc.page_content, metadata=flat)


def _doc_id(doc: Document, index: int) -> str:
    payload = (
        f"{doc.metadata.get('source', '')}|"
        f"{doc.metadata.get('page', '')}|"
        f"{index}|"
        f"{doc.page_content[:400]}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_sources(paths: Iterable[str]) -> list[str]:
    sources: list[str] = []
    for raw in paths:
        if raw.startswith(("http://", "https://")):
            sources.append(raw)
            continue
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if path.is_file():
            sources.append(str(path))
            continue
        files = [
            str(p)
            for p in sorted(path.rglob("*"))
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        if not files:
            logger.warning("No supported documents found under %s", path)
        sources.extend(files)
    return sources


def load_chunks(sources: list[str]) -> list[Document]:
    if not sources:
        return []
    chunker = build_chunker()
    docs: list[Document] = []
    for source in sources:
        try:
            loader = DoclingLoader(
                file_path=source,
                export_type=ExportType.DOC_CHUNKS,
                chunker=chunker,
            )
            docs.extend(_flatten_metadata(doc) for doc in loader.load())
            logger.info("Chunked %s", source)
        except Exception:
            logger.exception("Failed to load %s", source)
    return [doc for doc in docs if doc.page_content and doc.page_content.strip()]


def ingest(paths: Iterable[str]) -> int:
    sources = resolve_sources(paths)
    if not sources:
        raise ValueError("No ingestible files or URLs were provided.")

    logger.info("Loading %d source(s) with Docling + HybridChunker", len(sources))
    chunks = load_chunks(sources)
    if not chunks:
        raise ValueError("Docling produced no chunks from the given sources.")

    ids = [_doc_id(doc, i) for i, doc in enumerate(chunks)]
    vectorstore = get_vectorstore()
    with engine_lock:
        vectorstore.add_documents(documents=chunks, ids=ids)
        reset_retriever()
    logger.info("Indexed %d chunks into Chroma (%s)", len(chunks), settings.chroma_dir)
    return len(chunks)
