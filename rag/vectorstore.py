"""ChromaDB vector store backed by OpenAI embeddings."""

from __future__ import annotations

import threading
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from rag.config import settings

engine_lock = threading.Lock()


def get_embeddings() -> OpenAIEmbeddings:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def collection_count() -> int:
    return get_vectorstore()._collection.count()


def get_all_documents() -> list[Document]:
    """Load every stored chunk so lexical retrievers can index the same corpus."""
    raw = get_vectorstore().get(include=["documents", "metadatas"])
    contents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    docs: list[Document] = []
    for i, content in enumerate(contents):
        if not content:
            continue
        metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        docs.append(Document(page_content=content, metadata=dict(metadata)))
    return docs
