"""Hybrid BM25 + dense retrieval with FlashRank reranking."""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag.config import settings
from rag.vectorstore import get_all_documents, get_vectorstore

logger = logging.getLogger(__name__)


def _candidate_k() -> int:
    return max(settings.retrieve_candidate_k, settings.retrieve_k)


@lru_cache(maxsize=1)
def _bm25_retriever() -> BM25Retriever:
    docs = get_all_documents()
    if not docs:
        raise RuntimeError(
            "The Chroma collection is empty. Run `python main.py ingest <path>` first."
        )
    return BM25Retriever.from_documents(docs, k=_candidate_k())


@lru_cache(maxsize=1)
def _flashrank() -> FlashrankRerank:
    """FlashRank ONNX cross-encoder; models are hosted on Hugging Face Hub."""
    return FlashrankRerank(model=settings.flashrank_model, top_n=settings.retrieve_k)


@lru_cache(maxsize=1)
def get_retriever() -> ContextualCompressionRetriever:
    """BM25 keyword + Chroma semantic search, fused then reranked with FlashRank."""
    k = _candidate_k()
    bm25 = _bm25_retriever()
    bm25.k = k
    dense = get_vectorstore().as_retriever(search_kwargs={"k": k})
    hybrid = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[settings.bm25_weight, settings.dense_weight],
    )
    return ContextualCompressionRetriever(
        base_compressor=_flashrank(),
        base_retriever=hybrid,
    )


def reset_retriever() -> None:
    """Rebuild BM25 after new chunks are ingested."""
    _bm25_retriever.cache_clear()
    get_retriever.cache_clear()


def retrieve_documents(query: str) -> list[Document]:
    docs = list(get_retriever().invoke(query))
    logger.info("Hybrid retrieval returned %d chunk(s) for query: %s", len(docs), query)
    return docs
