"""Per-turn RAGAS scores for the generated answer and retrieved context.

RAGAS fits this LangGraph service better than DeepEval: it scores a live
(question, contexts, answer) triple without pytest test cases or a labeled
gold set. Faithfulness and answer relevancy are the metrics that do not need
a reference answer.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import types
from collections.abc import Coroutine
from typing import Any

from rag.config import settings

logger = logging.getLogger(__name__)


def _patch_ragas_langchain_import() -> None:
    """RAGAS still imports ChatVertexAI from langchain-community; 0.4+ moved it."""
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ImportError:
        module = types.ModuleType(name)

        class ChatVertexAI:  # pragma: no cover - import shim
            pass

        module.ChatVertexAI = ChatVertexAI
        sys.modules[name] = module


_patch_ragas_langchain_import()


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Already inside an event loop (unlikely for run_turn, but safe).
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _sample(question: str, answer: str, contexts: list[str]):
    try:
        from ragas import SingleTurnSample
    except ImportError:
        from ragas.dataset_schema import SingleTurnSample

    return SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )


def _ragas_llm():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.chat_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
    )


def _ragas_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from rag.vectorstore import get_embeddings

    return LangchainEmbeddingsWrapper(get_embeddings())


def _faithfulness_metric(llm):
    try:
        from ragas.metrics import Faithfulness

        return Faithfulness(llm=llm)
    except Exception:
        from ragas.metrics.collections import Faithfulness

        return Faithfulness(llm=llm)


def _relevancy_metric(llm, embeddings):
    try:
        from ragas.metrics import ResponseRelevancy

        return ResponseRelevancy(llm=llm, embeddings=embeddings)
    except Exception:
        try:
            from ragas.metrics import AnswerRelevancy

            return AnswerRelevancy(llm=llm, embeddings=embeddings)
        except Exception:
            from ragas.metrics.collections import AnswerRelevancy

            return AnswerRelevancy(llm=llm, embeddings=embeddings)


async def _score(metric, sample) -> float | None:
    if hasattr(metric, "single_turn_ascore"):
        value = await metric.single_turn_ascore(sample)
        return float(value) if value is not None else None
    if hasattr(metric, "ascore"):
        result = await metric.ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts,
        )
        value = getattr(result, "value", result)
        return float(value) if value is not None else None
    raise RuntimeError(f"Unsupported RAGAS metric API: {type(metric)!r}")


async def _aevaluate(question: str, answer: str, contexts: list[str]) -> dict[str, float | None]:
    llm = _ragas_llm()
    embeddings = _ragas_embeddings()
    sample = _sample(question, answer, contexts)
    faithfulness = _faithfulness_metric(llm)
    relevancy = _relevancy_metric(llm, embeddings)

    async def one(name: str, metric) -> tuple[str, float | None]:
        try:
            return name, await _score(metric, sample)
        except Exception:
            logger.exception("RAGAS metric %s failed", name)
            return name, None

    pairs = await asyncio.gather(
        one("faithfulness", faithfulness),
        one("answer_relevancy", relevancy),
    )
    return dict(pairs)


def evaluate_turn(question: str, answer: str, contexts: list[str]) -> dict[str, float | None]:
    """Return faithfulness and answer relevancy in [0, 1], or None on failure."""
    if not settings.ragas_enabled:
        return {}
    if not question or not answer or not contexts:
        return {}
    try:
        scores = _run(_aevaluate(question, answer, contexts))
    except Exception:
        logger.exception("RAGAS evaluation failed")
        return {"faithfulness": None, "answer_relevancy": None}

    logger.info(
        "RAGAS faithfulness=%s answer_relevancy=%s",
        scores.get("faithfulness"),
        scores.get("answer_relevancy"),
    )
    return scores
