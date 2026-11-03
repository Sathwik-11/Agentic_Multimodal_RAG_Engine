"""LangGraph state and structured outputs for agentic CRAG."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RAGState(TypedDict):
    """Shared graph state for one conversation thread.

    `messages` is the durable chat history (checkpointer + add_messages).
    The remaining fields are per-turn working memory for retrieval/CRAG.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    search_query: str
    documents: list[Document]
    relevant_documents: list[Document]
    retrieval_grade: Literal["correct", "incorrect", "ambiguous", ""]
    retry_count: int
    used_web: bool
    answer: str
    guardrails_blocked: bool
    guardrails: dict
    ragas: dict


class RetrievalAssessment(BaseModel):
    """CRAG-style relevance judgment over a retrieved set."""

    relevant_indices: list[int] = Field(
        default_factory=list,
        description="0-based indices of documents that help answer the question.",
    )
    overall: Literal["correct", "incorrect", "ambiguous"] = Field(
        description=(
            "correct: enough relevant context to answer; "
            "ambiguous: partially related but incomplete; "
            "incorrect: none of the documents are useful."
        )
    )
    reasoning: str = Field(default="", description="Short justification.")
