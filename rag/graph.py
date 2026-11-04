"""LangGraph agentic retrieval + Corrective RAG with conversation memory."""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Literal
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rag.config import settings
from rag.evaluate import evaluate_turn
from rag.guardrails import REFUSAL, check_input, check_output
from rag.memory import compact_conversation, conversation_text, llm_history
from rag.prompts import (
    AGENT_SYSTEM,
    GENERATE_PROMPT,
    GRADE_PROMPT,
    REWRITE_PROMPT,
    SUMMARIZE_PROMPT,
)
from rag.retriever import retrieve_documents
from rag.state import RAGState, RetrievalAssessment
from rag.vectorstore import collection_count, engine_lock

logger = logging.getLogger(__name__)

_checkpointer_conn: sqlite3.Connection | None = None


def _chat(temperature: float = 0) -> ChatOpenAI:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
    )


@tool
def retrieve_knowledge(query: str) -> str:
    """Search ingested documents for passages relevant to a standalone query."""
    return query


def _last_human_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _summarize_older(previous: str, turns: str) -> str:
    prompt = SUMMARIZE_PROMPT.format(summary=previous, turns=turns)
    content = _chat().invoke(prompt).content
    return content if isinstance(content, str) else str(content)


def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "(no documents)"
    parts: list[str] = []
    for i, doc in enumerate(docs):
        filename = doc.metadata.get("filename") or doc.metadata.get("source") or "unknown"
        page = doc.metadata.get("page")
        headings = doc.metadata.get("headings")
        header = f"[{i}] {filename}"
        if page is not None:
            header += f", p.{page}"
        if headings:
            header += f" | {headings}"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _tool_call_query(message: BaseMessage) -> str | None:
    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
        if name != retrieve_knowledge.name:
            continue
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
        query = (args or {}).get("query")
        if query:
            return str(query)
    return None


def _tool_call_ids(message: BaseMessage) -> list[str]:
    ids: list[str] = []
    for call in getattr(message, "tool_calls", None) or []:
        call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", "")
        if call_id:
            ids.append(str(call_id))
    return ids


def prepare(state: RAGState) -> dict:
    question = _last_human_text(state["messages"])
    update = {
        "question": question,
        "search_query": question,
        "documents": [],
        "relevant_documents": [],
        "retrieval_grade": "",
        "retry_count": 0,
        "used_web": False,
        "answer": "",
        "guardrails_blocked": False,
        "guardrails": {},
        "ragas": {},
    }
    update.update(compact_conversation(state, _summarize_older))
    return update


def input_guard(state: RAGState) -> dict:
    decision = check_input(state.get("question") or "")
    if not decision.blocked:
        return {
            "guardrails_blocked": False,
            "guardrails": {"input": "pass"},
        }
    return {
        "answer": decision.message,
        "messages": [AIMessage(content=decision.message, id=str(uuid4()))],
        "guardrails_blocked": True,
        "guardrails": {"input": "blocked", "output": "skipped"},
        "ragas": {},
    }


def agent(state: RAGState) -> dict:
    llm = _chat().bind_tools([retrieve_knowledge])
    history = llm_history(state)
    response = llm.invoke([("system", AGENT_SYSTEM), *history])
    query = _tool_call_query(response)
    if query:
        return {"messages": [response], "search_query": query}
    content = response.content if isinstance(response.content, str) else str(response.content)
    return {"answer": content}


def retrieve(state: RAGState) -> dict:
    if collection_count() == 0:
        raise RuntimeError(
            "The Chroma collection is empty. Run `python main.py ingest <path>` first."
        )

    query = state.get("search_query") or state.get("question") or ""
    last = state["messages"][-1] if state["messages"] else None
    if last is not None:
        query = _tool_call_query(last) or query

    docs = retrieve_documents(query)
    if int(state.get("retry_count") or 0) > 0:
        prior = state.get("relevant_documents") or []
        seen = {doc.page_content for doc in docs}
        docs = docs + [doc for doc in prior if doc.page_content not in seen]
    logger.info("Retrieved %d chunk(s) for query: %s", len(docs), query)

    messages: list[BaseMessage] = []
    if last is not None and getattr(last, "tool_calls", None):
        preview = _format_docs(docs)
        for call_id in _tool_call_ids(last) or [f"retrieve-{uuid4().hex[:8]}"]:
            messages.append(
                ToolMessage(
                    content=preview[:12000],
                    tool_call_id=call_id,
                    id=str(uuid4()),
                )
            )

    return {
        "documents": docs,
        "search_query": query,
        "messages": messages,
    }


def grade_documents(state: RAGState) -> dict:
    docs = state.get("documents") or []
    if not docs:
        return {
            "relevant_documents": [],
            "retrieval_grade": "incorrect",
        }

    numbered = "\n\n".join(
        f"[{i}]\n{doc.page_content[:2000]}" for i, doc in enumerate(docs)
    )
    prompt = GRADE_PROMPT.format(
        question=state.get("question", ""),
        search_query=state.get("search_query", ""),
        documents=numbered,
    )
    assessment = (
        _chat()
        .with_structured_output(RetrievalAssessment)
        .invoke(prompt)
    )
    indices = sorted(
        {i for i in assessment.relevant_indices if 0 <= i < len(docs)}
    )
    relevant = [docs[i] for i in indices] if indices else []
    if assessment.overall == "correct" and not relevant:
        relevant = docs
    logger.info(
        "CRAG grade=%s relevant=%d/%d (%s)",
        assessment.overall,
        len(relevant),
        len(docs),
        assessment.reasoning,
    )
    return {
        "relevant_documents": relevant,
        "retrieval_grade": assessment.overall,
        "documents": relevant or docs,
    }


def rewrite_query(state: RAGState) -> dict:
    prompt = REWRITE_PROMPT.format(
        question=state.get("question", ""),
        search_query=state.get("search_query", ""),
        conversation=conversation_text(state),
        reason=state.get("retrieval_grade", "incorrect"),
    )
    rewritten = _chat().invoke(prompt).content
    query = rewritten.strip() if isinstance(rewritten, str) else str(rewritten)
    retry = int(state.get("retry_count") or 0) + 1
    logger.info("Rewrote query (retry %d): %s", retry, query)
    return {"search_query": query, "retry_count": retry}


def web_search(state: RAGState) -> dict:
    query = state.get("search_query") or state.get("question") or ""
    docs = _run_web_search(query)
    merged = list(state.get("relevant_documents") or []) + docs
    logger.info("Web search returned %d document(s)", len(docs))
    return {
        "documents": merged,
        "relevant_documents": merged,
        "used_web": True,
    }


def generate(state: RAGState) -> dict:
    """LLM writes the user-facing answer from the graded retrieved chunks."""
    context_docs = state.get("relevant_documents") or state.get("documents") or []
    prompt = GENERATE_PROMPT.format(
        conversation=conversation_text(state),
        question=state.get("question", ""),
        context=_format_docs(context_docs),
    )
    logger.info("Generating answer from %d retrieved chunk(s)", len(context_docs))
    response = _chat(temperature=0).invoke(prompt)
    answer = response.content if isinstance(response.content, str) else str(response.content)
    return {"answer": answer}


def output_guard(state: RAGState) -> dict:
    answer = state.get("answer") or ""
    docs = state.get("relevant_documents") or state.get("documents") or []
    contexts = [doc.page_content for doc in docs if doc.page_content]
    decision = check_output(state.get("question") or "", answer, contexts)
    payload = dict(state.get("guardrails") or {})
    if decision.blocked:
        payload["output"] = "blocked"
        final = decision.message
        blocked = True
    else:
        payload["output"] = "pass"
        final = answer
        blocked = bool(state.get("guardrails_blocked"))
    return {
        "answer": final,
        "messages": [AIMessage(content=final, id=str(uuid4()))] if final else [],
        "guardrails": payload,
        "guardrails_blocked": blocked,
    }


def evaluate(state: RAGState) -> dict:
    if state.get("guardrails_blocked"):
        return {"ragas": {}}
    docs = state.get("relevant_documents") or state.get("documents") or []
    contexts = [doc.page_content for doc in docs if doc.page_content]
    scores = evaluate_turn(state.get("question") or "", state.get("answer") or "", contexts)
    update: dict = {"ragas": scores}
    threshold = settings.ragas_min_faithfulness
    faithfulness = scores.get("faithfulness")
    if (
        threshold > 0
        and isinstance(faithfulness, (int, float))
        and faithfulness < threshold
    ):
        logger.info(
            "Faithfulness %.3f below threshold %.3f; replacing the answer",
            faithfulness,
            threshold,
        )
        update["answer"] = REFUSAL
        update["messages"] = [AIMessage(content=REFUSAL, id=str(uuid4()))]
        payload = dict(state.get("guardrails") or {})
        payload["ragas"] = "blocked"
        update["guardrails"] = payload
    return update


def route_input_guard(state: RAGState) -> Literal["agent", "end"]:
    if state.get("guardrails_blocked"):
        return "end"
    return "agent"


def route_agent(state: RAGState) -> Literal["retrieve", "output_guard"]:
    last = state["messages"][-1] if state["messages"] else None
    if last is not None and getattr(last, "tool_calls", None):
        return "retrieve"
    return "output_guard"


def route_after_grade(
    state: RAGState,
) -> Literal["generate", "rewrite", "web_search"]:
    grade = state.get("retrieval_grade") or "incorrect"
    retries = int(state.get("retry_count") or 0)
    relevant = state.get("relevant_documents") or []

    if grade == "correct" and relevant:
        return "generate"
    if retries < settings.max_retries:
        return "rewrite"
    if settings.web_search_enabled and grade == "incorrect":
        return "web_search"
    return "generate"


def _run_web_search(query: str) -> list[Document]:
    if not settings.web_search_enabled:
        return []
    try:
        from langchain_tavily import TavilySearch

        raw = TavilySearch(max_results=3).invoke({"query": query})
    except Exception as exc:
        logger.warning("Web search unavailable: %s", exc)
        return []

    results = raw
    if isinstance(raw, str):
        try:
            results = json.loads(raw)
        except json.JSONDecodeError:
            return [Document(page_content=raw, metadata={"source": "web", "filename": "web"})]
    if isinstance(results, dict):
        results = results.get("results") or results.get("content") or [results]
    if not isinstance(results, list):
        return []

    docs: list[Document] = []
    for item in results:
        if isinstance(item, str):
            docs.append(
                Document(page_content=item, metadata={"source": "web", "filename": "web"})
            )
            continue
        if not isinstance(item, dict):
            continue
        content = item.get("content") or item.get("snippet") or ""
        url = item.get("url") or "web"
        if content:
            docs.append(
                Document(
                    page_content=str(content),
                    metadata={"source": str(url), "filename": str(url)},
                )
            )
    return docs


def _build_checkpointer():
    global _checkpointer_conn
    path = settings.checkpoint_path
    if not path:
        return MemorySaver()
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        logger.warning("langgraph-checkpoint-sqlite is not installed; using in-memory state.")
        return MemorySaver()

    from pathlib import Path

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _checkpointer_conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(_checkpointer_conn)
    saver.setup()
    return saver


def build_graph():
    workflow = StateGraph(RAGState)
    workflow.add_node("prepare", prepare)
    workflow.add_node("input_guard", input_guard)
    workflow.add_node("agent", agent)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("output_guard", output_guard)
    workflow.add_node("evaluate", evaluate)

    workflow.add_edge(START, "prepare")
    workflow.add_edge("prepare", "input_guard")
    workflow.add_conditional_edges(
        "input_guard",
        route_input_guard,
        {"agent": "agent", "end": END},
    )
    workflow.add_conditional_edges(
        "agent",
        route_agent,
        {"retrieve": "retrieve", "output_guard": "output_guard"},
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
            "web_search": "web_search",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "output_guard")
    workflow.add_edge("output_guard", "evaluate")
    workflow.add_edge("evaluate", END)
    return workflow.compile(checkpointer=_build_checkpointer())


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _source_payload(doc: Document) -> dict:
    return {
        "content": doc.page_content,
        "filename": doc.metadata.get("filename"),
        "source": doc.metadata.get("source"),
        "page": doc.metadata.get("page"),
        "headings": doc.metadata.get("headings"),
        "bbox": doc.metadata.get("bbox"),
    }


def run_turn(question: str, *, thread_id: str = "default") -> dict:
    """Run one conversational turn and return the answer plus retrieval metadata."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    payload = {"messages": [HumanMessage(content=question, id=str(uuid4()))]}
    with engine_lock:
        result = graph.invoke(payload, config=config)

    docs = result.get("relevant_documents") or result.get("documents") or []
    return {
        "answer": result.get("answer") or _last_ai_text(result.get("messages") or []),
        "thread_id": thread_id,
        "search_query": result.get("search_query") or question,
        "retrieval_grade": result.get("retrieval_grade") or "",
        "used_web": bool(result.get("used_web")),
        "sources": [_source_payload(doc) for doc in docs],
        "guardrails": result.get("guardrails") or {},
        "ragas": result.get("ragas") or {},
    }


def thread_history(thread_id: str) -> dict:
    snapshot = get_graph().get_state({"configurable": {"thread_id": thread_id}})
    values = snapshot.values if snapshot else {}
    history: list[dict] = []
    for message in values.get("messages") or []:
        role = getattr(message, "type", "")
        if role not in {"human", "ai"} or getattr(message, "tool_calls", None):
            continue
        content = message.content
        text = content if isinstance(content, str) else str(content)
        if not text:
            continue
        history.append(
            {
                "role": "user" if role == "human" else "assistant",
                "content": text,
            }
        )
    return {
        "summary": values.get("conversation_summary") or "",
        "messages": history,
    }


def ask(
    question: str,
    *,
    thread_id: str = "default",
    verbose: bool = False,
) -> str:
    if verbose:
        logger.info("Asking on thread %s: %s", thread_id, question)
    return run_turn(question, thread_id=thread_id)["answer"]


def _last_ai_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) or getattr(message, "type", "") == "ai":
            if getattr(message, "tool_calls", None):
                continue
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""
