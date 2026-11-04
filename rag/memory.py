"""Thread memory: last N exchanges stay verbatim; older turns fold into a summary."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from rag.config import settings

logger = logging.getLogger(__name__)

try:
    from langchain_core.messages import RemoveMessage
except ImportError:  # pragma: no cover
    RemoveMessage = None  # type: ignore[misc, assignment]


def _text(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content or "")


def _is_human(message: BaseMessage) -> bool:
    return isinstance(message, HumanMessage) or getattr(message, "type", "") == "human"


def _is_chat_ai(message: BaseMessage) -> bool:
    if not (isinstance(message, AIMessage) or getattr(message, "type", "") == "ai"):
        return False
    if getattr(message, "tool_calls", None):
        return False
    return bool(_text(message).strip())


def chat_turns(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Human and final assistant messages only (no tool calls)."""
    turns: list[BaseMessage] = []
    for message in messages:
        if _is_human(message) or _is_chat_ai(message):
            if _text(message).strip():
                turns.append(message)
    return turns


def split_exchanges(
    turns: list[BaseMessage],
) -> tuple[list[list[BaseMessage]], list[BaseMessage]]:
    """Complete human/assistant pairs, then any trailing unpaired messages."""
    pairs: list[list[BaseMessage]] = []
    i = 0
    while i < len(turns):
        if _is_human(turns[i]) and i + 1 < len(turns) and _is_chat_ai(turns[i + 1]):
            pairs.append([turns[i], turns[i + 1]])
            i += 2
            continue
        break
    return pairs, turns[i:]


def _format_turns(groups: list[list[BaseMessage]]) -> str:
    lines: list[str] = []
    for human, assistant in groups:
        lines.append(f"USER: {_text(human)}")
        lines.append(f"ASSISTANT: {_text(assistant)}")
    return "\n".join(lines)


def conversation_text(state: dict[str, Any]) -> str:
    """Prompt-ready conversation: running summary + last N exchanges + current question."""
    parts: list[str] = []
    summary = (state.get("conversation_summary") or "").strip()
    if summary:
        parts.append(f"Earlier conversation summary:\n{summary}")
    for message in llm_history(state):
        if isinstance(message, SystemMessage):
            continue
        role = "USER" if _is_human(message) else "ASSISTANT"
        text = _text(message).strip()
        if text:
            parts.append(f"{role}: {text}")
    return "\n".join(parts) if parts else "(empty)"


def llm_history(state: dict[str, Any]) -> list[BaseMessage]:
    """Messages the models should see this turn."""
    messages = list(state.get("messages") or [])
    summary = (state.get("conversation_summary") or "").strip()
    turns = chat_turns(messages)
    pairs, trailing = split_exchanges(turns)
    keep = pairs[-settings.max_history_exchanges :] + [trailing]
    recent = [item for group in keep for item in group]
    history: list[BaseMessage] = []
    if summary:
        history.append(
            SystemMessage(
                content="Earlier conversation summary (older turns already folded in):\n"
                + summary
            )
        )
    history.extend(recent)
    return history


def compact_conversation(state: dict[str, Any], summarize) -> dict[str, Any]:
    """Fold exchanges older than the last N into `conversation_summary`.

    Conversations live in the LangGraph checkpointer (`messages` per `thread_id`).
    This keeps that list short: last five Q&A pairs plus the current question.
    """
    messages = list(state.get("messages") or [])
    turns = chat_turns(messages)
    pairs, trailing = split_exchanges(turns)
    keep_n = settings.max_history_exchanges
    if len(pairs) <= keep_n:
        return {}

    older, recent = pairs[:-keep_n], pairs[-keep_n:]
    previous = (state.get("conversation_summary") or "").strip() or "(none)"
    extra = _format_turns(older)
    try:
        raw = summarize(previous, extra)
        summary = raw.strip() if isinstance(raw, str) else str(raw).strip()
    except Exception:
        logger.exception("Conversation summarization failed; keeping older turns")
        return {}

    if not summary:
        return {}

    logger.info(
        "Folded %d older exchange(s) into the thread summary; keeping %d recent",
        len(older),
        len(recent),
    )
    update: dict[str, Any] = {"conversation_summary": summary}
    if RemoveMessage is None:
        return update

    keep_ids = {
        getattr(message, "id", None)
        for group in recent + [trailing]
        for message in group
    }
    removals = []
    for message in messages:
        message_id = getattr(message, "id", None)
        if not message_id or message_id in keep_ids:
            continue
        removals.append(RemoveMessage(id=message_id))
    if removals:
        update["messages"] = removals
    return update
