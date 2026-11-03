"""NeMo Guardrails input/output checks around the LangGraph turn."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from rag.config import settings

logger = logging.getLogger(__name__)

REFUSAL = "I can only help with questions about the ingested documents. I can't respond to that."

_REFUSAL_MARKERS = (
    "i can't respond to that",
    "i cannot respond to that",
    "i'm sorry, i can't",
    "i am sorry, i can't",
    "i can't help with that",
    "i cannot help with that",
)


@dataclass(frozen=True)
class GuardDecision:
    blocked: bool
    message: str
    rail: str


@lru_cache(maxsize=1)
def _rails():
    try:
        from nemoguardrails import LLMRails, RailsConfig
    except ImportError as exc:
        raise RuntimeError(
            "nemoguardrails is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    config_dir = settings.guardrails_dir
    if not (config_dir / "config.yml").exists():
        raise RuntimeError(f"NeMo Guardrails config not found at {config_dir / 'config.yml'}")

    from langchain_openai import ChatOpenAI

    logger.info("Loading NeMo Guardrails from %s", config_dir)
    llm = ChatOpenAI(
        model=settings.chat_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )
    return LLMRails(RailsConfig.from_path(str(config_dir)), llm=llm, verbose=False)


def _response_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        content = result.get("content") or result.get("response") or ""
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                return str(first.get("content") or "").strip()
            return str(first).strip()
        return str(content).strip()
    response = getattr(result, "response", None)
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, list) and response:
        first = response[0]
        if isinstance(first, dict):
            return str(first.get("content") or "").strip()
        return str(first).strip()
    return str(result).strip()


def _stopped(result: Any) -> bool:
    log = getattr(result, "log", None)
    rails = getattr(log, "activated_rails", None) if log is not None else None
    if not rails:
        return False
    for rail in rails:
        if getattr(rail, "stop", False) or getattr(rail, "aborted", False):
            return True
        decisions = getattr(rail, "decisions", None) or []
        if any(str(item).lower() in {"stop", "abort", "refuse"} for item in decisions):
            return True
    return False


def _is_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def check_input(text: str) -> GuardDecision:
    """Run input rails. On infrastructure failure, allow the request."""
    if not settings.guardrails_enabled:
        return GuardDecision(blocked=False, message=text, rail="input")
    try:
        result = _rails().generate(
            messages=[{"role": "user", "content": text}],
            options={
                "rails": {
                    "input": True,
                    "dialog": False,
                    "retrieval": False,
                    "output": False,
                },
                "log": {"activated_rails": True},
            },
        )
        reply = _response_text(result)
        blocked = _stopped(result) or _is_refusal(reply)
        if blocked:
            logger.info("Input rail blocked the request")
            return GuardDecision(blocked=True, message=REFUSAL, rail="input")
        return GuardDecision(blocked=False, message=text, rail="input")
    except Exception:
        logger.exception("Input guardrail failed; allowing the request")
        return GuardDecision(blocked=False, message=text, rail="input")


def check_output(question: str, answer: str, contexts: list[str]) -> GuardDecision:
    """Run output (+ optional fact) rails on an already generated answer."""
    if not settings.guardrails_enabled or not answer:
        return GuardDecision(blocked=False, message=answer, rail="output")
    try:
        messages: list[dict] = []
        if contexts:
            messages.append(
                {
                    "role": "context",
                    "content": {
                        "relevant_chunks": "\n\n".join(contexts),
                        "evidence": "\n\n".join(contexts),
                        "check_facts": True,
                    },
                }
            )
        messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        result = _rails().generate(
            messages=messages,
            options={
                "rails": ["output"],
                "log": {"activated_rails": True},
            },
        )
        reply = _response_text(result) or answer
        blocked = _stopped(result) or _is_refusal(reply)
        if blocked:
            logger.info("Output rail blocked the response")
            return GuardDecision(blocked=True, message=REFUSAL, rail="output")
        return GuardDecision(blocked=False, message=answer, rail="output")
    except Exception:
        logger.exception("Output guardrail failed; allowing the response")
        return GuardDecision(blocked=False, message=answer, rail="output")
