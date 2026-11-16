"""Одноагентные базовые методы для сравнения с дебатами."""
from __future__ import annotations

from . import prompts
from .aggregation import vote
from .graph import call_model
from .models import ModelFactory, make_model
from .parsing import parse_reply
from .schemas import DebateState, Reply

DIRECT_SYSTEM = "Ответь на вопрос кратко и точно."
COT_SYSTEM = "Рассуждай шаг за шагом, затем дай краткий итоговый ответ."


def _samples(question: str, spec: str, system: str, count: int, temperature: float,
             factory: ModelFactory) -> list[Reply]:
    model = factory(spec, temperature)
    user = prompts.PROPOSE.format(question=question, format=prompts.FORMAT)
    replies: list[Reply] = []
    for index in range(count):
        text, tokens_in, tokens_out, seconds = call_model(model, system, user)
        answer, rationale, confidence = parse_reply(text)
        replies.append(Reply(
            round=0, agent=f"sample_{index + 1}", role="baseline", model=spec,
            answer=answer, rationale=rationale, confidence=confidence, raw=text,
            tokens_in=tokens_in, tokens_out=tokens_out, seconds=seconds,
        ))
    return replies


def _state(question: str, replies: list[Reply], reason: str) -> DebateState:
    winner = vote(replies, weighted=False)
    return DebateState(
        question=question, round=0, replies=replies, final_answer=winner["answer"],
        final_rationale=winner["rationale"], stop_reason=reason,
    )


def run_single(question: str, spec: str, factory: ModelFactory = make_model) -> DebateState:
    """Одиночный запрос к модели без рассуждения."""
    return _state(question, _samples(question, spec, DIRECT_SYSTEM, 1, 0.0, factory), "single")


def run_cot(question: str, spec: str, factory: ModelFactory = make_model) -> DebateState:
    """Одиночный запрос с цепочкой рассуждений."""
    return _state(question, _samples(question, spec, COT_SYSTEM, 1, 0.0, factory), "cot")


def run_self_consistency(question: str, spec: str, samples: int = 6, temperature: float = 0.7,
                         factory: ModelFactory = make_model) -> DebateState:
    """Самосогласованность: несколько независимых прогонов и голосование большинством."""
    return _state(question, _samples(question, spec, COT_SYSTEM, samples, temperature, factory),
                  "self_consistency")
