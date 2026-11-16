"""Структуры данных системы коллективных дебатов."""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


class AgentConfig(BaseModel):
    """Описание одного агента-участника."""

    name: str
    model: str = Field(description="Спецификация модели, например 'ollama:qwen2.5:7b'")
    role: str = "analyst"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class DebateConfig(BaseModel):
    """Параметры протокола дебатов."""

    agents: list[AgentConfig]
    max_rounds: int = Field(default=2, ge=0, description="Число раундов обсуждения R")
    aggregation: Literal["majority", "weighted", "judge"] = "weighted"
    consensus_share: float = Field(default=1.0, gt=0.5, le=1.0)
    judge_model: str | None = None
    anonymize: bool = True

    @model_validator(mode="after")
    def _check(self) -> "DebateConfig":
        if len(self.agents) < 2:
            raise ValueError("Для дебатов требуется не менее двух агентов")
        names = [a.name for a in self.agents]
        if len(set(names)) != len(names):
            raise ValueError("Имена агентов должны быть уникальными")
        if self.aggregation == "judge" and not self.judge_model:
            raise ValueError("Для агрегации 'judge' необходимо указать judge_model")
        return self


class Reply(TypedDict):
    """Одно сообщение агента в протоколе."""

    round: int
    agent: str
    role: str
    model: str
    answer: str
    rationale: str
    confidence: float
    raw: str
    tokens_in: int
    tokens_out: int
    seconds: float


class DebateState(TypedDict, total=False):
    """Общее состояние графа LangGraph."""

    question: str
    round: int
    replies: Annotated[list[Reply], operator.add]
    final_answer: str
    final_rationale: str
    stop_reason: str
