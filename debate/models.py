"""Создание чат-моделей по строковой спецификации 'поставщик:имя'."""
from __future__ import annotations

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel

ModelFactory = Callable[[str, float], BaseChatModel]


def make_model(spec: str, temperature: float = 0.7) -> BaseChatModel:
    """Поддерживаются 'ollama:<модель>' и 'openai:<модель>' (совместимый API).

    Интеграционные пакеты импортируются только при обращении,
    поэтому установка обоих не обязательна.
    """
    provider, _, name = spec.partition(":")
    if not name:
        raise ValueError(f"Ожидается спецификация вида 'поставщик:модель', получено: {spec!r}")
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=name, temperature=temperature)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=name, temperature=temperature)
    raise ValueError(f"Неизвестный поставщик моделей: {provider!r}")
