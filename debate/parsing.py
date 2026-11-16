"""Разбор ответов моделей и нормализация ответов для сравнения."""
from __future__ import annotations

import json
import re

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize(answer: str) -> str:
    """Приводит ответ к виду, пригодному для сопоставления."""
    text = answer.lower().replace("ё", "е")
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def _extract_json(text: str) -> dict | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        for pos in range(start, len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : pos + 1])
                    except json.JSONDecodeError:
                        break
                    return value if isinstance(value, dict) else None
        start = text.find("{", start + 1)
    return None


def parse_reply(text: str) -> tuple[str, str, float]:
    """Возвращает (ответ, обоснование, уверенность).

    Если модель нарушила формат, весь текст считается ответом,
    а уверенность принимается равной 0,5.
    """
    data = _extract_json(text)
    if not data or "answer" not in data:
        return text.strip(), "", 0.5
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(1.0, max(0.0, confidence))
    return str(data["answer"]).strip(), str(data.get("rationale", "")).strip(), confidence
