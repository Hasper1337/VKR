"""Механизмы формирования итогового ответа и проверка согласия."""
from __future__ import annotations

from collections import defaultdict

from .parsing import normalize
from .schemas import Reply


def _scores(replies: list[Reply], weighted: bool) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for reply in replies:
        scores[normalize(reply["answer"])] += reply["confidence"] if weighted else 1.0
    return scores


def vote(replies: list[Reply], weighted: bool = False) -> Reply:
    """Голосование большинством либо взвешенное по уверенности.

    Возвращает сообщение агента с наибольшей уверенностью среди
    сторонников победившего ответа. При равенстве сумм побеждает ответ,
    у которого выше наибольшая уверенность сторонника.
    """
    if not replies:
        raise ValueError("Нет ответов для агрегации")
    scores = _scores(replies, weighted)

    def best_supporter(key: str) -> Reply:
        group = [r for r in replies if normalize(r["answer"]) == key]
        return max(group, key=lambda r: r["confidence"])

    winner = max(scores, key=lambda k: (scores[k], best_supporter(k)["confidence"]))
    return best_supporter(winner)


def agreement_share(replies: list[Reply]) -> float:
    """Доля агентов, поддержавших самый частый ответ раунда."""
    if not replies:
        return 0.0
    scores = _scores(replies, weighted=False)
    return max(scores.values()) / len(replies)
