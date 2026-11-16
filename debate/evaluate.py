"""Экспериментальная оценка фактической точности методов."""
from __future__ import annotations

import csv
import json
import math
import random
from typing import Callable

from .parsing import normalize
from .schemas import DebateState
from .storage import Storage, costs

Method = Callable[[str], DebateState]


def load_dataset(path: str) -> list[dict]:
    """Читает JSONL-файл со строками вида {"question": ..., "answers": [...]}."""
    items = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if "answers" not in item:
                    item["answers"] = [item["answer"]]
                items.append(item)
    return items


def is_correct(predicted: str, references: list[str]) -> bool:
    """Ответ верен, если совпадает с эталоном либо содержит его как отдельные слова."""
    answer = f" {normalize(predicted)} "
    return any(f" {normalize(ref)} " in answer for ref in references if normalize(ref))


def bootstrap_ci(flags: list[bool], samples: int = 2000, level: float = 0.95,
                 seed: int = 0) -> tuple[float, float]:
    """Доверительный интервал точности методом бутстрепа."""
    if not flags:
        return 0.0, 0.0
    rng = random.Random(seed)
    size = len(flags)
    means = sorted(sum(rng.choices(flags, k=size)) / size for _ in range(samples))
    low = means[int((1 - level) / 2 * samples)]
    high = means[min(samples - 1, int((1 + level) / 2 * samples))]
    return low, high


def mcnemar_p(first: list[bool], second: list[bool]) -> float:
    """Точный двусторонний критерий Мак-Немара для парных результатов."""
    only_first = sum(a and not b for a, b in zip(first, second))
    only_second = sum(b and not a for a, b in zip(first, second))
    total = only_first + only_second
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(only_first, only_second) + 1)) / 2**total
    return min(1.0, 2 * tail)


def evaluate(dataset: list[dict], methods: dict[str, Method], storage: Storage | None = None,
             details_csv: str | None = None) -> dict[str, dict]:
    """Прогоняет все методы по набору вопросов и возвращает сводные показатели."""
    flags: dict[str, list[bool]] = {name: [] for name in methods}
    totals: dict[str, dict] = {
        name: {"calls": 0, "tokens": 0, "seconds": 0.0} for name in methods
    }
    rows = []
    for index, item in enumerate(dataset):
        for name, method in methods.items():
            state = method(item["question"])
            correct = is_correct(state["final_answer"], item["answers"])
            cost = costs(state)
            flags[name].append(correct)
            totals[name]["calls"] += cost["calls"]
            totals[name]["tokens"] += cost["tokens_in"] + cost["tokens_out"]
            totals[name]["seconds"] += cost["seconds"]
            if storage is not None:
                storage.save(state, name)
            rows.append({
                "index": index, "method": name, "question": item["question"],
                "predicted": state["final_answer"], "correct": int(correct),
                "stop_reason": state.get("stop_reason", ""), "rounds": state.get("round", 0),
                **cost,
            })
    if details_csv:
        with open(details_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["index"])
            writer.writeheader()
            writer.writerows(rows)
    count = max(1, len(dataset))
    summary = {}
    for name in methods:
        low, high = bootstrap_ci(flags[name])
        summary[name] = {
            "accuracy": sum(flags[name]) / count,
            "ci_low": low,
            "ci_high": high,
            "calls_per_question": totals[name]["calls"] / count,
            "tokens_per_question": totals[name]["tokens"] / count,
            "seconds_per_question": totals[name]["seconds"] / count,
            "flags": flags[name],
        }
    return summary
