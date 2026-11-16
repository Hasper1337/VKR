"""Интерфейс командной строки: одиночный вопрос и серия экспериментов."""
from __future__ import annotations

import argparse
import json
from functools import partial

from debate.baselines import run_cot, run_self_consistency, run_single
from debate.evaluate import evaluate, load_dataset, mcnemar_p
from debate.graph import run_debate
from debate.schemas import DebateConfig
from debate.storage import Storage, costs


def load_config(path: str) -> DebateConfig:
    with open(path, encoding="utf-8") as handle:
        return DebateConfig.model_validate(json.load(handle))


def command_ask(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    state = run_debate(args.question, config)
    for reply in state["replies"]:
        print(f"[раунд {reply['round']}] {reply['agent']} ({reply['role']}): "
              f"{reply['answer']} | уверенность {reply['confidence']:.2f}")
    print(f"\nИтоговый ответ: {state['final_answer']}")
    print(f"Причина останова: {state['stop_reason']}; затраты: {costs(state)}")
    storage = Storage(args.db)
    storage.save(state, "debate", config.model_dump_json())
    storage.close()


def command_eval(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    base = args.baseline_model or config.agents[0].model
    calls = len(config.agents) * (config.max_rounds + 1)
    methods = {
        "single": partial(run_single, spec=base),
        "cot": partial(run_cot, spec=base),
        "self_consistency": partial(run_self_consistency, spec=base, samples=calls),
        "debate": partial(run_debate, config=config),
    }
    storage = Storage(args.db)
    summary = evaluate(load_dataset(args.dataset), methods, storage, args.details)
    storage.close()
    for name, row in summary.items():
        print(f"{name:18s} точность {row['accuracy']:.3f} "
              f"[{row['ci_low']:.3f}; {row['ci_high']:.3f}]  "
              f"обращений {row['calls_per_question']:.1f}  "
              f"токенов {row['tokens_per_question']:.0f}  "
              f"секунд {row['seconds_per_question']:.1f}")
    for name in ("single", "cot", "self_consistency"):
        p_value = mcnemar_p(summary["debate"]["flags"], summary[name]["flags"])
        print(f"Критерий Мак-Немара, debate против {name}: p = {p_value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Система коллективных дебатов LLM")
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--db", default="debates.sqlite3")
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="Задать один вопрос")
    ask.add_argument("question")
    ask.set_defaults(handler=command_ask)
    run = commands.add_parser("eval", help="Оценить методы на наборе вопросов")
    run.add_argument("dataset", help="JSONL-файл с вопросами и эталонными ответами")
    run.add_argument("--baseline-model", default=None)
    run.add_argument("--details", default="results.csv")
    run.set_defaults(handler=command_eval)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
