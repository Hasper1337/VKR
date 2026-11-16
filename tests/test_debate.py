"""Модульные тесты протокола, агрегации, хранилища и оценки.

Вместо языковых моделей используется имитатор с заранее заданными ответами,
поэтому тесты не требуют сети и установленных моделей.
"""
from __future__ import annotations

import json
import sqlite3

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from debate.aggregation import agreement_share, vote
from debate.baselines import run_self_consistency, run_single
from debate.evaluate import bootstrap_ci, evaluate, is_correct, mcnemar_p
from debate.graph import run_debate
from debate.parsing import normalize, parse_reply
from debate.schemas import AgentConfig, DebateConfig
from debate.storage import Storage, costs


def reply(answer: str, confidence: float = 0.8) -> str:
    return json.dumps({"answer": answer, "rationale": "довод", "confidence": confidence},
                      ensure_ascii=False)


def scripted(script: dict[str, list[str]]):
    """Фабрика моделей: каждой спецификации соответствует свой список ответов."""
    counters: dict[str, int] = {}

    def factory(spec: str, temperature: float):
        counters[spec] = counters.get(spec, 0) + 1
        return FakeListChatModel(responses=script[spec])

    factory.counters = counters
    return factory


def config(**changes) -> DebateConfig:
    base = dict(
        agents=[
            AgentConfig(name="a", model="m:a", role="analyst"),
            AgentConfig(name="b", model="m:b", role="skeptic"),
            AgentConfig(name="c", model="m:c", role="fact_checker"),
        ],
        max_rounds=2,
        aggregation="weighted",
    )
    base.update(changes)
    return DebateConfig(**base)


def msg(answer: str, confidence: float, agent: str = "x") -> dict:
    return dict(round=0, agent=agent, role="r", model="m", answer=answer, rationale="",
                confidence=confidence, raw="", tokens_in=0, tokens_out=0, seconds=0.0)


def test_parse_valid_json_with_surrounding_text():
    assert parse_reply('Ответ: {"answer": " Париж ", "rationale": "столица", "confidence": 0.9}') == (
        "Париж", "столица", 0.9)


def test_parse_broken_format_falls_back_to_text():
    assert parse_reply("просто Париж") == ("просто Париж", "", 0.5)


def test_parse_clamps_confidence():
    assert parse_reply('{"answer": "x", "confidence": 7}')[2] == 1.0
    assert parse_reply('{"answer": "x", "confidence": "высокая"}')[2] == 0.5


def test_normalize():
    assert normalize("  Пётр  I. ") == "петр i"


def test_majority_and_weighted_vote_differ():
    replies = [msg("A", 0.3, "1"), msg("A", 0.3, "2"), msg("B", 0.9, "3")]
    assert vote(replies, weighted=False)["answer"] == "A"
    assert vote(replies, weighted=True)["answer"] == "B"


def test_agreement_share_ignores_case_and_punctuation():
    assert agreement_share([msg("Париж", 1), msg("париж.", 1), msg("Лион", 1)]) == pytest.approx(2 / 3)


def test_consensus_in_first_round_stops_without_debate():
    factory = scripted({"m:a": [reply("Париж")], "m:b": [reply("париж")], "m:c": [reply("Париж.")]})
    state = run_debate("Столица Франции?", config(), factory)
    assert state["stop_reason"] == "consensus"
    assert state["round"] == 0
    assert len(state["replies"]) == 3
    assert normalize(state["final_answer"]) == "париж"


def test_debate_converges_after_one_round():
    factory = scripted({
        "m:a": [reply("1961"), reply("1961")],
        "m:b": [reply("1957"), reply("1961")],
        "m:c": [reply("1961"), reply("1961")],
    })
    state = run_debate("Год первого полёта человека в космос?", config(), factory)
    assert state["stop_reason"] == "consensus"
    assert state["round"] == 1
    assert len(state["replies"]) == 6
    assert state["final_answer"] == "1961"


def test_round_limit_uses_weighted_vote():
    factory = scripted({
        "m:a": [reply("A", 0.4)], "m:b": [reply("A", 0.4)], "m:c": [reply("B", 0.95)],
    })
    state = run_debate("?", config(max_rounds=2), factory)
    assert state["stop_reason"] == "round_limit"
    assert state["round"] == 2
    assert len(state["replies"]) == 9
    assert state["final_answer"] == "B"


def test_round_limit_majority_vote():
    factory = scripted({
        "m:a": [reply("A", 0.4)], "m:b": [reply("A", 0.4)], "m:c": [reply("B", 0.95)],
    })
    state = run_debate("?", config(max_rounds=1, aggregation="majority"), factory)
    assert state["final_answer"] == "A"
    assert len(state["replies"]) == 6


def test_zero_rounds_is_plain_ensemble():
    factory = scripted({"m:a": [reply("A")], "m:b": [reply("B")], "m:c": [reply("A")]})
    state = run_debate("?", config(max_rounds=0, aggregation="majority"), factory)
    assert state["round"] == 0 and state["final_answer"] == "A"


def test_judge_decides_when_no_consensus():
    factory = scripted({
        "m:a": [reply("A")], "m:b": [reply("B")], "m:c": [reply("C")],
        "m:j": ['{"answer": "B", "rationale": "довод сильнее"}'],
    })
    state = run_debate("?", config(max_rounds=1, aggregation="judge", judge_model="m:j"), factory)
    assert state["final_answer"] == "B"
    assert state["replies"][-1]["agent"] == "judge"
    assert len(state["replies"]) == 7


def test_judge_not_called_on_consensus():
    factory = scripted({"m:a": [reply("A")], "m:b": [reply("A")], "m:c": [reply("A")], "m:j": ["x"]})
    state = run_debate("?", config(aggregation="judge", judge_model="m:j"), factory)
    assert all(r["agent"] != "judge" for r in state["replies"])


def test_config_validation():
    with pytest.raises(ValueError):
        DebateConfig(agents=[AgentConfig(name="a", model="m:a")])
    with pytest.raises(ValueError):
        config(aggregation="judge")
    with pytest.raises(ValueError):
        DebateConfig(agents=[AgentConfig(name="a", model="m:a"), AgentConfig(name="a", model="m:b")])


def test_storage_roundtrip(tmp_path):
    factory = scripted({"m:a": [reply("A")], "m:b": [reply("A")], "m:c": [reply("A")]})
    state = run_debate("?", config(), factory)
    path = tmp_path / "log.sqlite3"
    storage = Storage(str(path))
    session_id = storage.save(state, "debate", "{}")
    storage.close()
    connection = sqlite3.connect(path)
    assert connection.execute("SELECT calls, final_answer FROM sessions WHERE id = ?",
                              (session_id,)).fetchone() == (3, "A")
    assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3
    assert costs(state)["calls"] == 3


def test_baselines():
    factory = scripted({"m:a": [reply("A"), reply("B"), reply("A")]})
    assert run_single("?", "m:a", factory)["final_answer"] == "A"
    state = run_self_consistency("?", "m:a", samples=3, factory=factory)
    assert state["final_answer"] == "A" and len(state["replies"]) == 3


def test_is_correct():
    assert is_correct("Юрий Гагарин", ["Гагарин"])
    assert is_correct("1961 год", ["1961"])
    assert not is_correct("11961", ["1961"])
    assert not is_correct("Титов", ["Гагарин"])


def test_statistics():
    assert mcnemar_p([True] * 10, [True] * 10) == 1.0
    assert mcnemar_p([True] * 10, [False] * 10) == pytest.approx(2 / 2**10)
    low, high = bootstrap_ci([True] * 50 + [False] * 50)
    assert low < 0.5 < high


def test_evaluate_summary(tmp_path):
    dataset = [{"question": "q1", "answers": ["A"]}, {"question": "q2", "answers": ["B"]}]
    factory = scripted({"m:a": [reply("A")], "m:b": [reply("A")], "m:c": [reply("A")]})
    methods = {
        "single": lambda q: run_single(q, "m:a", factory),
        "debate": lambda q: run_debate(q, config(), factory),
    }
    details = tmp_path / "details.csv"
    summary = evaluate(dataset, methods, details_csv=str(details))
    assert summary["single"]["accuracy"] == 0.5
    assert summary["debate"]["calls_per_question"] == 3
    assert len(details.read_text(encoding="utf-8").splitlines()) == 5
