"""Граф состояний LangGraph, реализующий протокол дебатов."""
from __future__ import annotations

import time

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from . import prompts
from .aggregation import agreement_share, vote
from .models import ModelFactory, make_model
from .parsing import parse_reply
from .schemas import AgentConfig, DebateConfig, DebateState, Reply


def call_model(model: BaseChatModel, system: str, user: str) -> tuple[str, int, int, float]:
    """Одно обращение к модели: (текст, входные токены, выходные токены, секунды)."""
    started = time.perf_counter()
    message = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    seconds = time.perf_counter() - started
    usage = getattr(message, "usage_metadata", None) or {}
    text = message.content if isinstance(message.content, str) else str(message.content)
    return text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)), seconds


def last_round(replies: list[Reply]) -> list[Reply]:
    """Сообщения последнего завершённого раунда."""
    if not replies:
        return []
    number = max(r["round"] for r in replies)
    return [r for r in replies if r["round"] == number]


def build_graph(config: DebateConfig, factory: ModelFactory = make_model):
    """Собирает и компилирует граф дебатов для заданной конфигурации."""
    models = {a.name: factory(a.model, a.temperature) for a in config.agents}
    judge = factory(config.judge_model, 0.0) if config.aggregation == "judge" else None

    def ask(agent: AgentConfig, number: int, user: str) -> Reply:
        text, tokens_in, tokens_out, seconds = call_model(
            models[agent.name], prompts.system_prompt(agent.role), user
        )
        answer, rationale, confidence = parse_reply(text)
        return Reply(
            round=number, agent=agent.name, role=agent.role, model=agent.model,
            answer=answer, rationale=rationale, confidence=confidence, raw=text,
            tokens_in=tokens_in, tokens_out=tokens_out, seconds=seconds,
        )

    def propose(state: DebateState) -> dict:
        user = prompts.PROPOSE.format(question=state["question"], format=prompts.FORMAT)
        return {"round": 0, "replies": [ask(a, 0, user) for a in config.agents]}

    def debate(state: DebateState) -> dict:
        number = state["round"] + 1
        previous = {r["agent"]: r for r in last_round(state["replies"])}
        new: list[Reply] = []
        for agent in config.agents:
            others = [r for name, r in previous.items() if name != agent.name]
            lines = []
            for index, other in enumerate(others, start=1):
                label = f"Участник {index}" if config.anonymize else other["agent"]
                lines.append(
                    f"{label}: ответ: {other['answer']}; обоснование: {other['rationale']}; "
                    f"уверенность: {other['confidence']:.2f}"
                )
            own = previous[agent.name]
            user = prompts.DEBATE.format(
                question=state["question"],
                own=f"{own['answer']} (обоснование: {own['rationale']})",
                others="\n".join(lines),
                format=prompts.FORMAT,
            )
            new.append(ask(agent, number, user))
        return {"round": number, "replies": new}

    def route(state: DebateState) -> str:
        if agreement_share(last_round(state["replies"])) >= config.consensus_share:
            return "aggregate"
        if state["round"] >= config.max_rounds:
            return "aggregate"
        return "debate"

    def aggregate(state: DebateState) -> dict:
        current = last_round(state["replies"])
        consensus = agreement_share(current) >= config.consensus_share
        reason = "consensus" if consensus else "round_limit"
        if judge is None or consensus:
            winner = vote(current, weighted=config.aggregation != "majority")
            return {
                "final_answer": winner["answer"],
                "final_rationale": winner["rationale"],
                "stop_reason": reason,
            }
        positions = "\n".join(
            f"Участник {i}: ответ: {r['answer']}; обоснование: {r['rationale']}"
            for i, r in enumerate(current, start=1)
        )
        user = prompts.JUDGE.format(question=state["question"], positions=positions)
        text, tokens_in, tokens_out, seconds = call_model(judge, prompts.JUDGE_SYSTEM, user)
        answer, rationale, _ = parse_reply(text)
        verdict = Reply(
            round=state["round"] + 1, agent="judge", role="judge", model=config.judge_model or "",
            answer=answer, rationale=rationale, confidence=1.0, raw=text,
            tokens_in=tokens_in, tokens_out=tokens_out, seconds=seconds,
        )
        return {
            "replies": [verdict],
            "final_answer": answer,
            "final_rationale": rationale,
            "stop_reason": reason,
        }

    builder = StateGraph(DebateState)
    builder.add_node("propose", propose)
    builder.add_node("debate", debate)
    builder.add_node("aggregate", aggregate)
    builder.add_edge(START, "propose")
    builder.add_conditional_edges("propose", route, {"debate": "debate", "aggregate": "aggregate"})
    builder.add_conditional_edges("debate", route, {"debate": "debate", "aggregate": "aggregate"})
    builder.add_edge("aggregate", END)
    return builder.compile()


def run_debate(question: str, config: DebateConfig, factory: ModelFactory = make_model) -> DebateState:
    """Проводит один сеанс дебатов и возвращает конечное состояние."""
    graph = build_graph(config, factory)
    limit = 2 * (config.max_rounds + 3)
    return graph.invoke({"question": question, "round": 0, "replies": []}, {"recursion_limit": limit})
