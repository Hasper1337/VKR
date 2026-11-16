"""Веб-интерфейс пилотной версии. Запуск: streamlit run app.py"""
from __future__ import annotations

import json

import streamlit as st

from debate.graph import run_debate
from debate.schemas import DebateConfig
from debate.storage import Storage, costs

st.set_page_config(page_title="Коллективные дебаты LLM", layout="wide")
st.title("Коллективные дебаты языковых моделей")

with st.sidebar:
    st.header("Конфигурация")
    with open("config.example.json", encoding="utf-8") as handle:
        default_config = handle.read()
    config_text = st.text_area("Параметры протокола (JSON)", default_config, height=360)
    db_path = st.text_input("Файл протоколов", "debates.sqlite3")

question = st.text_input("Вопрос")

if st.button("Провести дебаты", disabled=not question.strip()):
    try:
        config = DebateConfig.model_validate(json.loads(config_text))
    except ValueError as error:
        st.error(f"Ошибка конфигурации: {error}")
        st.stop()
    with st.spinner("Агенты обсуждают вопрос"):
        try:
            state = run_debate(question, config)
        except Exception as error:  # сбой подключения к модели показывается пользователю
            st.error(f"Не удалось провести дебаты: {error}")
            st.stop()
    st.subheader("Итоговый ответ")
    st.success(state["final_answer"])
    st.caption(state.get("final_rationale", ""))
    cost = costs(state)
    columns = st.columns(4)
    columns[0].metric("Раундов обсуждения", state["round"])
    columns[1].metric("Обращений к моделям", cost["calls"])
    columns[2].metric("Токенов", cost["tokens_in"] + cost["tokens_out"])
    columns[3].metric("Секунд", f"{cost['seconds']:.1f}")
    st.caption(f"Причина останова: {state['stop_reason']}")
    st.subheader("Протокол")
    for number in sorted({r["round"] for r in state["replies"]}):
        with st.expander(f"Раунд {number}", expanded=True):
            st.table([
                {"Агент": r["agent"], "Роль": r["role"], "Модель": r["model"],
                 "Ответ": r["answer"], "Обоснование": r["rationale"],
                 "Уверенность": round(r["confidence"], 2)}
                for r in state["replies"] if r["round"] == number
            ])
    storage = Storage(db_path)
    storage.save(state, "debate", config.model_dump_json())
    storage.close()
