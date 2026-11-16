"""Протоколирование сеансов в базе данных SQLite."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .schemas import DebateState

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    method TEXT NOT NULL,
    config_json TEXT NOT NULL,
    question TEXT NOT NULL,
    final_answer TEXT NOT NULL,
    stop_reason TEXT NOT NULL,
    rounds INTEGER NOT NULL,
    calls INTEGER NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    seconds REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    round INTEGER NOT NULL,
    agent TEXT NOT NULL,
    role TEXT NOT NULL,
    model TEXT NOT NULL,
    answer TEXT NOT NULL,
    rationale TEXT NOT NULL,
    confidence REAL NOT NULL,
    raw TEXT NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    seconds REAL NOT NULL
);
"""


def costs(state: DebateState) -> dict:
    """Показатели затрат сеанса: обращения, токены, время."""
    replies = state.get("replies", [])
    return {
        "calls": len(replies),
        "tokens_in": sum(r["tokens_in"] for r in replies),
        "tokens_out": sum(r["tokens_out"] for r in replies),
        "seconds": sum(r["seconds"] for r in replies),
    }


class Storage:
    """Хранилище протоколов дебатов."""

    def __init__(self, path: str = "debates.sqlite3") -> None:
        self.connection = sqlite3.connect(path)
        self.connection.executescript(SCHEMA)

    def save(self, state: DebateState, method: str, config_json: str = "{}") -> int:
        """Сохраняет сеанс и все его сообщения, возвращает идентификатор сеанса."""
        cost = costs(state)
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO sessions (started_at, method, config_json, question, final_answer, "
                "stop_reason, rounds, calls, tokens_in, tokens_out, seconds) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"), method, config_json,
                    state["question"], state.get("final_answer", ""), state.get("stop_reason", ""),
                    state.get("round", 0), cost["calls"], cost["tokens_in"], cost["tokens_out"],
                    cost["seconds"],
                ),
            )
            session_id = int(cursor.lastrowid)
            self.connection.executemany(
                "INSERT INTO messages (session_id, round, agent, role, model, answer, rationale, "
                "confidence, raw, tokens_in, tokens_out, seconds) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (session_id, r["round"], r["agent"], r["role"], r["model"], r["answer"],
                     r["rationale"], r["confidence"], r["raw"], r["tokens_in"], r["tokens_out"],
                     r["seconds"])
                    for r in state.get("replies", [])
                ],
            )
        return session_id

    def close(self) -> None:
        self.connection.close()
