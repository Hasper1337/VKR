"""Мультиагентная система повышения фактической точности ответов LLM на основе дебатов."""
from .graph import build_graph, run_debate
from .schemas import AgentConfig, DebateConfig

__all__ = ["AgentConfig", "DebateConfig", "build_graph", "run_debate"]
